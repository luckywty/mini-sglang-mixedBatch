"""Matched-input numerical validation; extra forwards exist only in this test.

Attention is checked against explicit FP32 CPU GQA/causal math. Full forward
reference uses the disabled baseline operators with the same packed rows/KV.
LMHead has an additional FP32 CPU projection reference using real hidden states.
This intentionally synchronizes for observation, and is separate from overlap tests.
"""

from __future__ import annotations

import json
from copy import copy

from stage4_matrix import load_driver


def main() -> None:
    Driver, torch, _ = load_driver()
    torch.set_num_threads(4)
    driver = Driver(graph_bs=[], budget=7)
    driver.overlap = False
    backend = driver.engine.attn_backend
    model = driver.engine.model
    original_attn, original_model, original_head = (
        backend.forward,
        model.forward,
        model.lm_head.forward,
    )
    in_reference = False
    captures, records = [], []
    forward_id = 0
    projection = model.lm_head.tied_embedding or model.lm_head
    weight = projection.weight.detach().to(device="cpu", dtype=torch.float32)
    head_record = None

    def compare(actual, reference, *, kind, atol, rtol, **metadata):
        actual, reference = actual.float().cpu(), reference.float().cpu()
        assert actual.shape == reference.shape
        if not actual.numel():
            return
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
        error = (actual - reference).abs()
        violations = int((error > atol + rtol * reference.abs()).sum())
        row = dict(
            kind=kind,
            forward=forward_id,
            max_abs=error.max().item(),
            abs_p50_p95_p99=error.flatten().quantile(torch.tensor([0.5, 0.95, 0.99])).tolist(),
            violations=violations,
            atol=atol,
            rtol=rtol,
            **metadata,
        )
        records.append(row)
        print("C7_STAGE4_NUMERIC " + json.dumps(row), flush=True)
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)

    def attention(q, k, v, layer, batch):
        result = original_attn(q, k, v, layer, batch)
        if not in_reference:
            lengths = [(req.extend_len, req.device_len, req.cached_len) for req in batch.reqs]
            indices = [
                driver.engine.page_table[req.table_idx, : req.device_len].clone()
                for req in batch.reqs
            ]
            # Logical page lookup is independent of FlashInfer's packed indptr/indices.
            expected_indices = torch.cat(indices)
            torch.testing.assert_close(batch.attn_metadata.indices, expected_indices)
            logical = [
                (
                    driver.engine.kv_cache.k_cache(layer)[idx, 0].clone(),
                    driver.engine.kv_cache.v_cache(layer)[idx, 0].clone(),
                )
                for idx in indices
            ]
            captures.append((layer, lengths, q.clone(), logical, result.clone()))
        return result

    def head(hidden):
        nonlocal head_record
        batch = driver.engine.ctx.batch
        if not in_reference:
            selected, uid_rows, offset = [], [], 0
            # Independent sampling decision: did this row reach its FULL input?
            for req in batch.reqs:
                length = req.device_len - req.cached_len
                if req.device_len >= len(driver.prompts[req.uid]):
                    selected.append(offset + length - 1)
                    uid_rows.append(req.uid)
                offset += length
            assert tuple(uid_rows) == tuple(
                batch.reqs[i].uid for i in batch.layout.sample_req_indices
            )
            head_record = (hidden[selected].clone(), selected)
        return original_head(hidden)

    def forward():
        nonlocal in_reference, head_record, forward_id
        batch = driver.engine.ctx.batch
        expected_positions = torch.cat(
            [torch.arange(req.cached_len, req.device_len) for req in batch.reqs]
        )
        assert batch.positions.cpu().tolist() == expected_positions.tolist()
        assert batch.out_loc.unique().numel() == expected_positions.numel()
        history = torch.cat(
            [driver.engine.page_table[req.table_idx, : req.cached_len] for req in batch.reqs]
        )
        assert not torch.isin(batch.out_loc, history).any().item()
        captures.clear()
        result = original_model()
        saved_head = head_record
        # A complete reference model forward, with the same packed inputs and
        # history; no Req advance or sampler occurs in this auxiliary execution.
        ref = copy(batch)
        ref.layout = None
        ref.phase = "decode" if batch.is_decode else "prefill"
        ref.padded_reqs = ref.reqs
        ref_indices = [
            i for i, req in enumerate(batch.reqs) if req.device_len >= len(driver.prompts[req.uid])
        ]
        backend.prepare_metadata(ref)
        try:
            in_reference = True
            driver.engine.ctx._batch = ref
            reference_logits = original_model()[ref_indices].clone()
        finally:
            driver.engine.ctx._batch = batch
            in_reference = False
        compare(
            result,
            reference_logits,
            kind="matched-full-forward-logits",
            atol=0.1,
            rtol=0.01,
            phase=batch.phase,
        )
        if saved_head[0].shape[0]:
            cpu_hidden = saved_head[0].float().cpu()
            cpu_logits = torch.nn.functional.linear(cpu_hidden, weight)
            compare(
                result,
                cpu_logits,
                kind="fp32-head-projection",
                atol=0.1,
                rtol=0.01,
                phase=batch.phase,
            )
        for layer, lengths, q, logical, actual in captures:
            q = q.float().cpu()
            offset, pieces = 0, []
            for (q_len, kv_len, cached), (keys, values) in zip(lengths, logical):
                keys, values = keys.float().cpu(), values.float().cpu()
                repeat = q.shape[1] // keys.shape[1]
                keys = keys.repeat_interleave(repeat, dim=1).transpose(0, 1)
                values = values.repeat_interleave(repeat, dim=1).transpose(0, 1)
                query = q[offset : offset + q_len].transpose(0, 1)
                logits = query @ keys.transpose(1, 2) / q.shape[-1] ** 0.5
                mask = torch.arange(kv_len)[None, :] <= cached + torch.arange(q_len)[:, None]
                pieces.append(
                    (logits.masked_fill(~mask, float("-inf")).softmax(-1) @ values).transpose(0, 1)
                )
                offset += q_len
            compare(
                actual,
                torch.cat(pieces),
                kind="fp32-attention",
                atol=0.02,
                rtol=0.02,
                layer=layer,
                phase=batch.phase,
            )
        forward_id += 1
        return result

    backend.forward, model.forward, model.lm_head.forward = attention, forward, head
    try:
        cases = [
            ("cold-mixed", [(1, [100, 101, 102], 4), (2, list(range(200, 218)), 2)], False),
            (
                "multiple-prefills",
                [(3, list(range(300, 305)), 3), (4, list(range(400, 411)), 2), (5, [500, 501], 1)],
                False,
            ),
            ("prefix-seed", [(6, list(range(600, 614)), 2)], False),
            ("prefix-hit", [(7, list(range(600, 614)), 3)], True),
        ]
        for name, requests, keep in cases:
            driver.reset(overlap=False, keep_cache=keep)
            for uid, prompt, count in requests:
                driver.add(uid, prompt, count)
            driver.drain()
            driver.check_resources()
            if name == "prefix-hit":
                assert driver.traces[0]["cached_lens"] == (13,)
            print(
                "C7_STAGE4_NUMERIC_CASE "
                + json.dumps(
                    {
                        "case": name,
                        "forwards": len(driver.traces),
                        "tokens": {uid: driver.accepted_tokens[uid] for uid, _, _ in requests},
                    }
                ),
                flush=True,
            )
        assert any(record["phase"] == "mixed" for record in records)
        summary = {
            "comparisons": len(records),
            "forwards": forward_id,
            "violations": sum(row["violations"] for row in records),
            "groups": {
                kind: {
                    "count": sum(row["kind"] == kind for row in records),
                    "max_abs": max(row["max_abs"] for row in records if row["kind"] == kind),
                }
                for kind in {row["kind"] for row in records}
            },
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        print("C7_STAGE4_NUMERIC_PASS " + json.dumps(summary), flush=True)
    finally:
        driver.shutdown()


if __name__ == "__main__":
    main()
