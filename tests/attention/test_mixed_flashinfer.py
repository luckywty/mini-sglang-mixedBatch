"""Real GPU attention tests; no model, Engine, sampler or serving execution.

Run separately from the CPU scheduler suite. Tolerances were fixed in stage 1.
The oracle uses logical BF16 inputs and explicit FP32 CPU attention, independently
of FlashInfer metadata, physical cache gathers and FlashInfer's causal mask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from itertools import accumulate
from types import SimpleNamespace

import minisgl.core as core
import minisgl.distributed.info as distributed_info
import pytest
import torch
from minisgl.attention.fi import FlashInferBackend
from minisgl.core import Batch, BatchLayout, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.kvcache.naive_cache import NaiveCacheHandle
from minisgl.layers.attention import AttentionLayer
from minisgl.message import UserMsg
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.mixed import schedule_mixed_batch
from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
from minisgl.scheduler.table import TableManager

ATOL = RTOL = 2e-2
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128  # Qwen3-0.6B; TP=1
NUM_LAYERS, NUM_PAGES, MAX_CONTEXT = 2, 8192, 4096
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA + FlashInfer")


@pytest.fixture
def harness(monkeypatch):
    torch.cuda.set_device(0)
    assert torch.cuda.is_bf16_supported()
    torch.cuda.reset_peak_memory_stats()
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 1))
    ctx = core.Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    ctx.page_table = torch.full((9, MAX_CONTEXT), -1, dtype=torch.int32, device="cuda")
    ctx.kv_cache = MHAKVCache(
        KV_HEADS, NUM_LAYERS, HEAD_DIM, NUM_PAGES, 1, torch.bfloat16, torch.device("cuda:0")
    )
    config = ModelConfig(
        num_layers=NUM_LAYERS,
        num_qo_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=1024,
        vocab_size=151936,
        intermediate_size=3072,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(HEAD_DIM, HEAD_DIM, MAX_CONTEXT, 1e6, None),
        hidden_act="silu",
        tie_word_embeddings=True,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
    )
    ctx.attn_backend = backend = FlashInferBackend(config)
    with torch.inference_mode():
        yield SimpleNamespace(ctx=ctx, backend=backend)
    torch.cuda.synchronize()
    print(
        "C7_MEMORY "
        + json.dumps(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        ),
        flush=True,
    )


@dataclass
class Case:
    batch: Batch
    qkv_cpu: list[torch.Tensor]
    qkv_gpu: list[torch.Tensor]
    keys: list[list[torch.Tensor]]
    values: list[list[torch.Tensor]]
    physical: list[torch.Tensor]

    def qkv(self, layer):
        q, k, v = self.qkv_gpu[layer].split(
            [Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1
        )
        return q.view(-1, Q_HEADS, HEAD_DIM), k, v


def set_packed_indices(harness, batch):
    layout = batch.layout
    batch.padded_reqs = batch.reqs
    batch.positions = torch.cat(
        [
            torch.arange(cached, kv, dtype=torch.int32)
            for cached, kv in zip(layout.cached_lens, layout.kv_lens)
        ]
    ).to("cuda")
    batch.out_loc = torch.cat(
        [
            harness.ctx.page_table[slot, cached:kv]
            for slot, cached, kv in zip(layout.table_indices, layout.cached_lens, layout.kv_lens)
        ]
    )


def populate_case(harness, batch, *, seed=20261028, shared_prefix=0, previous=None):
    """Fill historical KV only; new KV locations remain NaN until real store_kv."""
    set_packed_indices(harness, batch)
    generator = torch.Generator().manual_seed(seed)
    layout = batch.layout
    physical = [
        harness.ctx.page_table[slot, :kv].clone()
        for slot, kv in zip(layout.table_indices, layout.kv_lens)
    ]
    if previous is None:
        harness.ctx.kv_cache._kv_buffer.fill_(float("nan"))
    qkv_cpu, qkv_gpu, keys, values = [], [], [], []
    for layer in range(NUM_LAYERS):
        packed = torch.randn(
            layout.num_tokens, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM, generator=generator
        ).to(torch.bfloat16)
        qkv_cpu.append(packed)
        qkv_gpu.append(packed.to("cuda"))
        _, new_k, new_v = packed.split(
            [Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1
        )
        new_k, new_v = new_k.view(-1, KV_HEADS, HEAD_DIM), new_v.view(-1, KV_HEADS, HEAD_DIM)
        shared_k = torch.randn(shared_prefix, KV_HEADS, HEAD_DIM, generator=generator).to(
            torch.bfloat16
        )
        shared_v = torch.randn(shared_prefix, KV_HEADS, HEAD_DIM, generator=generator).to(
            torch.bfloat16
        )
        layer_keys, layer_values = [], []
        offset = 0
        for i, (q_len, kv_len, cached_len) in enumerate(
            zip(layout.query_lens, layout.kv_lens, layout.cached_lens)
        ):
            k = torch.randn(kv_len, KV_HEADS, HEAD_DIM, generator=generator).to(torch.bfloat16)
            v = (
                torch.randn(kv_len, KV_HEADS, HEAD_DIM, generator=generator) + (i + layer) * 0.5
            ).to(torch.bfloat16)
            k[:shared_prefix], v[:shared_prefix] = shared_k, shared_v
            k[cached_len:], v[cached_len:] = (
                new_k[offset : offset + q_len],
                new_v[offset : offset + q_len],
            )
            if previous is not None:
                # Keep actual previous GPU KV, including the just-computed chunk.
                assert cached_len == previous.batch.layout.kv_lens[i]
                torch.testing.assert_close(physical[i][:cached_len], previous.physical[i])
                k[:cached_len], v[:cached_len] = previous.keys[layer][i], previous.values[layer][i]
            layer_keys.append(k)
            layer_values.append(v)
            if previous is None:
                harness.ctx.kv_cache.k_cache(layer)[physical[i][:cached_len], 0] = k[
                    :cached_len
                ].to("cuda")
                harness.ctx.kv_cache.v_cache(layer)[physical[i][:cached_len], 0] = v[
                    :cached_len
                ].to("cuda")
            offset += q_len
        keys.append(layer_keys)
        values.append(layer_values)
    return Case(batch, qkv_cpu, qkv_gpu, keys, values, physical)


def make_case(harness, query_lens, kv_lens, phases, *, seed=20261028, shared_prefix=0):
    generator = torch.Generator().manual_seed(seed + 1)
    pages = torch.randperm(NUM_PAGES, generator=generator).to("cuda", dtype=torch.int32)
    shared, cursor = pages[:shared_prefix], shared_prefix
    reqs = []
    for i, (q_len, kv_len, phase) in enumerate(zip(query_lens, kv_lens, phases)):
        slot = [5, 1, 7, 3][i]  # non-contiguous request slots, independent of UID/order
        cls = Req if phase == "decode" else ChunkedReq
        req = cls(
            torch.zeros(kv_len, dtype=torch.int32),
            slot,
            kv_len - q_len,
            8,
            100 + i * 10,
            SamplingParams(),
            NaiveCacheHandle(),
        )
        tail_len = kv_len - shared_prefix
        harness.ctx.page_table[slot, :kv_len] = torch.cat(
            [shared, pages[cursor : cursor + tail_len]]
        )
        cursor += tail_len
        reqs.append(req)
    layout = BatchLayout.capture(reqs, tuple(phases))
    phase = "mixed" if set(phases) == {"prefill", "decode"} else phases[0]
    return populate_case(
        harness, Batch(reqs, phase, layout), seed=seed, shared_prefix=shared_prefix
    )


def reference(case, layer):
    """Per-request GQA with an explicit right-aligned causal mask, all in FP32 CPU."""
    layout, outputs = case.batch.layout, []
    query = case.qkv_cpu[layer][:, : Q_HEADS * HEAD_DIM].view(-1, Q_HEADS, HEAD_DIM).float()
    offset = 0
    for i, (q_len, kv_len, cached_len) in enumerate(
        zip(layout.query_lens, layout.kv_lens, layout.cached_lens)
    ):
        q = query[offset : offset + q_len].transpose(0, 1)
        k = (
            case.keys[layer][i]
            .float()
            .repeat_interleave(Q_HEADS // KV_HEADS, dim=1)
            .transpose(0, 1)
        )
        v = (
            case.values[layer][i]
            .float()
            .repeat_interleave(Q_HEADS // KV_HEADS, dim=1)
            .transpose(0, 1)
        )
        logits = q @ k.transpose(1, 2) / HEAD_DIM**0.5
        mask = torch.arange(kv_len)[None, :] <= cached_len + torch.arange(q_len)[:, None]
        outputs.append(
            (logits.masked_fill(~mask, float("-inf")).softmax(dim=-1) @ v).transpose(0, 1)
        )
        offset += q_len
    return torch.cat(outputs)


def assert_numeric(actual, expected, name):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    assert torch.isfinite(actual).all(), name
    error = (actual - expected).abs()
    relative = error / expected.abs().clamp_min(1e-6)
    allowed = ATOL + RTOL * expected.abs()
    stats = {
        "name": name,
        "max_abs": error.max().item(),
        "abs_p50_p95_p99": error.flatten().quantile(torch.tensor([0.5, 0.95, 0.99])).tolist(),
        "max_relative_floor_1e-6": relative.max().item(),
        "violations": int((error > allowed).sum()),
        "atol": ATOL,
        "rtol": RTOL,
    }
    print("C7_NUMERIC " + json.dumps(stats), flush=True)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL, msg=name)


def assert_metadata(harness, case):
    batch, layout = case.batch, case.batch.layout
    harness.backend.prepare_metadata(batch)
    metadata = batch.attn_metadata
    assert metadata.wrapper is harness.backend.prefill_wrapper
    assert not metadata.wrapper.is_cuda_graph_enabled
    assert metadata.cu_seqlens_q_cpu.dtype == metadata.cu_seqlens_k_cpu.dtype == torch.int32
    assert metadata.cu_seqlens_q_cpu.tolist() == list(layout.qo_indptr)
    assert metadata.cu_seqlens_k_cpu.tolist() == [0, *accumulate(layout.kv_lens)]
    assert metadata.seq_lens_cpu.tolist() == list(layout.kv_lens)
    assert metadata.last_page_len_cpu.tolist() == [1] * batch.size
    assert metadata.page_size == 1 and metadata.pos_encoding_mode == "NONE"
    assert metadata.num_qo_heads == Q_HEADS and metadata.num_kv_heads == KV_HEADS
    torch.testing.assert_close(metadata.indices, torch.cat(case.physical))
    assert int(metadata.indices.min()) >= 0 and int(metadata.indices.max()) < NUM_PAGES
    expected_write = torch.cat(
        [indices[cached:] for indices, cached in zip(case.physical, layout.cached_lens)]
    )
    torch.testing.assert_close(batch.out_loc, expected_write)
    assert batch.out_loc.unique().numel() == layout.num_tokens


CASES = [
    (
        (1, 1, 1, 125),
        (501, 1001, 2001, 125),
        ("decode", "decode", "decode", "prefill"),
        0,
        20261028,
    ),
    ((1, 1, 1, 125), (501, 1001, 2001, 125), ("decode", "decode", "decode", "prefill"), 0, 7),
    ((1, 7), (35, 73), ("decode", "prefill"), 0, 11),
    ((1, 1, 5, 11), (9, 24, 45, 11), ("decode", "decode", "prefill", "prefill"), 0, 17),
    ((1, 1, 1), (13, 49, 17), ("decode", "decode", "prefill"), 0, 23),
    ((1, 3, 7), (64, 55, 70), ("decode", "prefill", "prefill"), 16, 29),
]


@pytest.mark.parametrize("query_lens,kv_lens,phases,shared_prefix,seed", CASES)
def test_mixed_matches_fp32_and_individual_prefills(
    harness, query_lens, kv_lens, phases, shared_prefix, seed
):
    case = make_case(harness, query_lens, kv_lens, phases, seed=seed, shared_prefix=shared_prefix)
    assert_metadata(harness, case)
    outputs = []
    for layer in range(NUM_LAYERS):
        q, k, v = case.qkv(layer)
        assert not k.is_contiguous() and not v.is_contiguous()  # real packed QKV strides
        output = harness.backend.forward(q, k, v, layer, case.batch)
        outputs.append(output.cpu())
        assert_numeric(output, reference(case, layer), f"mixed/seed={seed}/layer={layer}")
        for i, cached_len in enumerate(case.batch.layout.cached_lens):
            indices = case.physical[i]
            torch.testing.assert_close(
                harness.ctx.kv_cache.k_cache(layer)[indices, 0].cpu(),
                case.keys[layer][i],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                harness.ctx.kv_cache.v_cache(layer)[indices, 0].cpu(),
                case.values[layer][i],
                atol=0,
                rtol=0,
            )
    assert not torch.equal(outputs[0], outputs[1])  # different layers have different logical KV

    # A second, real FlashInfer path: one request per pure prefill wrapper plan/run.
    q, k, v = case.qkv(0)
    pieces, offset = [], 0
    for req, length in zip(case.batch.reqs, query_lens):
        single = Batch([req], "prefill")
        single.padded_reqs = single.reqs
        single.positions = case.batch.positions[offset : offset + length]
        single.out_loc = case.batch.out_loc[offset : offset + length]
        harness.backend.prepare_metadata(single)
        pieces.append(
            harness.backend.forward(
                q[offset : offset + length],
                k[offset : offset + length],
                v[offset : offset + length],
                0,
                single,
            ).cpu()
        )
        offset += length
    assert_numeric(outputs[0], torch.cat(pieces), f"mixed-vs-individual/seed={seed}")


def test_history_perturbation_cannot_cross_requests(harness):
    case = make_case(harness, (1, 5), (31, 47), ("decode", "prefill"))
    assert_metadata(harness, case)
    q, k, v = case.qkv(0)
    before = harness.backend.forward(q, k, v, 0, case.batch).clone()
    cached_len = case.batch.layout.cached_lens[0]
    changed = case.physical[0][:cached_len]
    harness.ctx.kv_cache.v_cache(0)[changed, 0] += 4
    case.values[0][0][:cached_len] += 4
    after = harness.backend.forward(q, k, v, 0, case.batch)
    torch.testing.assert_close(after[1:], before[1:], atol=0, rtol=0)
    assert (after[0] - before[0]).abs().max() > 1
    assert_numeric(after, reference(case, 0), "history-perturbation/no-cross-request")


def test_causal_mask_excludes_future_chunk_tokens(harness):
    case = make_case(harness, (1, 3), (5, 7), ("decode", "prefill"))
    # Uniform attention gives an analytic oracle; only the final query may see V=100.
    case.qkv_cpu[0].zero_()
    value_offset = (Q_HEADS + KV_HEADS) * HEAD_DIM
    case.qkv_cpu[0][-1, value_offset:] = 100
    case.qkv_gpu[0].copy_(case.qkv_cpu[0])
    for i in range(2):
        case.keys[0][i].zero_()
        case.values[0][i].zero_()
        harness.ctx.kv_cache.k_cache(0)[case.physical[i], 0] = 0
        harness.ctx.kv_cache.v_cache(0)[case.physical[i], 0] = 0
    case.values[0][1][-1] = 100
    assert_metadata(harness, case)
    actual = harness.backend.forward(*case.qkv(0), 0, case.batch)
    assert torch.count_nonzero(actual[:-1]) == 0
    assert_numeric(actual, reference(case, 0), "causal/future-value-100")
    torch.testing.assert_close(
        actual[-1].float(), torch.full_like(actual[-1].float(), 100 / 7), atol=ATOL, rtol=RTOL
    )


def test_plan_once_per_batch_and_sequential_workspace_reuse(harness, monkeypatch):
    calls = []
    plan = harness.backend.prefill_wrapper.plan

    def record_plan(*args, **kwargs):
        calls.append(kwargs["qo_indptr"].clone())
        return plan(*args, **kwargs)

    monkeypatch.setattr(harness.backend.prefill_wrapper, "plan", record_plan)
    # The engine submits every layer of A before B. Metadata never replays an old
    # initialized wrapper plan after another batch has overwritten the wrapper.
    for round_id, (q_lens, kv_lens) in enumerate(
        [((1, 9), (31, 23)), ((1, 1), (9, 41)), ((1, 125), (501, 125))]
    ):
        case = make_case(harness, q_lens, kv_lens, ("decode", "prefill"), seed=round_id)
        assert_metadata(harness, case)
        assert not case.batch.attn_metadata.initialized
        for layer in range(NUM_LAYERS):
            output = harness.backend.forward(*case.qkv(layer), layer, case.batch)
            assert_numeric(
                output, reference(case, layer), f"plan-reuse/round={round_id}/layer={layer}"
            )
            assert len(calls) == round_id + 1
    assert [values.tolist() for values in calls] == [[0, 1, 10], [0, 1, 2], [0, 1, 126]]


def test_real_radix_hit_and_chunk_continuation_to_attention(harness):
    table = TableManager(8, harness.ctx.page_table)
    cache = CacheManager(NUM_PAGES, 1, harness.ctx.page_table, "radix")
    # Make physical mapping non-contiguous even for the real allocator.
    cache.free_slots = torch.randperm(NUM_PAGES, generator=torch.Generator().manual_seed(99)).to(
        "cuda", dtype=torch.int32
    )
    decode = DecodeManager(1)
    prefill = PrefillManager(cache, table, decode)
    prompt = torch.arange(100, 140, dtype=torch.int32)
    prefix_indices = cache._allocate(32)
    cache.prefix_cache.insert_prefix(prompt[:32], prefix_indices)
    active = Req(
        torch.arange(11, dtype=torch.int32),
        table.allocate(),
        10,
        8,
        1,
        SamplingParams(),
        cache.prefix_cache.match_prefix(prompt[:0]).cuda_handle,
    )
    harness.ctx.page_table[active.table_idx, :10] = cache._allocate(10)
    decode.running_reqs.add(active)
    prefill.add_one_req(UserMsg(2, prompt, SamplingParams(max_tokens=8)))
    previous = None
    for round_id, (q_len, kv_len) in enumerate([(7, 39), (1, 40)]):
        batch = schedule_mixed_batch(decode, prefill, total_token_budget=8)
        assert batch.layout.query_lens == (1, q_len)
        assert batch.layout.kv_lens == (11 + round_id, kv_len)
        assert batch.layout.sample_mask == (True, round_id == 1)
        assert batch.reqs[1].cache_handle.cached_len == 32
        torch.testing.assert_close(
            harness.ctx.page_table[batch.reqs[1].table_idx, :32], prefix_indices
        )
        cache.allocate_paged(batch.reqs)
        case = populate_case(harness, batch, seed=round_id + 101, previous=previous)
        assert_metadata(harness, case)
        for layer in range(NUM_LAYERS):
            output = harness.backend.forward(*case.qkv(layer), layer, batch)
            assert_numeric(
                output, reference(case, layer), f"radix-hit/round={round_id}/layer={layer}"
            )
        # Only emulate state advancement. No sampling/Engine/output-path claim.
        for req in batch.reqs:
            req.complete_one()
        decode.filter_reqs(batch.reqs)
        previous = case
    assert not prefill.runnable
    assert len(decode.running_reqs) == 2


def test_back_to_back_engine_stream_batches_with_deferred_checks(harness):
    prepared = []
    specifications = [
        ((1, 1, 1, 125), (501, 1001, 2001, 125), ("decode", "decode", "decode", "prefill")),
        ((1, 1), (9, 41), ("decode", "decode")),
        ((1, 3, 7), (64, 55, 70), ("prefill", "prefill", "prefill")),
        ((1, 9), (31, 23), ("decode", "prefill")),
    ]
    for i, (q_lens, kv_lens, phases) in enumerate(specifications):
        case = make_case(harness, q_lens, kv_lens, phases, seed=i + 300)
        harness.backend.prepare_metadata(case.batch)
        prepared.append(
            (
                case,
                harness.ctx.kv_cache._kv_buffer.clone(),
                [reference(case, layer) for layer in range(NUM_LAYERS)],
            )
        )
    engine_stream = torch.cuda.Stream()
    engine_stream.wait_stream(torch.cuda.current_stream())
    pending = []
    with torch.cuda.stream(engine_stream):
        for case, saved_cache, expected in prepared:
            # GPU copies and all layers execute in the same stream as real Engine.
            harness.ctx.kv_cache._kv_buffer.copy_(saved_cache)
            for layer in range(NUM_LAYERS):
                output = harness.backend.forward(*case.qkv(layer), layer, case.batch)
                pending.append((output.clone(), expected[layer], layer))
            # No .cpu(), .item(), or explicit synchronize between batches.
    torch.cuda.current_stream().wait_stream(engine_stream)
    for i, (actual, expected, layer) in enumerate(pending):
        assert_numeric(actual, expected, f"deferred/forward={i}/layer={layer}")


def test_attention_layer_uses_absolute_rope_positions(harness):
    case = make_case(harness, (1, 7), (35, 73), ("decode", "prefill"), seed=401)
    assert_metadata(harness, case)
    # Independent NeoX-style RoPE reference; historical K is already post-RoPE.
    layout = case.batch.layout
    positions = torch.cat(
        [torch.arange(cached, kv) for cached, kv in zip(layout.cached_lens, layout.kv_lens)]
    )
    torch.testing.assert_close(case.batch.positions.cpu().long(), positions)
    frequencies = (
        positions.float()[:, None]
        * (1e6 ** (-torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))[None, :]
    )

    def rotate(x):
        first, second = x.float().chunk(2, dim=-1)
        cos, sin = frequencies.cos()[:, None, :], frequencies.sin()[:, None, :]
        return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1).to(
            torch.bfloat16
        )

    packed = case.qkv_cpu[0]
    query, key, _ = packed.split(
        [Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1
    )
    # Compute both rotations before changing the packed reference inputs.
    rotated_q = rotate(query.view(-1, Q_HEADS, HEAD_DIM)).flatten(1)
    rotated_k = rotate(key.view(-1, KV_HEADS, HEAD_DIM))
    packed[:, : Q_HEADS * HEAD_DIM] = rotated_q
    offset = 0
    for i, (q_len, cached_len) in enumerate(zip(layout.query_lens, layout.cached_lens)):
        case.keys[0][i][cached_len:] = rotated_k[offset : offset + q_len]
        offset += q_len
    with torch.device("cuda"):
        layer = AttentionLayer(0, Q_HEADS, KV_HEADS, HEAD_DIM, harness.backend.config.rotary_config)
    with harness.ctx.forward_batch(case.batch):
        output = layer.forward(case.qkv_gpu[0]).view(-1, Q_HEADS, HEAD_DIM)
    assert_numeric(output, reference(case, 0), "attention-layer/absolute-rope-positions")


@pytest.mark.parametrize(
    "invalid",
    [
        "missing-layout",
        "padding",
        "stale-request",
        "phase",
        "slot",
        "kv-bound",
        "positions",
        "out-loc",
        "page-size",
        "tp",
    ],
)
def test_invalid_mixed_layout_rejected_before_kernel(harness, monkeypatch, invalid):
    case = make_case(harness, (1, 3), (11, 7), ("decode", "prefill"))
    batch = case.batch
    if invalid == "missing-layout":
        batch.layout = None
    elif invalid == "padding":
        batch.padded_reqs = batch.reqs + [batch.reqs[0]]
    elif invalid == "stale-request":
        batch.reqs[0].complete_one()
    elif invalid == "phase":
        batch.layout = replace(batch.layout, req_phases=("prefill", "prefill"))
    elif invalid in ("slot", "kv-bound"):
        req = batch.reqs[1]
        if invalid == "slot":
            req.table_idx = harness.ctx.page_table.shape[0]
        else:
            req.device_len = MAX_CONTEXT + 1
        batch.layout = BatchLayout.capture(batch.reqs, ("decode", "prefill"))
    elif invalid == "positions":
        batch.positions = batch.positions[:-1]
    elif invalid == "out-loc":
        batch.out_loc = batch.out_loc[:-1]
    elif invalid == "page-size":
        harness.ctx.page_size = 4
    elif invalid == "tp":
        monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 2))
    with pytest.raises(ValueError):
        harness.backend.prepare_metadata(batch)
