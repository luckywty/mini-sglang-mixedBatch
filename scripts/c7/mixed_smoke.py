"""Small baseline/mixed model run with an inspectable forward/output trace.

Use independent processes for each flag combination. No performance assertions.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--disable-overlap", action="store_true")
    parser.add_argument("--graph-bs", nargs="*", type=int, default=[])
    args = parser.parse_args()
    os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = str(int(args.disable_overlap))

    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        max_seq_len_override=256,
        max_running_req=2,
        max_extend_tokens=16,
        num_page_override=512,
        page_size=1,
        attention_backend="fi",
        cuda_graph_bs=args.graph_bs,
        enable_mixed_batching=args.mixed,
    )
    forward, send = llm.engine.forward_batch, llm.send_result
    trace, replies = [], []

    def trace_forward(batch, sample_args):
        layout = batch.layout
        row = {
            "phase": batch.phase,
            "uids": [req.uid for req in batch.reqs],
            "query_lens": [req.extend_len for req in batch.reqs],
            "sample_mask": list(layout.sample_mask) if layout is not None else None,
            "graph": llm.engine.graph_runner.can_use_cuda_graph(batch),
        }
        output = forward(batch, sample_args)
        row["sample_count"] = output.next_tokens_gpu.numel()
        if layout is not None:
            assert row["sample_count"] == sum(layout.sample_mask)
        trace.append(row)
        print("C7_FORWARD " + json.dumps(row), flush=True)
        return output

    def record_replies(messages):
        replies.extend(messages)
        send(messages)

    llm.engine.forward_batch, llm.send_result = trace_forward, record_replies
    try:
        results = llm.generate(
            ["The capital of France is", "Repeat the word hello. " * 12],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
        )
        assert all(len(result["token_ids"]) == 4 for result in results)
        if args.mixed:
            assert any(row["phase"] == "mixed" for row in trace)
            for uid in (0, 1):
                output = [message for message in replies if message.uid == uid]
                assert len(output) == 4
                assert [message.finished for message in output] == [False, False, False, True]
            assert not llm.mixed_lifecycle.active
        assert not llm.decode_manager.runnable and not llm.prefill_manager.runnable
        assert llm.table_manager.available_size == 2
        llm.cache_manager.check_integrity()
        print("C7_OUTPUT " + json.dumps(results, ensure_ascii=False), flush=True)
        print("C7_MODEL_SMOKE PASS " + json.dumps(vars(args)), flush=True)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
