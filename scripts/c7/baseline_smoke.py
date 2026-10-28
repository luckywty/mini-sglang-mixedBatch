"""Small baseline-only smoke test; run each configuration in a fresh process.

Use PYTHONPATH=$PWD/python with an existing compatible Python environment.
This deliberately does not benchmark or enable the unfinished mixed executor.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--graph-bs", nargs="*", type=int, default=[])
    parser.add_argument("--disable-overlap", action="store_true")
    parser.add_argument("--chunk-budget", type=int, default=16)
    args = parser.parse_args()
    os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = str(int(args.disable_overlap))

    import minisgl.core
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    print("source:", minisgl.core.__file__, flush=True)
    print("configuration:", vars(args), flush=True)
    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        max_seq_len_override=256,
        max_running_req=2,
        max_extend_tokens=args.chunk_budget,
        num_page_override=512,
        page_size=1,
        attention_backend="fi",
        cuda_graph_bs=args.graph_bs,
    )
    forward = llm.engine.forward_batch

    def trace_forward(batch, sample_args):
        print(
            "batch:",
            batch.phase,
            "uids:",
            [req.uid for req in batch.reqs],
            "query_lens:",
            [req.extend_len for req in batch.reqs],
            "kv_lens:",
            [req.device_len for req in batch.reqs],
            "graph:",
            llm.engine.graph_runner.can_use_cuda_graph(batch),
            flush=True,
        )
        return forward(batch, sample_args)

    llm.engine.forward_batch = trace_forward
    try:
        with torch.inference_mode():
            results = llm.generate(
                ["The capital of France is", "Repeat the word hello. " * 12],
                SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
            )
            print("outputs:", results, flush=True)
            assert all(len(result["token_ids"]) == 4 for result in results), results
            assert not llm.decode_manager.runnable
            assert not llm.prefill_manager.runnable
            assert llm.table_manager.available_size == 2
            llm.cache_manager.check_integrity()
        print("peak_allocated_bytes:", torch.cuda.max_memory_allocated(), flush=True)
        print("BASELINE SMOKE PASSED", flush=True)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
