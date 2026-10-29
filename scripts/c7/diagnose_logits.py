"""Measure cross-batch-shape BF16 logits drift with mixed batching DISABLED.

This is diagnostic evidence, not a passing numerical-correctness test.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(
        args.model,
        max_seq_len_override=128,
        max_running_req=3,
        max_extend_tokens=24,
        num_page_override=768,
        attention_backend="fi",
        page_size=1,
        cache_type="naive",
        cuda_graph_bs=[],
    )
    original_forward, original_sample = llm.engine.forward_batch, llm.engine.sampler.sample
    current, captures = None, {}

    def forward(batch, sampling_args):
        nonlocal current
        current = batch
        assert batch.layout is None
        return original_forward(batch, sampling_args)

    def sample(logits, sampling_args):
        for row, req in enumerate(current.reqs):
            captures.setdefault(req.uid, []).append(logits[row].clone())
        return original_sample(logits, sampling_args)

    llm.engine.forward_batch, llm.engine.sampler.sample = forward, sample
    try:
        prompt = [100, 101, 102, 103, 104, 105]
        params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=3)
        one = llm.generate([prompt], params)
        reference = [value.float().cpu() for value in captures[0]]
        captures.clear()
        three = llm.generate([prompt] * 3, params)
        print("BASELINE_ONLY_TOKENS " + json.dumps({"single": one, "batch3": three}), flush=True)
        for uid, values in captures.items():
            for step, (actual, expected) in enumerate(zip(values, reference)):
                actual = actual.float().cpu()
                error = (actual - expected).abs()
                print(
                    "BASELINE_ONLY_LOGITS "
                    + json.dumps(
                        {
                            "uid": uid,
                            "step": step,
                            "max_abs": error.max().item(),
                            "violations_atol_0.1_rtol_0.01": int(
                                (error > 0.1 + 0.01 * expected.abs()).sum()
                            ),
                            "argmax_equal": int(actual.argmax()) == int(expected.argmax()),
                        }
                    ),
                    flush=True,
                )
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
