"""Benchmark-only entrypoint. Primary mode leaves per-request runtime untouched.

events/profile hooks are installed after Engine/Graph initialization. CUDA timing
is collected at shutdown; no per-forward synchronization or artificial delays.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import time
from pathlib import Path


def instrument():
    import torch
    from minisgl.message import AbortBackendMsg, UserMsg
    from minisgl.scheduler import Scheduler

    output = Path(os.environ["C7_BENCH_OUTPUT"])
    mode = os.environ["C7_BENCH_MODE"]
    original_init = Scheduler.__init__

    def initialize(self, config):
        original_init(self, config)
        metadata = {
            "mode": mode,
            "model": config.model_path,
            "dtype": str(self.engine.dtype),
            "tp": config.tp_info.size,
            "page_size": config.page_size,
            "attention_backend": config.attention_backend,
            "context": self.engine.max_seq_len,
            "slots": config.max_running_req,
            "pages": self.engine.num_pages,
            "budget": self.prefill_budget,
            "cache": config.cache_type,
            "mixed": config.enable_mixed_batching,
            "graph_sizes": self.engine.graph_runner.graph_bs_list,
            "gpu": torch.cuda.get_device_name(),
            "sm": torch.cuda.get_device_capability(),
            "total_vram_bytes": torch.cuda.get_device_properties(self.device).total_memory,
            "cuda": torch.version.cuda,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "flashinfer-python", "transformers", "triton", "sgl-kernel")
            },
            "environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith("MINISGL_")
                or key
                in (
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                    "PYTORCH_CUDA_ALLOC_CONF",
                    "PYTORCH_ALLOC_CONF",
                    "OMP_NUM_THREADS",
                    "CUBLAS_WORKSPACE_CONFIG",
                )
            },
        }
        (output / "server-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if mode == "primary":
            return
        trace = (output / "scheduler.jsonl").open("w", buffering=65536)
        self._c7_trace = trace
        self._c7_timings = []

        def emit(event, **values):
            trace.write(json.dumps(dict(event=event, t=time.perf_counter(), **values)) + "\n")

        self._c7_emit = emit
        send = self.send_result

        def send_result(reply):
            for msg in reply:
                emit("token", uid=msg.uid, token=msg.next_token, finished=msg.finished)
            return send(reply)

        self.send_result = send_result
        if mode == "profile":
            engine_forward = self.engine.forward_batch

            def timed_forward(batch, args):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                graph = self.engine.graph_runner.can_use_cuda_graph(batch)
                label = f"C7/{batch.phase}/{'graph' if graph else 'eager'}/bs={batch.size}"
                begin = time.perf_counter()
                start.record(self.engine.stream)
                with torch.cuda.nvtx.range(label):
                    result = engine_forward(batch, args)
                end.record(self.engine.stream)
                self._c7_timings.append((begin, batch.phase, graph, start, end))
                return result

            self.engine.forward_batch = timed_forward

    Scheduler.__init__ = initialize
    if mode == "primary":
        return
    original_process, original_forward = Scheduler._process_one_msg, Scheduler._forward
    original_idle, original_shutdown = Scheduler.run_when_idle, Scheduler.shutdown

    def process(self, msg):
        if isinstance(msg, UserMsg):
            from common import digest

            self._c7_emit(
                "receive",
                uid=msg.uid,
                input_sha256=digest(msg.input_ids.tolist()),
                input_tokens=len(msg.input_ids),
                max_tokens=msg.sampling_params.max_tokens,
            )
        elif isinstance(msg, AbortBackendMsg):
            self._c7_emit("abort", uid=msg.uid)
        return original_process(self, msg)

    def forward(self, data):
        batch, layout = data.batch, data.batch.layout
        rows = [
            {
                "uid": req.uid,
                "phase": layout.req_phases[i] if layout is not None else batch.phase,
                "cached": req.cached_len,
                "query": req.extend_len,
                "sample": req.needs_sampling,
            }
            for i, req in enumerate(batch.reqs)
        ]
        stamp = time.perf_counter()
        graph = self.engine.graph_runner.can_use_cuda_graph(batch)
        result = original_forward(self, data)
        cm = self.cache_manager
        self._c7_emit(
            "forward",
            submit_t=stamp,
            phase=batch.phase,
            graph=graph,
            rows=rows,
            sample_count=result.next_tokens_gpu.numel(),
            kv_pages=cm.num_pages,
            free_pages=len(cm.free_slots),
            evictable_tokens=cm.prefix_cache.evictable_size,
            protected_tokens=cm.prefix_cache.protected_size,
            allocated_bytes=torch.cuda.memory_allocated(self.device),
            reserved_bytes=torch.cuda.memory_reserved(self.device),
        )
        return result

    def idle(self):
        self._c7_emit("idle", free_slots=self.table_manager.available_size)
        self._c7_trace.flush()
        return original_idle(self)

    def shutdown(self):
        try:
            # The production shutdown already synchronizes all CUDA work.
            original_shutdown(self)
            for begin, phase, graph, start, end in self._c7_timings:
                self._c7_emit(
                    "cuda_forward",
                    submit_t=begin,
                    phase=phase,
                    graph=graph,
                    milliseconds=start.elapsed_time(end),
                )
            self._c7_emit(
                "shutdown",
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        finally:
            self._c7_trace.close()

    Scheduler._process_one_msg, Scheduler._forward = process, forward
    Scheduler.run_when_idle, Scheduler.shutdown = idle, shutdown


if __name__ in ("__main__", "__mp_main__"):
    instrument()

if __name__ == "__main__":
    from minisgl.server.launch import launch_server

    launch_server()
