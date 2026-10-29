"""Test-only server entrypoint: traces and a controlled cancellation window.

The optional CPU delay is only in this harness, never in minisgl production code.
Multiprocessing spawn re-executes the instrumentation in child processes.
"""

from __future__ import annotations

import json
import os
import time


def instrument() -> None:
    from minisgl.message import AbortBackendMsg
    from minisgl.scheduler import Scheduler

    forward, process = Scheduler._forward, Scheduler._process_one_msg

    def trace_forward(self, data):
        batch, layout = data.batch, data.batch.layout
        record = {
            "phase": batch.phase,
            "uids": [req.uid for req in batch.reqs],
            "query_lens": list(layout.query_lens),
            "cached_lens": list(layout.cached_lens),
        }
        output = forward(self, data)
        record["sample_count"] = output.next_tokens_gpu.numel()
        print("C7_HTTP_FORWARD " + json.dumps(record), flush=True)
        time.sleep(float(os.environ.get("MINISGL_STAGE4_TEST_DELAY_MS", "0")) / 1000)
        return output

    def trace_message(self, message):
        if isinstance(message, AbortBackendMsg):
            active = self.mixed_lifecycle.find(message.uid) is not None
            process(self, message)
            print(
                "C7_HTTP_ABORT "
                + json.dumps(
                    {
                        "uid": message.uid,
                        "was_active": active,
                        "remaining_active": len(self.mixed_lifecycle.active),
                        "free_slots": self.table_manager.available_size,
                    }
                ),
                flush=True,
            )
        else:
            process(self, message)

    Scheduler._forward, Scheduler._process_one_msg = trace_forward, trace_message


if __name__ in ("__main__", "__mp_main__"):
    instrument()

if __name__ == "__main__":
    from minisgl.server.launch import launch_server

    launch_server()
