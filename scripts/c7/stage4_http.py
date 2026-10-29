"""Local HTTP/SSE/cancellation correctness check; no serving benchmarks."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def free_port():
    # The distributed process group also binds port+1.
    while True:
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            try:
                second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port


def exercise(url, server_log):
    results = []
    short = "The capital of France is"
    long = "Repeat the word hello. " * 12

    def post(prompt, count, streaming):
        payload = {
            "model": "local",
            "prompt": prompt,
            "max_tokens": count,
            "temperature": 0,
            "ignore_eos": True,
            "stream": streaming,
        }
        return urllib.request.urlopen(
            urllib.request.Request(
                url + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=45,
        )

    with urllib.request.urlopen(url + "/v1/models", timeout=5) as response:
        models = json.load(response)
        assert len(models["data"]) == 1

    def stream(prompt, count):
        text, done, finished, roles = "", 0, 0, 0
        with post(prompt, count, True) as response:
            assert response.status == 200
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done += 1
                    continue
                choice = json.loads(data)["choices"][0]
                text += choice["delta"].get("content", "")
                roles += int("role" in choice["delta"])
                finished += int(choice["finish_reason"] is not None)
        assert done == finished == roles == 1
        assert text
        return {"done": done, "finished": finished, "role_chunks": roles, "text": text}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(stream, short, 96), pool.submit(stream, long, 4)]
        pair = [future.result() for future in futures]
    results.append({"case": "concurrent-sse", "responses": pair})
    streamed = stream(short, 4)
    with post(short, 4, False) as response:
        assert response.status == 200
        content = json.load(response)["choices"][0]["message"]["content"]
    assert content == streamed["text"]
    results.append({"case": "stream-vs-json-prefix-hit", "content": content})

    with post(long, 60, True) as response:
        assert response.status == 200
        for raw in response:
            if raw.startswith(b"data: {"):
                break
    # Wait for the actual AbortBackendMsg, not just a closed client socket.
    deadline = time.monotonic() + 5
    aborts = []
    while time.monotonic() < deadline:
        aborts = [
            json.loads(line.split("C7_HTTP_ABORT ", 1)[1])
            for line in server_log.read_text().splitlines()
            if line.startswith("C7_HTTP_ABORT ")
        ]
        if any(row["was_active"] for row in aborts):
            break
        time.sleep(0.05)
    assert any(
        row["was_active"] and row["remaining_active"] == 0 and row["free_slots"] == 3
        for row in aborts
    ), aborts
    with post(short, 4, False) as response:
        assert response.status == 200
        assert json.load(response)["choices"][0]["message"]["content"] == content
    results.append({"case": "disconnect-active-abort-and-reuse", "aborts": aborts})
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    server_log = root / "docs/c7/evidence/stage4-http-server.txt"
    port = free_port()
    env = dict(os.environ, PYTHONPATH=str(root / "python"), MINISGL_STAGE4_TEST_DELAY_MS="10")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    command = [
        sys.executable,
        "-u",
        "scripts/c7/stage4_http_server.py",
        "--model",
        "/home/wty238388/ai-infra/models/Qwen3-0.6B",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "fi",
        "--page-size",
        "1",
        "--tp-size",
        "1",
        "--enable-mixed-batching",
        "--max-running-requests",
        "3",
        "--max-seq-len-override",
        "128",
        "--max-extend-length",
        "7",
        "--num-pages",
        "768",
        "--graph",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print("C7_HTTP_COMMAND " + json.dumps(command), flush=True)
    with server_log.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            ready = False
            for _ in range(300):
                if process.poll() is not None:
                    raise RuntimeError("HTTP server exited before readiness; inspect server log")
                try:
                    with urllib.request.urlopen(url + "/v1", timeout=1) as response:
                        ready = response.status == 200
                except OSError:
                    pass
                if ready:
                    break
                time.sleep(0.2)
            assert ready, "HTTP readiness timeout"
            results = exercise(url, server_log)
            forwards = [
                json.loads(line.split("C7_HTTP_FORWARD ", 1)[1])
                for line in server_log.read_text().splitlines()
                if line.startswith("C7_HTTP_FORWARD ")
            ]
            assert any(row["phase"] == "mixed" for row in forwards)
            print(
                "C7_STAGE4_HTTP_PASS "
                + json.dumps(
                    {
                        "checks": results,
                        "mixed_forwards": sum(row["phase"] == "mixed" for row in forwards),
                        "test_only_cpu_delay_ms": 10,
                        "graph": False,
                    }
                ),
                flush=True,
            )
        finally:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
