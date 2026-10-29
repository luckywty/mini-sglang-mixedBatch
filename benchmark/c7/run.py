"""Plan or execute sequential A/B trials. Default is dry-run; never infers a win."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import read_json, source_fingerprint, validate_plan, write_json
from prepare import model_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def free_port():
    while True:
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port == 65535:
                continue
            try:
                second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port


def server_command(plan, policy, port):
    if policy not in ("baseline", "mixed"):
        raise ValueError("No independent Decode-priority policy exists in this branch")
    config = plan["server"]
    command = [
        sys.executable,
        "-u",
        str(ROOT / "benchmark/c7/server.py"),
        "--model",
        plan["model"],
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "fi",
        "--page-size",
        "1",
        "--tp-size",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-running-requests",
        str(config["slots"]),
        "--max-seq-len-override",
        str(config["context"]),
        "--num-pages",
        str(config["pages"]),
        "--max-extend-length",
        str(config["budget"]),
        "--graph",
        str(config["graph"]),
        "--cache-type",
        config["cache"],
    ]
    if policy == "mixed":
        command.append("--enable-mixed-batching")
    return command


def command_output(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def provenance():
    versions = {}
    for name in ("torch", "flashinfer-python", "transformers", "triton", "sgl-kernel"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "git": command_output(["git", "rev-parse", "HEAD"]),
        "status": command_output(["git", "status", "--porcelain"]),
        "diff": command_output(["git", "diff", "HEAD"]),
        "code_sha256": source_fingerprint(ROOT),
        "python": sys.version,
        "executable": sys.executable,
        "versions": versions,
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": command_output([shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc", "--version"]),
        "freeze": command_output(["uv", "pip", "freeze", "--python", sys.executable]),
        "nsys": command_output(["nsys", "--version"]),
    }


def profile_command(command, directory):
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--wait=all",
        "--force-overwrite=false",
        f"--output={directory / 'nsys'}",
        *command,
    ]


def trial(plan, plan_path, scenario, policy, repeat, mode, directory, args, environment):
    directory.mkdir(parents=True)
    port = free_port()
    env = dict(environment, C7_BENCH_OUTPUT=str(directory), C7_BENCH_MODE=mode)
    env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "0" if plan["server"]["overlap"] else "1"
    command = server_command(plan, policy, port)
    if args.nsys:
        command = profile_command(command, directory)
    metadata = {
        "schema": "c7-run-v1",
        "plan_sha256": plan["sha256"],
        "kind": plan["kind"],
        "scenario": scenario,
        "policy": policy,
        "repeat": repeat,
        "mode": mode,
        "nsys": args.nsys,
        "command": command,
        "max_clients": args.max_clients,
        "timeout_s": args.request_timeout,
        "status": "STARTED",
        "errors": [],
        "code_sha256": source_fingerprint(ROOT),
    }
    write_json(directory / "run.json", metadata)
    server, telemetry, telemetry_file = None, None, None
    try:
        with (directory / "server.log").open("w") as log:
            server = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + args.startup_timeout
            url = f"http://127.0.0.1:{port}"
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited before readiness: {server.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server readiness timeout")
                try:
                    with urllib.request.urlopen(url + "/v1", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.2)
            if mode != "primary":
                telemetry_file = (directory / "gpu.csv").open("w")
                telemetry = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm",
                        "--format=csv,nounits",
                        "--loop-ms=500",
                    ],
                    stdout=telemetry_file,
                    stderr=subprocess.STDOUT,
                )
            for warmup in (True, False):
                client_command = [
                    sys.executable,
                    str(ROOT / "benchmark/c7/client.py"),
                    "--plan",
                    str(plan_path),
                    "--scenario",
                    scenario,
                    "--url",
                    url,
                    "--timeout",
                    str(args.request_timeout),
                    "--max-clients",
                    str(args.max_clients),
                    "--output",
                    str(directory / ("warmup.json" if warmup else "client.json")),
                ]
                if warmup:
                    client_command.append("--warmup")
                with (directory / ("warmup.log" if warmup else "client.log")).open("w") as log:
                    result = subprocess.run(
                        client_command,
                        env=env,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=(
                            args.request_timeout
                            + 15
                            + (0 if warmup else plan["scenarios"][scenario]["window_s"])
                        ),
                    )
                metadata["warmup_exit" if warmup else "client_exit"] = result.returncode
                if result.returncode and warmup:
                    raise RuntimeError("Warmup failed; measured load was not started")
                if warmup:
                    time.sleep(0.5)  # fixed settling interval, identical for both policies
            time.sleep(0.5)  # drain remaining overlapped bookkeeping; outside client window
            if server.poll() is not None:
                raise RuntimeError("Server exited during the load")
            metadata["status"] = "COMPLETE" if metadata["client_exit"] == 0 else "CLIENT_ERRORS"
    except Exception as error:
        metadata["status"] = "FAILED"
        metadata["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                metadata["server_exit"] = server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                metadata["server_exit"] = server.wait()
                metadata["status"] = "FAILED"
                metadata["errors"].append("Server required SIGKILL; trace may be incomplete")
        if telemetry is not None:
            if telemetry.poll() is None:
                telemetry.terminate()
            metadata["telemetry_exit"] = telemetry.wait(timeout=5)
        if telemetry_file is not None:
            telemetry_file.close()
        write_json(directory / "run.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenarios", nargs="+", default=["low", "near", "overload", "interference"]
    )
    parser.add_argument(
        "--policies", nargs="+", choices=("baseline", "mixed"), default=["baseline", "mixed"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("primary", "events", "profile"), default="primary")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--max-clients", type=int, default=256)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    plan = read_json(args.plan)
    validate_plan(plan)
    if (
        args.repeats < 1
        or args.max_clients < 1
        or min(args.request_timeout, args.startup_timeout) <= 0
    ):
        parser.error("Repeats, capacity and timeouts must be positive")
    if len(args.policies) != len(set(args.policies)) or len(args.scenarios) != len(
        set(args.scenarios)
    ):
        parser.error("Duplicate policy/scenario")
    if not set(args.scenarios) <= plan["scenarios"].keys():
        parser.error("Unknown scenario")
    if args.nsys and args.mode != "profile":
        parser.error("Nsight must be separate from the primary experiment: use --mode profile")
    if args.execute and args.nsys and not shutil.which("nsys"):
        parser.error("nsys is unavailable; no installation or implicit fallback")
    schedule = []
    for repeat in range(args.repeats):
        # Counterbalance policy and scenario order without changing the request trace.
        for scenario in args.scenarios[:: 1 if repeat % 2 == 0 else -1]:
            for policy in args.policies[:: 1 if repeat % 2 == 0 else -1]:
                schedule.append(
                    {"scenario": scenario, "policy": policy, "repeat": repeat, "mode": args.mode}
                )
    if not args.execute:
        for row in schedule:
            directory = (
                args.output.resolve() / f"{row['scenario']}-{row['policy']}-r{row['repeat']}"
            )
            command = server_command(plan, row["policy"], 1919)
            if args.nsys:
                command = profile_command(command, directory)
            print(
                dict(
                    **row,
                    command=command,
                    environment={
                        "C7_BENCH_OUTPUT": str(directory),
                        "C7_BENCH_MODE": args.mode,
                        "PYTHONPATH": str(ROOT / "python"),
                        "MINISGL_DISABLE_OVERLAP_SCHEDULING": (
                            "0" if plan["server"]["overlap"] else "1"
                        ),
                    },
                )
            )
        print("DRY_RUN: no server or load started. Frozen plan:", plan["sha256"])
        return 0
    if args.output.exists():
        parser.error("Output exists; each experiment must use a new directory")
    if model_fingerprint(Path(plan["model"])) != plan["model_files"]:
        parser.error("Model/tokenizer contents differ from the frozen plan")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    args.plan = args.plan.resolve()
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "provenance.json", provenance())
    write_json(args.output / "schedule.json", schedule)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    failures = []
    for row in schedule:
        directory = args.output / f"{row['scenario']}-{row['policy']}-r{row['repeat']}"
        result = trial(
            plan, args.plan, directory=directory, args=args, environment=environment, **row
        )
        print(
            {key: result[key] for key in ("scenario", "policy", "repeat", "status", "errors")},
            flush=True,
        )
        if result["status"] != "COMPLETE":
            failures.append(directory.name)
    write_json(args.output / "execution.json", {"failures": failures, "kind": plan["kind"]})
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
