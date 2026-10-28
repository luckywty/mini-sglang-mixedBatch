"""Record the actual interpreter, source checkout and installed FlashInfer API."""

from __future__ import annotations

import importlib.metadata as metadata
import inspect
import os
import platform
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


def main() -> None:
    print("time:", datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), flush=True)
    print("platform:", platform.platform())
    print("python:", sys.version, sys.executable)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    for args in [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--short", "--branch"],
        ["nvidia-smi"],
        ["/usr/local/cuda/bin/nvcc", "--version"],
    ]:
        print("command:", args, flush=True)
        subprocess.run(args, check=True)
    for name in [
        "torch", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
        "transformers", "triton", "pytest", "pytest-cov", "apache-tvm-ffi",
        "sgl-kernel", "quack-kernels", "minisgl",
    ]:
        try:
            print(name, metadata.version(name))
        except metadata.PackageNotFoundError:
            print(name, "NOT INSTALLED")

    import torch
    import flashinfer
    import minisgl.core

    print("source:", minisgl.core.__file__)
    print("torch CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name())
        print("SM:", torch.cuda.get_device_capability(), "BF16:", torch.cuda.is_bf16_supported())
    for name in ["BatchPrefillWithPagedKVCacheWrapper", "BatchDecodeWithPagedKVCacheWrapper"]:
        cls = getattr(flashinfer, name)
        print(name, "source:", inspect.getfile(cls))
        for method in [cls.__init__, cls.plan, cls.run]:
            print(method.__name__, inspect.signature(method))


if __name__ == "__main__":
    main()
