from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    enable_mixed_batching: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def __post_init__(self) -> None:
        if not self.enable_mixed_batching:
            return
        if self.max_extend_tokens <= 0:
            raise ValueError("Mixed batching requires max_extend_tokens > 0 (total token budget)")
        if self.tp_info.size != 1:
            raise ValueError("Mixed batching currently supports TP=1 only")
        if self.attention_backend != "fi":
            raise ValueError("Mixed batching requires explicit attention_backend='fi'")
        if self.page_size != 1:
            raise ValueError("Mixed batching currently requires page_size=1")

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
