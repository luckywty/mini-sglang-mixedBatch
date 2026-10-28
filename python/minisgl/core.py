from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Tuple, TypeAlias

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


RequestPhase: TypeAlias = Literal["prefill", "decode"]


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    @property
    def needs_sampling(self) -> bool:
        """Decode and the final prefill chunk produce an output token."""
        return True

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass(frozen=True)
class BatchLayout:
    """Unpadded CPU layout captured before shared Req state advances.

    Sampling rows are ordered by request, with intermediate prefill chunks omitted.
    This is an execution contract for the future mixed executor, not GPU metadata.
    """

    request_uids: Tuple[int, ...]
    table_indices: Tuple[int, ...]
    req_phases: Tuple[RequestPhase, ...]
    query_lens: Tuple[int, ...]
    kv_lens: Tuple[int, ...]
    sample_mask: Tuple[bool, ...]

    def __post_init__(self) -> None:
        size = len(self.request_uids)
        if not size or any(
            len(values) != size
            for values in (
                self.table_indices,
                self.req_phases,
                self.query_lens,
                self.kv_lens,
                self.sample_mask,
            )
        ):
            raise ValueError("Batch layout requires nonempty, equally sized request fields")
        if len(set(self.request_uids)) != size or len(set(self.table_indices)) != size:
            raise ValueError("A scheduled request UID and table slot must each be unique")
        for phase, q_len, kv_len, sample in zip(
            self.req_phases, self.query_lens, self.kv_lens, self.sample_mask
        ):
            if phase not in ("prefill", "decode") or not 0 < q_len <= kv_len:
                raise ValueError("Invalid request phase or query/KV length")
            if phase == "decode" and (q_len != 1 or not sample):
                raise ValueError("Each decode request must compute one token and sample")

    @classmethod
    def capture(cls, reqs: List[Req], req_phases: Tuple[RequestPhase, ...]) -> BatchLayout:
        return cls(
            request_uids=tuple(req.uid for req in reqs),
            table_indices=tuple(req.table_idx for req in reqs),
            req_phases=req_phases,
            query_lens=tuple(req.extend_len for req in reqs),
            kv_lens=tuple(req.device_len for req in reqs),
            sample_mask=tuple(req.needs_sampling for req in reqs),
        )

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def cached_lens(self) -> Tuple[int, ...]:
        return tuple(k - q for q, k in zip(self.query_lens, self.kv_lens))

    @property
    def qo_indptr(self) -> Tuple[int, ...]:
        from itertools import accumulate

        return (0, *accumulate(self.query_lens))

    @property
    def sample_req_indices(self) -> Tuple[int, ...]:
        """Compressed sampling/output row -> unpadded batch request row."""
        return tuple(i for i, sample in enumerate(self.sample_mask) if sample)

    @property
    def sample_token_indices(self) -> Tuple[int, ...]:
        """Packed hidden-state rows for requests that actually produce a token."""
        return tuple(end - 1 for end, sample in zip(self.qo_indptr[1:], self.sample_mask) if sample)


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode", "mixed"]
    # Only the opt-in mixed scheduling path takes a CPU snapshot (including pure batches).
    layout: BatchLayout | None = None
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def is_mixed(self) -> bool:
        return self.phase == "mixed"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
