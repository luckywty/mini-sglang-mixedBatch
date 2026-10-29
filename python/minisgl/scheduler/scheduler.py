from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .lifecycle import MixedLifecycle, RequestState
from .mixed import schedule_mixed_batch
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)
    request_states: Tuple[RequestState, ...] | None = None


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.enable_mixed_batching = config.enable_mixed_batching
        if self.enable_mixed_batching:
            self.mixed_lifecycle = MixedLifecycle()
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        if last_data[0].batch.layout is not None:
            self._process_mixed_result(last_data)
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _wait_for_engine(self) -> None:
        # A previous completion event does not cover the NEXT speculative forward.
        # Order ALL already-submitted engine work before mutating/reusing resources.
        stream = torch.cuda.current_stream(self.device)
        if stream != self.engine.stream:
            stream.wait_stream(self.engine.stream)

    def _cache_mixed_req(
        self, req: Req, *, finished: bool, prefill_len: int | None = None
    ) -> torch.Tensor | None:
        cached_req = req
        if not finished:
            assert prefill_len is not None
            # Req may already include next iteration's Decode KV. Cache only
            # this Prefill's frozen range; keep the Decode tail privately owned.
            cached_req = copy(req)
            cached_req.cached_len = prefill_len
        if cached_req.cached_len > len(cached_req.input_ids):
            raise RuntimeError("Cannot cache KV without the corresponding accepted host tokens")
        self._wait_for_engine()
        self.cache_manager.cache_req(cached_req, finished=finished)
        if not finished:
            req.cache_handle = cached_req.cache_handle
            # Return the canonical indices. The caller remaps AFTER lazy_free_region
            # consumes its page-table views; remapping earlier would free the new
            # canonical pages instead of the old duplicate pages.
            handle = req.cache_handle
            if handle.cached_len:
                return handle.get_matched_indices()
        return None

    def _retire_mixed_req(self, state: RequestState) -> None:
        if not self.mixed_lifecycle.retire(state):
            return
        req = state.req
        self.prefill_manager.abort_req(req.uid)
        self.decode_manager.remove_req(req)
        self._cache_mixed_req(req, finished=True)
        self.table_manager.free(req.table_idx)

    def _process_mixed_result(self, last_data: ForwardData) -> None:
        forward_input, (_, next_tokens_cpu, copy_done) = last_data
        batch, states = forward_input.batch, forward_input.request_states
        layout = batch.layout
        assert layout is not None and states is not None and len(states) == batch.size
        copy_done.synchronize()
        indices = layout.sample_req_indices
        assert next_tokens_cpu.numel() == len(indices)
        reply: List[DetokenizeMsg] = []
        remaps: List[Tuple[Req, torch.Tensor]] = []
        with self.cache_manager.lazy_free_region():
            for output_row, req_row in enumerate(indices):
                state, req = states[req_row], batch.reqs[req_row]
                if state.terminal:
                    continue
                next_token = next_tokens_cpu[output_row]
                req.append_host(next_token.unsqueeze(0))
                token_id = int(next_token.item())
                # Req may already have advanced in the NEXT overlapped forward.
                finished = layout.kv_lens[req_row] + 1 >= req.max_device_len
                if not req.sampling_params.ignore_eos:
                    finished |= token_id == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=token_id, finished=finished))
                if finished:
                    self._retire_mixed_req(state)
                elif layout.req_phases[req_row] == "prefill":
                    canonical = self._cache_mixed_req(
                        req, finished=False, prefill_len=layout.kv_lens[req_row]
                    )
                    if canonical is not None:
                        remaps.append((req, canonical))
        for req, canonical in remaps:
            self.engine.page_table[req.table_idx, : len(canonical)].copy_(
                canonical, non_blocking=True
            )
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            if self.enable_mixed_batching:
                state = self.mixed_lifecycle.find(msg.uid)
                self.prefill_manager.abort_req(msg.uid)
                self.decode_manager.abort_req(msg.uid)
                if state is not None:
                    self._retire_mixed_req(state)
                return
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        if self.enable_mixed_batching:
            if batch.layout is None:
                raise ValueError("Mixed scheduling requires a layout for every batch")
        elif batch.is_mixed or batch.layout is not None:
            raise ValueError("Mixed batch execution requires enable_mixed_batching=True")
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if batch.layout is not None and not batch.is_decode:
            batch.sample_hidden_indices = torch.tensor(
                batch.layout.sample_token_indices, dtype=torch.int64, pin_memory=True
            ).to(self.device, non_blocking=True)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
            request_states=(
                self.mixed_lifecycle.register(batch.reqs) if batch.layout is not None else None
            ),
        )

    def _select_next_batch(self) -> Batch | None:
        """CPU selection boundary, also usable without constructing a GPU Engine."""
        if self.enable_mixed_batching:
            return schedule_mixed_batch(
                self.decode_manager, self.prefill_manager, self.prefill_budget
            )
        return (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        batch = self._select_next_batch()
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args = forward_input.batch, forward_input.sample_args
        input_mapping, output_mapping = forward_input.input_tuple, forward_input.write_tuple
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        if batch.layout is None or forward_output.next_tokens_gpu.numel():
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        if batch.layout is not None:
            # ForwardInput's GPU tensors must outlive token_pool writeback too,
            # which is submitted AFTER Engine's original D2H completion event.
            forward_output.copy_done_event.record(self.engine.stream)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    if batch.layout is not None:
        indices = batch.layout.sample_req_indices
        mapping_list = [batch.layout.table_indices[i] for i in indices]
        write_list = [batch.layout.kv_lens[i] for i in indices]
    else:
        mapping_list = [req.table_idx for req in batch.reqs]
        write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
