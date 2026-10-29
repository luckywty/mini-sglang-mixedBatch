"""CPU contracts for compact sampling and overlap lifecycle.

GPU streams/events and pinned storage are replaced at their boundary; real
Scheduler result handling, cache/slot ownership, radix operations, Engine state
advance, LMHead math and sampling-parameter preparation are exercised.
"""

from __future__ import annotations

from types import SimpleNamespace

import minisgl.core as core
import minisgl.distributed.info as distributed_info
import pytest
import torch
from minisgl.core import Batch, BatchLayout, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine.engine import Engine
from minisgl.engine.sample import Sampler
from minisgl.kvcache.naive_cache import NaiveCacheHandle
from minisgl.layers.embedding import ParallelLMHead
from minisgl.message import AbortBackendMsg, UserMsg
from minisgl.scheduler import Scheduler
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.lifecycle import MixedLifecycle
from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
from minisgl.scheduler.scheduler import ForwardInput, _make_write_tuple
from minisgl.scheduler.table import TableManager


class Event:
    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1

    def record(self, stream):
        pass


@pytest.fixture(autouse=True)
def cpu_boundary(monkeypatch):
    monkeypatch.setenv("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(1))
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    for name in ("empty", "tensor"):
        original = getattr(torch, name)

        def unpinned(*args, _original=original, **kwargs):
            kwargs.pop("pin_memory", None)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, unpinned)
    yield
    assert not torch.cuda.is_initialized()


def batch_with_sampling():
    dec = Req(torch.tensor([1, 2, 3]), 0, 2, 5, 10, SamplingParams(), NaiveCacheHandle())
    mid = ChunkedReq(
        torch.tensor([4, 5, 6]),
        1,
        0,
        5,
        20,
        SamplingParams(temperature=3, top_k=1, top_p=0.1),
        NaiveCacheHandle(),
    )
    final = Req(
        torch.tensor([7, 8]),
        2,
        0,
        1,
        30,
        SamplingParams(temperature=0.7, top_k=3, top_p=0.8),
        NaiveCacheHandle(),
    )
    reqs = [dec, mid, final]
    batch = Batch(reqs, "mixed", BatchLayout.capture(reqs, ("decode", "prefill", "prefill")))
    batch.padded_reqs = reqs
    batch.sample_hidden_indices = torch.tensor(batch.layout.sample_token_indices)
    return batch


def test_sampler_params_and_write_tuple_use_same_compressed_rows():
    batch = batch_with_sampling()
    args = Sampler(torch.device("cpu"), 8).prepare(batch)
    torch.testing.assert_close(args.temperatures, torch.tensor([1e-6, 0.7]))
    assert args.top_k.tolist() == [8, 3]
    torch.testing.assert_close(args.top_p, torch.tensor([1.0, 0.8]))
    assert batch.layout.sample_req_indices == (0, 2)
    assert batch.layout.sample_token_indices == (0, 5)
    for req in batch.reqs:
        req.complete_one()  # write mapping must still use the immutable snapshot
    slots, positions = _make_write_tuple(batch, torch.device("cpu"))
    assert slots.tolist() == [0, 2] and positions.tolist() == [3, 2]
    assert -1 not in positions


def test_lmhead_selects_only_correct_hidden_rows():
    batch = batch_with_sampling()
    head = ParallelLMHead(4, 4)
    head.weight = torch.eye(4)
    hidden = torch.zeros(6, 4)
    hidden[0, 1], hidden[3, 0], hidden[5, 3] = 2, 1000, 4
    with core.get_global_ctx().forward_batch(batch):
        logits = ParallelLMHead.forward.__wrapped__(head, hidden)
    assert logits.shape == (2, 4)
    assert logits.argmax(-1).tolist() == [1, 3]


@pytest.mark.parametrize("only_intermediate", [False, True])
def test_real_engine_advances_kv_without_fake_sampling(monkeypatch, only_intermediate):
    batch = batch_with_sampling()
    if only_intermediate:
        req = batch.reqs[1]
        batch = Batch([req], "prefill", BatchLayout.capture([req], ("prefill",)))
        batch.sample_hidden_indices = torch.tensor([], dtype=torch.int64)
    engine = Engine.__new__(Engine)
    engine.stream = object()
    engine.device = torch.device("cpu")
    engine.ctx = core.get_global_ctx()
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: False)
    sample_calls = []

    def sample(logits, args):
        sample_calls.append(logits.shape[0])
        return logits.argmax(-1)

    engine.model = SimpleNamespace(
        forward=lambda: torch.zeros(len(batch.layout.sample_req_indices), 4)
    )
    engine.sampler = SimpleNamespace(sample=sample)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: engine.stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    output = engine.forward_batch(batch, None)
    assert sample_calls == ([] if only_intermediate else [2])
    assert output.next_tokens_cpu.numel() == (0 if only_intermediate else 2)
    for req, length, needs_sample in zip(
        batch.reqs, batch.layout.kv_lens, batch.layout.sample_mask
    ):
        assert req.cached_len == length and req.device_len == length + int(needs_sample)


@pytest.fixture
def scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_mixed_batching = True
    scheduler.mixed_lifecycle = MixedLifecycle()
    table = torch.full((4, 64), -1, dtype=torch.int32)
    scheduler.table_manager = TableManager(3, table)
    scheduler.cache_manager = CacheManager(128, 1, table, "radix")
    scheduler.decode_manager = DecodeManager(1)
    scheduler.prefill_manager = PrefillManager(
        scheduler.cache_manager, scheduler.table_manager, scheduler.decode_manager
    )
    scheduler.engine = SimpleNamespace(page_table=table)
    scheduler.eos_token_id = 99
    scheduler.replies, scheduler.barriers = [], []
    scheduler.send_result = lambda reply: scheduler.replies.extend(reply)
    scheduler._wait_for_engine = lambda: scheduler.barriers.append(
        "engine-before-resource-mutation"
    )
    return scheduler


def prefill(scheduler, uid=1, ids=None, output_len=4, budget=64, ignore_eos=False):
    if ids is None:
        ids = torch.tensor([10, 11, 12], dtype=torch.int32)
    scheduler.prefill_manager.add_one_req(
        UserMsg(uid, ids, SamplingParams(ignore_eos=ignore_eos, max_tokens=output_len))
    )
    batch = scheduler.prefill_manager.schedule_next_batch(budget)
    batch.layout = BatchLayout.capture(batch.reqs, ("prefill",) * batch.size)
    return batch


def submit(scheduler, batch, tokens):
    """Emulate GPU completion state; result handling itself stays unmodified."""
    states = scheduler.mixed_lifecycle.register(batch.reqs)
    scheduler.cache_manager.allocate_paged(batch.reqs)
    for req, length, sample in zip(batch.reqs, batch.layout.kv_lens, batch.layout.sample_mask):
        req.cached_len, req.device_len = length, length + int(sample)
    scheduler.decode_manager.filter_reqs(batch.reqs)
    mapping = (torch.tensor([], dtype=torch.int64),) * 2
    forward = ForwardInput(batch, None, mapping, mapping, states)
    event = Event()
    return forward, (None, torch.tensor(tokens, dtype=torch.int32), event)


def decode_batch(req):
    return Batch([req], "decode", BatchLayout.capture([req], ("decode",)))


def assert_free_integrity(scheduler):
    cm = scheduler.cache_manager
    cm.check_integrity()
    assert cm.free_slots.unique().numel() == cm.free_slots.numel()
    nodes, protected, evictable = [cm.prefix_cache.root_node], 0, 0
    cached_pages = []
    while nodes:
        node = nodes.pop()
        nodes.extend(node.children.values())
        if node.is_root():
            continue
        assert node.ref_count == 0
        evictable += node.length
        cached_pages.extend(node.value.tolist())
    assert cm.prefix_cache.protected_size == protected == 0
    assert cm.prefix_cache.evictable_size == evictable
    assert not set(cached_pages) & set(cm.free_slots.tolist())
    assert len(set(cached_pages)) + len(cm.free_slots) == cm.num_pages


def test_overlap_termination_uses_snapshot_and_sends_finished_once(scheduler):
    batch = prefill(scheduler, output_len=2)
    first = submit(scheduler, batch, [40])
    req = batch.reqs[0]
    second = submit(scheduler, decode_batch(req), [41])
    assert not req.can_decode
    scheduler._process_last_data(first)
    assert [msg.finished for msg in scheduler.replies] == [False]
    assert req.cache_handle.cached_len == batch.layout.kv_lens[0] == 3
    assert req.cached_len == 4  # speculative Decode KV must remain private
    scheduler._process_last_data(second)
    assert [msg.finished for msg in scheduler.replies] == [False, True]
    assert req.input_ids.tolist() == [10, 11, 12, 40, 41]
    assert not scheduler.mixed_lifecycle.active
    assert scheduler.table_manager.available_size == 3
    assert scheduler.barriers
    assert_free_integrity(scheduler)


def test_eos_suppresses_speculative_result_after_slot_and_uid_reuse(scheduler):
    batch = prefill(scheduler)
    first = submit(scheduler, batch, [99])
    old_req = batch.reqs[0]
    speculative = submit(scheduler, decode_batch(old_req), [42])
    scheduler._process_last_data(first)
    assert len(scheduler.replies) == 1 and scheduler.replies[0].finished
    new_batch = prefill(scheduler, uid=1, ids=torch.tensor([55, 56]), output_len=1)
    assert new_batch.reqs[0].table_idx == old_req.table_idx
    new_data = submit(scheduler, new_batch, [57])
    new_state = new_data[0].request_states[0]
    scheduler._process_last_data(speculative)
    assert not new_state.terminal and len(scheduler.replies) == 1
    scheduler._process_last_data(new_data)
    assert [msg.next_token for msg in scheduler.replies] == [99, 57]
    assert scheduler.table_manager.available_size == 3
    assert_free_integrity(scheduler)


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_ignore_eos_keeps_original_meaning(scheduler, ignore_eos):
    batch = prefill(scheduler, output_len=2, ignore_eos=ignore_eos)
    first = submit(scheduler, batch, [99])
    scheduler._process_last_data(first)
    assert scheduler.replies[0].finished == (not ignore_eos)
    if ignore_eos:
        second = submit(scheduler, decode_batch(batch.reqs[0]), [1])
        scheduler._process_last_data(second)
        assert scheduler.replies[-1].finished
    assert_free_integrity(scheduler)


@pytest.mark.parametrize("kind", ["pending", "chunk", "prefill", "decode-final"])
def test_abort_before_output_does_not_release_twice_or_emit(scheduler, kind):
    if kind == "pending":
        scheduler.prefill_manager.add_one_req(UserMsg(1, torch.tensor([1, 2]), SamplingParams()))
        data = None
    else:
        batch = prefill(
            scheduler,
            output_len=1 if kind == "decode-final" else 4,
            budget=1 if kind == "chunk" else 64,
        )
        data = submit(scheduler, batch, [] if kind == "chunk" else [1])
    scheduler._process_one_msg(AbortBackendMsg(1))
    scheduler._process_one_msg(AbortBackendMsg(1))
    scheduler._process_last_data(data)
    assert not scheduler.replies
    assert not scheduler.mixed_lifecycle.active
    assert not scheduler.prefill_manager.runnable and not scheduler.decode_manager.runnable
    assert scheduler.table_manager.available_size == 3
    assert_free_integrity(scheduler)


def test_chunk_owner_updates_but_old_result_cannot_retire_new_generation(scheduler):
    batch = prefill(scheduler, budget=1)
    first = submit(scheduler, batch, [])
    following = scheduler.prefill_manager.schedule_next_batch(1)
    following.layout = BatchLayout.capture(following.reqs, ("prefill",))
    second = submit(scheduler, following, [])
    assert first[0].request_states[0] is second[0].request_states[0]
    assert first[0].request_states[0].req is following.reqs[0]
    scheduler._process_one_msg(AbortBackendMsg(1))
    scheduler._process_last_data(first)
    scheduler._process_last_data(second)
    assert not scheduler.replies
    assert scheduler.table_manager.available_size == 3
    assert_free_integrity(scheduler)


def test_duplicate_prefix_remaps_live_page_table_before_freed_pages_reuse(scheduler):
    first = prefill(scheduler, uid=1)
    first_data = submit(scheduler, first, [40])
    second = prefill(scheduler, uid=2)  # admitted before either prefix was inserted
    second_data = submit(scheduler, second, [41])
    old_second_pages = scheduler.engine.page_table[second.reqs[0].table_idx, :3].clone()
    scheduler._process_last_data(first_data)
    scheduler._process_last_data(second_data)
    req = second.reqs[0]
    canonical = req.cache_handle.get_matched_indices()
    torch.testing.assert_close(scheduler.engine.page_table[req.table_idx, :3], canonical)
    assert not torch.equal(canonical, old_second_pages)
    assert set(old_second_pages.tolist()) <= set(scheduler.cache_manager.free_slots.tolist())
    assert not set(canonical.tolist()) & set(scheduler.cache_manager.free_slots.tolist())
    for uid in (1, 2):
        scheduler._process_one_msg(AbortBackendMsg(uid))
    assert_free_integrity(scheduler)


def test_gpu_resource_dependency_targets_the_current_stream(scheduler, monkeypatch):
    order = []
    current, engine_stream = (
        SimpleNamespace(wait_stream=lambda other: order.append(other)),
        object(),
    )
    scheduler.device, scheduler.engine.stream = torch.device("cpu"), engine_stream
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current)
    Scheduler._wait_for_engine(scheduler)
    assert order == [engine_stream]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: engine_stream)
    Scheduler._wait_for_engine(scheduler)
    assert order == [engine_stream]


def test_completion_event_covers_token_pool_writeback(scheduler):
    batch = prefill(scheduler)
    batch.input_ids = None
    req = batch.reqs[0]
    slot, pos = req.table_idx, len(req.input_ids)
    pool = scheduler.table_manager.token_pool
    pool[slot, pos] = -777
    observations = []

    class WriteEvent(Event):
        def record(self, stream):
            observations.append(int(pool[slot, pos]))

    from minisgl.engine.engine import ForwardOutput

    event = WriteEvent()
    scheduler.engine.stream = object()
    scheduler.engine.forward_batch = lambda batch, args: ForwardOutput(
        torch.tensor([42], dtype=torch.int32), torch.tensor([42], dtype=torch.int32), event
    )
    scheduler.token_pool = pool
    inputs = (torch.tensor([slot]), torch.tensor([0]))
    writes = (torch.tensor([slot]), torch.tensor([pos]))
    scheduler._forward(ForwardInput(batch, None, inputs, writes))
    assert observations == [42]
