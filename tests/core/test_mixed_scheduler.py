"""CPU selection/layout tests, without an Engine, model, CUDA or sampling.

Only pinned storage is replaced, and TVM's optional CUDA DLPack probing is disabled.
The real managers, prefix cache, admission, chunk continuation and packed-index helpers run on CPU tensors.
These tests do not validate attention, sampling or in-flight resource safety.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import minisgl.core as core
import minisgl.distributed.info as distributed_info
import pytest
import torch
from minisgl.core import Batch, BatchLayout, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine.graph import GraphRunner
from minisgl.message import AbortBackendMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.mixed import schedule_mixed_batch
from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
from minisgl.scheduler.scheduler import _make_input_tuple, _make_positions
from minisgl.scheduler.table import TableManager
from minisgl.server.args import parse_args


@pytest.fixture(autouse=True)
def cpu_storage(monkeypatch):
    # CPU radix comparison uses TVM FFI, but does not need its optional CUDA addon.
    monkeypatch.setenv("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
    # Admission copies to a device pool using .pin_memory(); use CPU storage here.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1))
    # Rank-aware logging needs TP metadata, but no process group or Engine.
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 1))
    yield
    assert not torch.cuda.is_initialized(), "CPU scheduling tests must not initialize CUDA"


@pytest.fixture
def make_scheduler():
    def make(budget=16, *, slots=16, pages=8192, cache_type="naive", enabled=True):
        scheduler = Scheduler.__new__(Scheduler)
        page_table = torch.full((slots + 1, 4096), -1, dtype=torch.int32)
        scheduler.table_manager = TableManager(slots, page_table)
        scheduler.cache_manager = CacheManager(pages, 1, page_table, cache_type)
        scheduler.decode_manager = DecodeManager(1)
        scheduler.prefill_manager = PrefillManager(
            scheduler.cache_manager, scheduler.table_manager, scheduler.decode_manager
        )
        scheduler.prefill_budget = budget
        scheduler.enable_mixed_batching = enabled
        return scheduler

    return make


def add_decode(scheduler, uid, *, context=5, output_len=8):
    slot = scheduler.table_manager.allocate()
    ids = torch.arange(uid * 10000, uid * 10000 + context, dtype=torch.int32)
    handle = scheduler.cache_manager.prefix_cache.match_prefix(ids[:0]).cuda_handle
    req = Req(
        ids, slot, context - 1, output_len, uid, SamplingParams(max_tokens=output_len), handle
    )
    # Materialize historical KV locations only; the current query is not cached yet.
    history = scheduler.cache_manager._allocate(context - 1)
    scheduler.table_manager.page_table[slot, : context - 1] = history
    scheduler.table_manager.token_pool[slot, :context] = ids
    scheduler.decode_manager.running_reqs.add(req)
    return req


def add_prefill(scheduler, uid, length, *, output_len=8, ids=None):
    if ids is None:
        ids = torch.arange(uid * 10000, uid * 10000 + length, dtype=torch.int32)
    scheduler.prefill_manager.add_one_req(UserMsg(uid, ids, SamplingParams(max_tokens=output_len)))
    return ids


def advance_request_state(scheduler, batch):
    """Emulate existing Engine's state advance, without pretending to sample."""
    scheduler.cache_manager.allocate_paged(batch.reqs)
    for req in batch.reqs:
        req.complete_one()
    scheduler.decode_manager.filter_reqs(batch.reqs)


def test_idle_and_pure_decode_preserve_uid_order(make_scheduler):
    scheduler = make_scheduler()
    assert scheduler._select_next_batch() is None
    reqs = [add_decode(scheduler, uid) for uid in [30, 10, 20]]
    batch = scheduler._select_next_batch()
    assert batch.is_decode and not batch.is_prefill and not batch.is_mixed
    assert batch.reqs == [reqs[1], reqs[2], reqs[0]]
    assert batch.layout.query_lens == (1, 1, 1)
    assert batch.layout.qo_indptr == (0, 1, 2, 3)
    assert batch.layout.sample_req_indices == (0, 1, 2)
    assert scheduler.decode_manager.running_reqs == set(reqs)


def test_pure_prefill_multiple_requests_share_budget(make_scheduler):
    scheduler = make_scheduler(7)
    add_prefill(scheduler, 1, 3)
    add_prefill(scheduler, 2, 10)
    batch = scheduler._select_next_batch()
    assert batch.is_prefill and not batch.is_decode and not batch.is_mixed
    assert batch.layout.query_lens == (3, 4)
    assert batch.layout.sample_mask == (True, False)
    assert batch.layout.sample_req_indices == (0,)
    assert batch.layout.sample_token_indices == (2,)
    assert [req.uid for req in scheduler.prefill_manager.pending_list] == [2]


def test_one_decode_one_prefill_unaligned_chunk(make_scheduler):
    scheduler = make_scheduler(128)
    decode = add_decode(scheduler, 1)
    add_prefill(scheduler, 2, 1000)
    batch = scheduler._select_next_batch()
    assert batch.is_mixed and not batch.is_prefill and not batch.is_decode
    assert batch.reqs[0] is decode
    assert batch.layout.req_phases == ("decode", "prefill")
    assert batch.layout.query_lens == (1, 127)
    assert batch.layout.num_tokens == 128
    assert batch.layout.sample_mask == (True, False)
    assert batch.layout.sample_token_indices == (0,)
    assert isinstance(batch.reqs[1], ChunkedReq)


def test_multiple_decodes_and_prefills_share_remainder(make_scheduler):
    scheduler = make_scheduler(12)
    for uid in [3, 1, 2]:
        add_decode(scheduler, uid)
    for uid, length in [(4, 2), (5, 3), (6, 20), (7, 1)]:
        add_prefill(scheduler, uid, length)
    batch = scheduler._select_next_batch()
    assert batch.layout.request_uids == (1, 2, 3, 4, 5, 6)
    assert batch.layout.query_lens == (1, 1, 1, 2, 3, 4)
    assert batch.layout.num_tokens == 12
    assert batch.layout.sample_mask == (True, True, True, True, True, False)
    assert batch.layout.sample_req_indices == (0, 1, 2, 3, 4)
    assert batch.layout.sample_token_indices == (0, 1, 2, 4, 7)
    assert [req.uid for req in scheduler.prefill_manager.pending_list] == [6, 7]


@pytest.mark.parametrize("prefill_len,expected_tokens", [(3, 4), (7, 8)])
def test_short_or_exact_fit_prefill(make_scheduler, prefill_len, expected_tokens):
    scheduler = make_scheduler(8)
    add_decode(scheduler, 1)
    add_prefill(scheduler, 2, prefill_len)
    batch = scheduler._select_next_batch()
    assert batch.layout.num_tokens == expected_tokens
    assert batch.layout.sample_mask == (True, True)
    assert batch.layout.sample_token_indices == (0, prefill_len)
    assert not scheduler.prefill_manager.runnable


@pytest.mark.parametrize("num_decode", [4, 6])
def test_decode_at_or_over_budget_never_drops_requests(make_scheduler, num_decode):
    scheduler = make_scheduler(4)
    decodes = [add_decode(scheduler, uid) for uid in range(num_decode)]
    add_prefill(scheduler, 99, 10)
    slots = scheduler.table_manager.available_size
    free = scheduler.cache_manager.available_size
    pending = list(scheduler.prefill_manager.pending_list)
    batch = scheduler._select_next_batch()
    assert batch.is_decode
    assert batch.reqs == decodes
    assert batch.layout.num_tokens == num_decode
    assert scheduler.prefill_budget == 4
    assert scheduler.prefill_manager.pending_list == pending
    assert pending[0].chunked_req is None
    assert scheduler.table_manager.available_size == slots
    assert scheduler.cache_manager.available_size == free


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_fails_before_admission(make_scheduler, budget):
    scheduler = make_scheduler(budget)
    add_prefill(scheduler, 1, 5)
    slots = scheduler.table_manager.available_size
    with pytest.raises(ValueError, match="positive total token budget"):
        scheduler._select_next_batch()
    assert scheduler.table_manager.available_size == slots
    assert scheduler.prefill_manager.runnable


def test_invalid_decode_length_fails_before_prefill_admission(make_scheduler):
    scheduler = make_scheduler()
    req = add_decode(scheduler, 1)
    req.cached_len -= 1
    add_prefill(scheduler, 2, 5)
    slots = scheduler.table_manager.available_size
    with pytest.raises(ValueError, match="one token"):
        scheduler._select_next_batch()
    assert scheduler.table_manager.available_size == slots
    assert scheduler.prefill_manager.pending_list[0].chunked_req is None


def test_multiround_chunk_reuses_slot_and_handle_then_enters_decode(make_scheduler, monkeypatch):
    scheduler = make_scheduler(5, slots=2)
    add_decode(scheduler, 1, output_len=20)
    ids = add_prefill(scheduler, 2, 14)
    locks = []
    lock = scheduler.cache_manager.lock

    def record_lock(handle):
        locks.append(handle)
        lock(handle)

    monkeypatch.setattr(scheduler.cache_manager, "lock", record_lock)
    chunks = []
    slot, handle = None, None
    for offset in [0, 4, 8, 12]:
        batch = scheduler._select_next_batch()
        req = batch.reqs[1]
        chunks.append(req.extend_len)
        assert req.cached_len == offset
        assert batch.layout.sample_mask == (True, offset == 12)
        assert scheduler.table_manager.available_size == 0
        if slot is None:
            slot, handle = req.table_idx, req.cache_handle
        assert req.table_idx == slot and req.cache_handle is handle
        torch.testing.assert_close(
            scheduler.table_manager.token_pool[slot, offset : req.device_len],
            ids[offset : req.device_len],
        )
        advance_request_state(scheduler, batch)
        assert (req in scheduler.decode_manager.running_reqs) == (offset == 12)
    assert chunks == [4, 4, 4, 2]
    assert len(locks) == 1
    assert not scheduler.prefill_manager.runnable
    assert [req.uid for req in scheduler._select_next_batch().reqs] == [1, 2]


def seed_prefix(scheduler, ids):
    indices = scheduler.cache_manager._allocate(len(ids))
    scheduler.cache_manager.prefix_cache.insert_prefix(ids, indices)
    return indices


@pytest.mark.parametrize("matched_len", [0, 7, 9])
def test_real_prefix_match_changes_query_length(make_scheduler, matched_len):
    scheduler = make_scheduler(16, cache_type="radix")
    add_decode(scheduler, 1)
    ids = torch.arange(100, 110, dtype=torch.int32)
    indices = seed_prefix(scheduler, ids[:matched_len])
    add_prefill(scheduler, 2, len(ids), ids=ids)
    batch = scheduler._select_next_batch()
    req = batch.reqs[1]
    assert batch.layout.query_lens == (1, 10 - matched_len)
    assert batch.layout.cached_lens == (4, matched_len)
    assert batch.layout.kv_lens == (5, 10)
    assert batch.layout.req_phases == ("decode", "prefill")
    assert batch.layout.sample_mask == (True, True)
    assert req.cache_handle.cached_len == matched_len
    torch.testing.assert_close(
        scheduler.table_manager.page_table[req.table_idx, :matched_len], indices
    )
    assert scheduler.cache_manager.prefix_cache.size_info.protected_size == matched_len


def test_full_input_prefix_still_leaves_one_query(make_scheduler):
    scheduler = make_scheduler(4, cache_type="radix")
    ids = torch.arange(10, dtype=torch.int32)
    seed_prefix(scheduler, ids)
    add_prefill(scheduler, 2, len(ids), ids=ids)
    batch = scheduler._select_next_batch()
    assert batch.layout.query_lens == (1,)
    assert batch.layout.cached_lens == (9,)
    assert batch.layout.sample_token_indices == (0,)


@pytest.mark.parametrize("prompt_len", [125, 300])
def test_example_layout_positions_and_physical_mapping(make_scheduler, prompt_len):
    scheduler = make_scheduler(128)
    decodes = [
        add_decode(scheduler, uid, context=context)
        for uid, context in [(1, 501), (2, 1001), (3, 2001)]
    ]
    prompt = add_prefill(scheduler, 4, prompt_len)
    batch = scheduler._select_next_batch()
    assert batch.layout.query_lens == (1, 1, 1, 125)
    assert batch.layout.kv_lens == (501, 1001, 2001, 125)
    assert batch.layout.qo_indptr == (0, 1, 2, 3, 128)
    assert batch.layout.sample_token_indices == ((0, 1, 2, 127) if prompt_len == 125 else (0, 1, 2))
    scheduler.cache_manager.allocate_paged(batch.reqs)
    batch.padded_reqs = batch.reqs
    batch.positions = _make_positions(batch, torch.device("cpu"))
    input_tuple = _make_input_tuple(batch, torch.device("cpu"))
    assert batch.positions.tolist() == [500, 1000, 2000, *range(125)]
    assert (
        input_tuple[0].tolist()
        == [req.table_idx for req in decodes] + [batch.reqs[3].table_idx] * 125
    )
    expected_ids = [int(req.input_ids[-1]) for req in decodes] + prompt[:125].tolist()
    assert scheduler.table_manager.token_pool[input_tuple].tolist() == expected_ids
    all_kv = torch.cat(
        [scheduler.table_manager.page_table[req.table_idx, : req.device_len] for req in batch.reqs]
    )
    assert all_kv.min() >= 0
    assert all_kv.max() < scheduler.cache_manager.num_pages
    assert all_kv.unique().numel() == all_kv.numel()
    assert scheduler.table_manager.page_table[input_tuple].unique().numel() == 128


def test_layout_snapshot_survives_mutable_request_advance(make_scheduler):
    scheduler = make_scheduler(8)
    add_decode(scheduler, 1)
    add_prefill(scheduler, 2, 20)
    batch = scheduler._select_next_batch()
    layout = batch.layout
    advance_request_state(scheduler, batch)
    assert layout.query_lens == (1, 7)
    assert layout.cached_lens == (4, 0)
    assert layout.kv_lens == (5, 7)
    assert layout.sample_mask == (True, False)
    assert batch.reqs[1].cached_len == 7 and batch.reqs[1].device_len == 8
    with pytest.raises(FrozenInstanceError):
        layout.sample_mask = (True, True)


def test_sampling_mapping_handles_intermediate_rows_between_valid_rows(make_scheduler):
    # The layout contract supports arbitrary row positions, including future policies.
    scheduler = make_scheduler()
    decode = add_decode(scheduler, 1)
    add_prefill(scheduler, 2, 20)
    intermediate = scheduler.prefill_manager.schedule_next_batch(3).reqs[0]
    final = Req(torch.tensor([1]), 10, 0, 1, 3, SamplingParams(), decode.cache_handle)
    layout = BatchLayout.capture([intermediate, decode, final], ("prefill", "decode", "prefill"))
    assert layout.sample_mask == (False, True, True)
    assert layout.sample_req_indices == (1, 2)
    assert layout.sample_token_indices == (3, 4)
    assert tuple(layout.request_uids[i] for i in layout.sample_req_indices) == (1, 3)


def test_no_slots_keeps_all_decode_and_pending_prefill(make_scheduler):
    scheduler = make_scheduler(slots=1)
    req = add_decode(scheduler, 1)
    add_prefill(scheduler, 2, 3)
    batch = scheduler._select_next_batch()
    assert batch.is_decode and batch.reqs == [req]
    assert scheduler.prefill_manager.runnable
    assert scheduler.table_manager.available_size == 0


@pytest.mark.parametrize("pages,expected_prefill", [(9, False), (10, True)])
def test_decode_kv_reservation_keeps_admission_boundary(make_scheduler, pages, expected_prefill):
    scheduler = make_scheduler(16, pages=pages)
    add_decode(scheduler, 1, context=2, output_len=3)
    add_prefill(scheduler, 2, 4, output_len=2)
    batch = scheduler._select_next_batch()
    # 1 historical KV token used; reserve 3 decode + 4 input + 2 output tokens.
    assert batch.is_mixed == expected_prefill
    assert scheduler.prefill_manager.runnable != expected_prefill


def test_prefills_reserve_complete_input_and_output_not_just_chunk(make_scheduler):
    scheduler = make_scheduler(2, pages=12)
    add_prefill(scheduler, 1, 10, output_len=3)
    assert scheduler._select_next_batch() is None
    assert scheduler.table_manager.available_size == 16
    assert scheduler.prefill_manager.pending_list[0].chunked_req is None


def test_multiple_prefills_accumulate_kv_reservations(make_scheduler):
    scheduler = make_scheduler(8, pages=10)
    add_prefill(scheduler, 1, 4, output_len=2)
    add_prefill(scheduler, 2, 4, output_len=2)
    batch = scheduler._select_next_batch()
    assert batch.layout.request_uids == (1,)
    assert [req.uid for req in scheduler.prefill_manager.pending_list] == [2]


def test_prefix_lock_capacity_recheck_rolls_back(make_scheduler):
    scheduler = make_scheduler(16, pages=16, cache_type="radix")
    ids = torch.arange(12, dtype=torch.int32)
    seed_prefix(scheduler, ids[:8])
    add_prefill(scheduler, 1, len(ids), output_len=5, ids=ids)
    assert scheduler.cache_manager.available_size == 16
    assert scheduler._select_next_batch() is None  # 4 + 5 > 8 free after locking prefix
    assert scheduler.cache_manager.available_size == 16
    assert scheduler.cache_manager.prefix_cache.size_info.protected_size == 0
    assert scheduler.cache_manager.prefix_cache.size_info.evictable_size == 8
    assert scheduler.table_manager.available_size == 16
    assert scheduler.prefill_manager.pending_list[0].chunked_req is None


def test_completion_removal_and_abort_change_selected_set(make_scheduler):
    scheduler = make_scheduler()
    done = add_decode(scheduler, 1, output_len=1)
    active = add_decode(scheduler, 2)
    done.complete_one()
    scheduler.decode_manager.filter_reqs([])
    assert scheduler._select_next_batch().reqs == [active]
    assert scheduler.decode_manager.abort_req(2) is active
    assert scheduler._select_next_batch() is None
    assert scheduler.decode_manager.abort_req(2) is None


@pytest.mark.parametrize("partial", [False, True])
def test_pending_and_partial_abort_release_once_and_reuse_slot(make_scheduler, partial):
    scheduler = make_scheduler(3, slots=1, cache_type="radix")
    add_prefill(scheduler, 1, 10)
    if partial:
        batch = scheduler._select_next_batch()
        advance_request_state(scheduler, batch)
        assert scheduler.table_manager.available_size == 0
    scheduler._process_one_msg(AbortBackendMsg(1))
    scheduler._process_one_msg(AbortBackendMsg(1))
    assert scheduler.table_manager.available_size == 1
    assert scheduler._select_next_batch() is None
    scheduler.cache_manager.check_integrity()
    add_prefill(scheduler, 2, 2)
    batch = scheduler._select_next_batch()
    assert batch.layout.request_uids == (2,)
    assert scheduler.table_manager.available_size == 0


def test_disabled_switch_restores_original_prefill_first(make_scheduler):
    scheduler = make_scheduler(5, enabled=False)
    decode = add_decode(scheduler, 1)
    add_prefill(scheduler, 2, 3)
    batch = scheduler._select_next_batch()
    assert batch.is_prefill and [req.uid for req in batch.reqs] == [2]
    assert batch.layout is None
    assert scheduler._select_next_batch().reqs == [decode]
    assert scheduler._select_next_batch().layout is None


def mixed_config(**kwargs):
    values = {
        "model_path": "not-loaded",
        "tp_info": DistributedInfo(0, 1),
        "dtype": torch.bfloat16,
        "enable_mixed_batching": True,
        "attention_backend": "fi",
        "page_size": 1,
    }
    values.update(kwargs)
    return SchedulerConfig(**values)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"attention_backend": "auto"}, "explicit"),
        ({"attention_backend": "fa,fi"}, "explicit"),
        ({"attention_backend": "trtllm"}, "explicit"),
        ({"attention_backend": "fa"}, "explicit"),
        ({"page_size": 4}, "page_size=1"),
        ({"tp_info": DistributedInfo(0, 2)}, "TP=1"),
        ({"max_extend_tokens": 0}, "total token budget"),
        ({"max_extend_tokens": -1}, "total token budget"),
    ],
)
def test_unsupported_mixed_configuration_is_explicit(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mixed_config(**kwargs)


def test_default_config_keeps_baseline_compatibility():
    config = SchedulerConfig("not-loaded", DistributedInfo(0, 1), torch.bfloat16)
    assert config.enable_mixed_batching is False
    assert replace(config, attention_backend="auto", page_size=4).page_size == 4


def test_cli_mixed_flag_and_runtime_gate_do_not_initialize_model():
    config, _ = parse_args(
        [
            "--model",
            "not-loaded",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "fi",
            "--enable-mixed-batching",
            "--max-extend-length",
            "128",
        ]
    )
    assert config.enable_mixed_batching is True and config.max_extend_tokens == 128
    with pytest.raises(NotImplementedError, match="stage 1 implements mixed batch selection only"):
        Scheduler(config)


def test_prepare_gate_blocks_all_new_layouts_before_allocation(make_scheduler):
    scheduler = make_scheduler()
    add_decode(scheduler, 1)
    batch = scheduler._select_next_batch()
    # No Engine exists, so this must fail before graph padding or page allocation.
    with pytest.raises(NotImplementedError, match="cannot execute"):
        scheduler._prepare_batch(batch)
    with pytest.raises(NotImplementedError, match="cannot execute"):
        scheduler._prepare_batch(Batch(batch.reqs, "mixed"))


def test_mixed_never_qualifies_for_existing_decode_graph(make_scheduler):
    scheduler = make_scheduler()
    req = add_decode(scheduler, 1)
    add_prefill(scheduler, 2, 4)
    batch = scheduler._select_next_batch()
    runner = GraphRunner.__new__(GraphRunner)
    runner.max_graph_bs = 16
    assert not runner.can_use_cuda_graph(batch)
    assert runner.can_use_cuda_graph(Batch([req], "decode"))
    assert not runner.can_use_cuda_graph(Batch(batch.reqs[1:], "prefill"))


def test_last_output_token_still_has_a_sampling_row(make_scheduler):
    scheduler = make_scheduler()
    add_decode(scheduler, 1, output_len=1)
    add_prefill(scheduler, 2, 2, output_len=1)
    batch = scheduler._select_next_batch()
    assert batch.layout.sample_mask == (True, True)
    advance_request_state(scheduler, batch)
    assert all(not req.can_decode for req in batch.reqs)
    assert batch.layout.sample_mask == (True, True)
    assert batch.layout.sample_req_indices == (0, 1)
    assert batch.layout.sample_token_indices == (0, 2)


@pytest.mark.parametrize(
    "change",
    [
        {"query_lens": (0,)},
        {"query_lens": (6,)},
        {"query_lens": (2,)},
        {"sample_mask": (False,)},
        {"kv_lens": ()},
        {"req_phases": ("mixed",)},
    ],
)
def test_layout_rejects_invalid_lengths_phases_or_decode_sampling(make_scheduler, change):
    scheduler = make_scheduler()
    req = add_decode(scheduler, 1)
    layout = BatchLayout.capture([req], ("decode",))
    with pytest.raises(ValueError):
        replace(layout, **change)


def test_layout_rejects_duplicate_uid_or_slot(make_scheduler):
    scheduler = make_scheduler()
    first, second = [add_decode(scheduler, uid) for uid in [1, 2]]
    layout = BatchLayout.capture([first, second], ("decode", "decode"))
    with pytest.raises(ValueError, match="unique"):
        replace(layout, request_uids=(1, 1))
    with pytest.raises(ValueError, match="unique"):
        replace(layout, table_indices=(0, 0))


def test_public_selection_entrypoint_works_without_scheduler_engine(make_scheduler):
    scheduler = make_scheduler(8)
    add_decode(scheduler, 1)
    add_prefill(scheduler, 2, 3)
    batch = schedule_mixed_batch(scheduler.decode_manager, scheduler.prefill_manager, 8)
    assert batch.is_mixed
    assert batch.layout.query_lens == (1, 3)
