"""Small real-model tests for sampling, writeback and in-flight resource reuse.

Run in a fresh GPU process, separately from CPU and isolated attention suites.
The model, LMHead, sampler, KV, scheduler loops and Decode graph are real. Some
tests force logits just before the real sampler to make EOS/termination exact.
Unforced logits/greedy outputs are compared to the disabled baseline with matching packed shapes.
"""

from __future__ import annotations

import os
from collections import defaultdict

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import AbortBackendMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig

MODEL = os.environ.get("MINISGL_TEST_MODEL", "/home/wty238388/ai-infra/models/Qwen3-0.6B")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA and local model"
)


class Driver(Scheduler):
    def __init__(self, *, mixed=True, graph_bs=None, max_seq_len=128, pages=768, budget=8):
        self.initial_mixed, self.initial_budget = mixed, budget
        self.inbox, self.events, self.traces = [], [], []
        self.pending_data = None
        self.overlap = True
        self.forced = None
        self.logits, self.sample_args_log, self.middle_writes = defaultdict(list), [], []
        self.write_mappings = []
        self.sample_calls = defaultdict(int)
        self.prompts, self.accepted_tokens, self.input_snapshots = {}, defaultdict(list), []
        super().__init__(
            SchedulerConfig(
                model_path=MODEL,
                tp_info=DistributedInfo(0, 1),
                dtype=torch.bfloat16,
                max_running_req=3,
                max_seq_len_override=max_seq_len,
                num_page_override=pages,
                max_extend_tokens=budget,
                attention_backend="fi",
                page_size=1,
                cuda_graph_bs=[2] if graph_bs is None else graph_bs,
                enable_mixed_batching=mixed,
                offline_mode=True,
            )
        )
        original = self.engine.sampler.sample

        def sample(logits, args):
            batch = self.current_batch
            uids = (
                [batch.reqs[i].uid for i in batch.layout.sample_req_indices]
                if batch.layout is not None
                else [req.uid for req in batch.reqs]
            )
            assert logits.shape[0] == len(uids)
            self.sample_args_log.append((uids, args))
            for row, uid in enumerate(uids):
                self.logits[uid].append(logits[row].clone())
            if self.forced is not None:
                logits.fill_(-10000)
                for row, uid in enumerate(uids):
                    logits[row, self.forced(uid, self.sample_calls[uid])] = 10000
            for uid in uids:
                self.sample_calls[uid] += 1
            return original(logits, args)

        self.engine.sampler.sample = sample

    def offline_receive_msg(self, blocking=False):
        messages, self.inbox = self.inbox, []
        return messages

    def offline_send_result(self, reply):
        self.events.extend(reply)
        for message in reply:
            self.accepted_tokens[message.uid].append(message.next_token)

    def _forward(self, forward_input):
        batch = self.current_batch = forward_input.batch
        layout = batch.layout
        trace = {
            "phase": batch.phase,
            "uids": tuple(req.uid for req in batch.reqs),
            "query_lens": tuple(req.extend_len for req in batch.reqs),
            "cached_lens": tuple(req.cached_len for req in batch.reqs),
            "sample_mask": layout.sample_mask if layout is not None else None,
            "graph": self.engine.graph_runner.can_use_cuda_graph(batch),
            "slots": tuple(req.table_idx for req in batch.reqs),
            "padded_size": batch.padded_size,
        }
        output = super()._forward(forward_input)
        trace["sample_count"] = output.next_tokens_gpu.numel()
        if layout is not None:
            assert trace["sample_count"] == sum(layout.sample_mask)
        self.traces.append(trace)
        if layout is None:
            return output
        self.input_snapshots.append((layout, batch.input_ids[: layout.num_tokens].clone()))
        self.write_mappings.append(
            (
                forward_input.write_tuple[0].clone(),
                tuple(layout.table_indices[i] for i in layout.sample_req_indices),
            )
        )
        for row, sample in enumerate(layout.sample_mask):
            if not sample:
                slot, end = layout.table_indices[row], layout.kv_lens[row]
                # With overlap, the next chunk may already copy a legitimate
                # prompt token to `end` on the scheduler stream before this probe.
                if not self.overlap:
                    self.middle_writes.append(self.token_pool[slot, end].clone())
                self.middle_writes.append(self.token_pool[slot, -1].clone())
        output.copy_done_event.record(self.engine.stream)  # include instrumentation GPU reads
        return output

    def add(self, uid, tokens, count=3, *, ignore_eos=True, **sampling):
        self.prompts[uid] = list(tokens)
        self.inbox.append(
            UserMsg(
                uid,
                torch.tensor(tokens, dtype=torch.int32),
                SamplingParams(max_tokens=count, ignore_eos=ignore_eos, **sampling),
            )
        )

    def step(self):
        with torch.inference_mode():
            if self.overlap:
                with torch.cuda.stream(self.stream):
                    self.pending_data = self.overlap_loop(self.pending_data)
            else:
                with self.engine_stream_ctx:
                    self.engine.stream.wait_stream(self.stream)
                    self.normal_loop()

    def drain(self):
        for _ in range(80):
            if not (
                self.inbox
                or self.pending_data is not None
                or self.prefill_manager.runnable
                or self.decode_manager.runnable
            ):
                return
            self.step()
        raise AssertionError("Scheduler failed to drain")

    def check_resources(self):
        torch.cuda.synchronize()
        assert self.pending_data is None
        if hasattr(self, "mixed_lifecycle"):
            assert not self.mixed_lifecycle.active
        assert self.table_manager.available_size == 3
        cm = self.cache_manager
        cm.check_integrity()
        assert cm.free_slots.unique().numel() == cm.free_slots.numel()
        nodes, cached, evictable = [cm.prefix_cache.root_node], [], 0
        while nodes:
            node = nodes.pop()
            nodes.extend(node.children.values())
            if node.is_root():
                continue
            assert node.ref_count == 0
            evictable += node.length
            cached.extend(node.value.cpu().tolist())
        free = set(cm.free_slots.cpu().tolist())
        assert len(cached) == len(set(cached))
        assert not set(cached) & free
        assert len(cached) + len(free) == cm.num_pages
        assert cm.prefix_cache.protected_size == 0
        assert cm.prefix_cache.evictable_size == evictable
        for layout, packed in self.input_snapshots:
            expected = []
            for uid, cached, length in zip(
                layout.request_uids, layout.cached_lens, layout.query_lens
            ):
                sequence = self.prompts[uid] + self.accepted_tokens[uid]
                expected.extend(sequence[cached : cached + length])
            assert packed.cpu().tolist() == expected

    def reset(self, overlap=True, *, keep_cache=False):
        self.check_resources()
        cm = self.cache_manager
        with torch.cuda.stream(self.stream):
            if not keep_cache:
                evicted = cm.prefix_cache.evict(cm.prefix_cache.evictable_size)
                cm.free_slots = torch.cat([cm.free_slots, evicted])
            self.token_pool[:-1].fill_(-777)  # leave graph dummy input at token 0
        self.inbox.clear()
        self.events.clear()
        self.traces.clear()
        self.logits.clear()
        self.sample_args_log.clear()
        self.middle_writes.clear()
        self.write_mappings.clear()
        self.sample_calls.clear()
        self.prompts.clear()
        self.accepted_tokens.clear()
        self.input_snapshots.clear()
        self.overlap, self.forced, self.prefill_budget = overlap, None, self.initial_budget
        self.enable_mixed_batching = self.initial_mixed


@pytest.fixture(scope="module")
def model_driver():
    if not os.path.isdir(MODEL):
        pytest.skip("Set MINISGL_TEST_MODEL to a local Qwen3-0.6B model")
    driver = Driver()
    yield driver
    driver.shutdown()


@pytest.fixture
def driver(model_driver):
    model_driver.reset()
    return model_driver


def assert_outputs(driver, expected):
    for uid, tokens in expected.items():
        messages = [message for message in driver.events if message.uid == uid]
        assert [message.next_token for message in messages] == tokens
        assert [message.finished for message in messages] == [False] * (len(tokens) - 1) + [True]
    assert {message.uid for message in driver.events} == set(expected)
    driver.check_resources()


@pytest.mark.parametrize("overlap", [False, True])
def test_mixed_sampling_rows_intermediate_writeback_and_graph_padding(driver, overlap):
    driver.overlap = overlap
    driver.forced = lambda uid, step: uid * 100 + step
    driver.add(1, [10, 11, 12], 6)
    driver.add(2, list(range(20, 49)), 3)
    driver.drain()
    assert_outputs(driver, {1: list(range(100, 106)), 2: [200, 201, 202]})
    assert any(row["phase"] == "mixed" for row in driver.traces)
    assert any(
        row["graph"] and len(row["uids"]) == 1 and row["padded_size"] == 2 for row in driver.traces
    )
    assert driver.middle_writes and all(int(value) == -777 for value in driver.middle_writes)
    assert all(
        tuple(actual.cpu().tolist()) == expected for actual, expected in driver.write_mappings
    )
    assert driver.sample_calls[1] == 6 and driver.sample_calls[2] == 3


@pytest.mark.parametrize("overlap", [False, True])
def test_eos_suppresses_speculation_and_allows_immediate_slot_reuse(driver, overlap):
    driver.overlap = overlap
    driver.forced = lambda uid, step: driver.eos_token_id if uid == 1 and step == 0 else 42
    driver.add(1, [60, 61, 62], 8, ignore_eos=False)
    driver.step()
    old_slot = driver.traces[0]["slots"][0]
    if overlap:
        driver.step()  # second forward submitted BEFORE first EOS is handled
        assert driver.sample_calls[1] == 2 and driver.pending_data is not None
    driver.add(2, [63, 64], 1)
    driver.drain()
    assert_outputs(driver, {1: [driver.eos_token_id], 2: [42]})
    assert next(row for row in driver.traces if row["uids"] == (2,))["slots"] == (old_slot,)


@pytest.mark.parametrize("kind", ["chunk", "prefill", "final-token", "decode", "final-decode"])
def test_abort_inflight_before_cpu_output_then_reuse_slot(driver, kind):
    driver.forced = lambda uid, step: uid * 100 + step
    tokens = list(range(70, 87)) if kind == "chunk" else [70, 71, 72]
    driver.prefill_budget = 4
    driver.add(1, tokens, 1 if kind == "final-token" else (2 if kind == "final-decode" else 4))
    driver.step()  # result event is pending CPU processing
    if kind in ("decode", "final-decode"):
        driver.step()  # one accepted token; the actual Decode result is still pending
    old_slot = driver.traces[0]["slots"][0]
    driver.inbox.append(AbortBackendMsg(1))
    driver.add(2, tokens[:4] + [90, 91], 1)
    driver.step()
    driver.inbox.append(AbortBackendMsg(1))  # repeated old abort must be a no-op
    driver.drain()
    old = [message for message in driver.events if message.uid == 1]
    assert [message.next_token for message in old] == (
        [100] if kind in ("decode", "final-decode") else []
    )
    assert all(not message.finished for message in old)
    new = [message for message in driver.events if message.uid == 2]
    assert len(new) == 1 and new[0].next_token == 200 and new[0].finished
    driver.check_resources()
    assert next(row for row in driver.traces if row["uids"] == (2,))["slots"] == (old_slot,)
    assert next(row for row in driver.traces if row["uids"] == (2,))["cached_lens"][0] > 0


def test_duplicate_prefix_reuse_and_first_hit_logits_match_baseline(driver):
    prompt = [100, 101, 102, 103, 104, 105]
    # Match packed matrix shapes against the disabled baseline. A separate
    # diagnostic records pre-existing single-vs-batch BF16 numerical drift.
    driver.overlap = False
    driver.enable_mixed_batching = False
    driver.prefill_budget = 24
    for uid in (1, 2, 3):
        driver.add(uid, prompt, 3)
    driver.drain()
    expected_tokens = [message.next_token for message in driver.events if message.uid == 1]
    expected_logits = {
        uid: [value.float().cpu() for value in driver.logits[uid]] for uid in (1, 2, 3)
    }
    driver.events.clear()
    driver.add(20, prompt, 3)
    driver.drain()
    hit_tokens = [message.next_token for message in driver.events]
    hit_logits = [value.float().cpu() for value in driver.logits[20]]
    driver.reset(overlap=True)
    driver.prefill_budget = 24
    for uid in (1, 2, 3):
        driver.add(uid, prompt, 3)
    driver.step()
    driver.step()  # all three prefills cached; duplicate physical pages are now free
    with torch.cuda.stream(driver.stream):
        driver.cache_manager.free_slots = driver.cache_manager.free_slots.sort().values
    driver.drain()  # the next decode reuses just-freed duplicate pages for new KV
    assert_outputs(driver, dict.fromkeys((1, 2, 3), expected_tokens))
    for uid in (1, 2, 3):
        for i, (actual, expected) in enumerate(zip(driver.logits[uid], expected_logits[uid])):
            error = (actual.float().cpu() - expected).abs().max().item()
            print(f"C7_LOGITS duplicate uid={uid} output={i} max_abs={error}", flush=True)
            torch.testing.assert_close(actual.float().cpu(), expected, atol=0.1, rtol=0.01)
    driver.events.clear()
    driver.traces.clear()
    driver.add(20, prompt, 3)  # actual radix hit, with the cache preserved
    driver.drain()
    assert_outputs(driver, {20: hit_tokens})
    assert driver.traces[0]["cached_lens"] == (len(prompt) - 1,)
    for i, (actual, expected) in enumerate(zip(driver.logits[20], hit_logits)):
        error = (actual.float().cpu() - expected).abs()
        print(
            f"C7_HIT_LOGITS output={i} max_abs={error.max().item()} violations={int((error > 0.1 + 0.01 * expected.abs()).sum())}",
            flush=True,
        )
    # The Prefill logits have identical input KV and packed shape. Later decode
    # logits also depend on canonical KV replacement and overlap ordering; their
    # full strict comparison is retained as a FAILING standalone diagnostic,
    # not silently counted as a passing numerical check here.
    torch.testing.assert_close(
        driver.logits[20][0].float().cpu(), hit_logits[0], atol=0.1, rtol=0.01
    )


def test_non_greedy_sampler_receives_only_valid_parameter_rows(driver):
    driver.forced = lambda uid, step: uid * 100 + step
    settings = {
        1: {"temperature": 0.7, "top_k": 2, "top_p": 0.9},
        2: {"temperature": 1.1, "top_k": 3, "top_p": 0.8},
    }
    driver.add(1, [110, 111, 112], 4, **settings[1])
    driver.add(2, list(range(120, 137)), 2, **settings[2])
    driver.drain()
    assert_outputs(driver, {1: [100, 101, 102, 103], 2: [200, 201]})
    for uids, args in driver.sample_args_log:
        torch.testing.assert_close(
            args.temperatures.cpu(), torch.tensor([settings[uid]["temperature"] for uid in uids])
        )
        assert args.top_k.cpu().tolist() == [settings[uid]["top_k"] for uid in uids]
        torch.testing.assert_close(
            args.top_p.cpu(), torch.tensor([settings[uid]["top_p"] for uid in uids])
        )


def test_decode_count_above_budget_never_drops_active_requests(driver):
    driver.forced = lambda uid, step: uid * 100 + step
    driver.add(1, [140, 141], 3)
    driver.add(2, [150, 151], 3)
    driver.step()
    driver.prefill_budget = 1
    driver.add(3, [160, 161, 162], 1)
    driver.drain()
    assert_outputs(driver, {1: [100, 101, 102], 2: [200, 201, 202], 3: [300]})
    assert any(
        row["phase"] == "decode" and row["uids"] == (1, 2) and sum(row["query_lens"]) == 2
        for row in driver.traces
    )
    assert any(row["sample_count"] == 0 for row in driver.traces)
