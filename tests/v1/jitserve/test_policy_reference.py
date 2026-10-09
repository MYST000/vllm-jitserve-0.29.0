# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential policy tests against actual JITServe 84584a1 function bodies.

Set JITSERVE_REFERENCE to a clean checkout. Only policy/state-machine classes
are AST-loaded, avoiding the old engine's binary/import dependencies. Sequence
fixtures represent queue inputs; this suite makes no model execution claims.
"""

import ast
import os
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm.v1.jitserve.execution import Candidate, reserve
from vllm.v1.jitserve.policy import (
    SLO,
    GroupController,
    Policy,
    RequestState,
    RequestType,
)


@pytest.fixture
def reference():
    root = os.environ.get("JITSERVE_REFERENCE")
    if not root:
        pytest.skip("Set JITSERVE_REFERENCE to test original source")
    assert root is not None
    clock = NS(now=100.0)
    status = NS(WAITING=0, RUNNING=1, SWAPPED=2, FINISHED=3)
    namespace = dict(
        __name__=__name__,
        dataclass=dataclass,
        field=field,
        ABC=ABC,
        abstractmethod=abstractmethod,
        deque=deque,
        time=NS(time=lambda: clock.now),
        logger=NS(info=lambda *a: None),
        RequestType=RequestType,
        RequestPhaseWeight=NS(DECODE=NS(value=8)),
        SequenceStatus=status,
        PreemptionMode=NS(SWAP=0, RECOMPUTE=1),
        AllocStatus=NS(OK=0, NEVER=1, LATER=2),
        Scheduler=object,
        DEFAULT_SCHEDULE_INTERVAL=0.02,
    )
    files = {
        "../third_party/vllm/vllm/core/scheduler.py": {
            "SchedulingBudget",
            "ScheduledSequenceGroup",
        },
        "scheduler/policy.py": {"BasePolicy", "SLOPolicy"},
        "scheduler/slo_scheduler.py": {"SLOScheduler"},
        "slo_tracker/slo_tracker.py": {"SLOTracker"},
    }
    for name, classes in files.items():
        path = Path(root) / "jitserve" / name
        tree = ast.parse(path.read_text(), filename=str(path))
        body = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in classes
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *body,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return NS(clock=clock, status=status, **namespace)


def test_execution_reservations_stop_and_commit_order_match_source(reference):
    """Run the original execution pass with controlled resource availability."""
    rng = random.Random(598)
    for case in range(240):
        inputs = [
            Candidate(
                str(i),
                rng.choice(["waiting", "swapped", "running"]),
                rng.randint(0, 40) if i % 2 else 1,
                bool(i % 2),
                rng.randint(1, 8),
            )
            for i in range(12)
        ]
        tokens, slots, blocks = (
            rng.randint(1, 48),
            rng.randint(1, 12),
            rng.randint(1, 64),
        )
        old = reference.SLOScheduler.__new__(reference.SLOScheduler)
        by_id = {c.request_id: c for c in inputs}
        objects = []
        for item in inputs:
            seq = NS(get_num_new_tokens=lambda item=item: item.remaining)
            obj = NS(
                request_id=item.request_id,
                seqs=[seq],
                is_prefill=lambda item=item: item.prefill,
                is_finished=lambda: False,
                get_num_uncomputed_tokens=lambda item=item: item.remaining,
                get_max_num_running_seqs=lambda: 1,
                get_seqs=lambda status, seq=seq: [seq],
            )
            objects.append(obj)
        old.waiting = [s for s in objects if by_id[s.request_id].status == "waiting"]
        old.swapped = [s for s in objects if by_id[s.request_id].status == "swapped"]
        old.running = [s for s in objects if by_id[s.request_id].status == "running"]
        budget = reference.SchedulingBudget(tokens, slots)
        reservations: dict[str, int] = {}
        commits: list[str] = []
        final_tokens: dict[str, int] = {}
        original_add = budget.add_num_batched_tokens

        def add_tokens(
            key, count, reservations=reservations, original_add=original_add
        ):
            reservations[key] = count
            original_add(key, count)

        budget.add_num_batched_tokens = add_tokens
        old._get_num_new_tokens = lambda seq, status, chunking, budget, by_id=by_id: (
            min(by_id[seq.request_id].remaining, budget.remaining_token_budget())
        )
        old._get_num_lookahead_slots = lambda *a: 0
        old._get_prompt_limit = lambda seq: 10000
        old.block_manager = NS(
            get_max_gpu_num_blocks_number=lambda blocks=blocks: blocks,
            get_num_blocks_number=lambda seq, *a, by_id=by_id: (
                by_id[seq.request_id].blocks
            ),
            can_allocate=lambda *a, **kw: 0,
            can_swap_in=lambda *a: 0,
        )
        old._preempt = lambda *a: 0
        old._can_append_slots = lambda *a: True
        old._append_slots = lambda seq, *a, commits=commits: commits.append(
            seq.request_id
        )
        old.cache_id = 0
        old._scheduled_seq_group_cache = [
            NS(get_object=lambda: reference.ScheduledSequenceGroup(None, 0))
        ]

        def swapped(seq, *a, commits=commits):
            commits.append(seq.request_id)
            return NS(
                is_prefill=seq.is_prefill(),
                seq_group=seq,
                blocks_to_swap_in=[],
                blocks_to_copy=[],
            )

        def waiting(seq, count, *a, commits=commits, final_tokens=final_tokens):
            commits.append(seq.request_id)
            final_tokens[seq.request_id] = count
            return NS(seq_group=seq)

        old._schedule_slo_swapped = swapped
        old._schedule_slo_prefills = waiting
        running = NS(
            swapped_out=[],
            preempted=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            leftover_running=[],
            prefill_seq_groups=[],
            prefill_seq_groups_list=[],
            decode_seq_groups=[],
            decode_seq_groups_list=[],
        )
        prefill = NS(seq_groups=[], ignored_seq_groups=[])
        restored = NS(
            prefill_seq_groups=[],
            decode_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
        )
        old._schedule_process_list(objects, running, prefill, restored, budget, True)
        actual = reserve(inputs, tokens, slots, blocks)
        assert actual.reserved == reservations, case
        assert actual.preempt == [s.request_id for s in running.swapped_out], case
        assert actual.commit_order == commits, case
        remaining = actual.remaining_tokens
        for key in actual.commit_order:
            item = by_id[key]
            if item.status == "waiting":
                extra = min(item.remaining - actual.reserved[key], remaining)
                remaining -= extra
                assert final_tokens[key] == actual.reserved[key] + extra
        assert budget.remaining_token_budget() == remaining


def sequence(state, status=0):
    seq = NS(
        get_prompt_len=lambda: state.prompt_len,
        get_output_len=lambda: state.output_len,
        get_decode_len=lambda: state.output_len,
        status=status,
    )
    seq.is_finished = lambda: seq.status == 3
    return NS(
        request_id=state.request_id,
        request_type=state.slo.request_type,
        collection_id=state.slo.collection_id,
        arrival_time=state.arrival,
        deadline=state.slo.deadline - state.arrival,
        TTFT_constraint=state.slo.ttft,
        TBT_constraint=state.slo.tbt,
        predict_output_length=state.slo.output_len,
        request_info=NS(output_len=state.predicted_len),
        first_seq=seq,
        seqs=[seq],
        concord_metrics=NS(
            TTFT=state.ttft,
            service_gain=state.service_gain,
            arrival_time=state.arrival,
            prompt_len=0,
            output_len=0,
            TTLT=None,
            last_schedule_time=None,
            TBT=[],
        ),
    )


def test_reward_prediction_cache_and_interval_match_source(reference):
    rng = random.Random(84584)
    old = reference.SLOPolicy(schedule_interval=4, penalty_factor=2)
    new = Policy(interval=4, penalty=2)
    states = []
    for i in range(36):
        kind = RequestType(i % 3)
        state = RequestState(
            str(i),
            SLO(
                kind,
                rng.uniform(70, 180),
                rng.randint(1, 90),
                f"{kind}-{i // 9}",
                ttft=0.2,
                tbt=0.01,
            ),
            rng.uniform(0, 20),
            rng.randint(1, 500),
        )
        states.append(state)
        old.add_seq_group(sequence(state))
    peers = [seq for group in old.seq_group_dict.values() for seq in group]
    by_id = {seq.request_id: seq for seq in peers}
    for step in range(1, 81):
        reference.clock.now += rng.uniform(0.001, 0.1)
        assert new.tick(step, reference.clock.now) == old.update_schedule_count()
        assert new.unit_time == pytest.approx(old._get_interval_time())
        for state in states:
            state.output_len += rng.randrange(3)
            state.predicted_len = rng.randint(1, 100)  # Including underpredictions.
            state.ttft = rng.choice([None, 0.1])
            state.service_gain = rng.choice([0, 0.2, 3.0])
            seq = by_id[state.request_id]
            seq.request_info.output_len = state.predicted_len
            seq.concord_metrics.TTFT = state.ttft
            seq.concord_metrics.service_gain = state.service_gain
        actual = new.priorities(states, reference.clock.now)
        for state in states:
            assert actual[state.request_id] == pytest.approx(
                old.get_priority(by_id[state.request_id])
            )


def test_latency_observation_during_prefill_pause_and_resume_matches_source(reference):
    state = RequestState("r", SLO(RequestType.LATENCY, 10, 8, "r", 0.1, 0.03), 0, 32)
    seq = sequence(state)
    tracker = reference.SLOTracker(penalty_factor=2)
    # No prompt reward is counted twice; a no-output observation resets gain.
    for now, output in [
        (0.01, 0),
        (0.02, 0),
        (0.1, 1),
        (0.13, 2),
        (0.2, 2),
        (0.9, 3),
        (1.2, 6),
    ]:
        state.output_len = output
        tracker._update_metrics(seq, now)
        state.observe(output, now, 2)
        assert state.ttft == seq.concord_metrics.TTFT
        assert state.service_gain == pytest.approx(seq.concord_metrics.service_gain)


@pytest.mark.parametrize("strategy", ["length_bin", "sliding_window"])
def test_selection_activation_pending_and_batch_trace_matches_source(
    reference, strategy
):
    rng = random.Random(1188)
    old = reference.SLOScheduler.__new__(reference.SLOScheduler)
    old.search_strategy = strategy
    old.max_batch_size = 4
    old.max_swaps_per_iter = 2
    old.top_k_selection = 3
    old.scheduler_config = NS(max_num_seqs=4, top_k_selection=3)
    old.policy = NS(schedule_interval=4, interval_time=0.02)
    old._cur_rank = 0
    old._exec_order = []
    old._bin_bounds = []
    old._bins_by_index = []
    old._current_bin_queue = deque()
    old._pending_swap_out_queue = deque()
    old._current_bounds = (0, 0)
    old._current_is_last = False
    old._preempt = lambda seq, blocks, mode: mode
    new = GroupController(4, 3, 2, strategy)
    states = [
        RequestState(
            str(i),
            SLO(RequestType(i % 3), 100, 50, str(i), 1, 0.03),
            0,
            8 * (i % 7 + 1),
        )
        for i in range(24)
    ]
    objects = {s.request_id: sequence(s) for s in states}
    running, swapped, waiting = states[:8], states[8:12], states[12:]
    for round_id in range(60):
        # Discrete scores exercise stable ties. Queue order matters.
        scores = {
            s.request_id: (-float(rng.randint(1, 6)), rng.random()) for s in states
        }
        old._service_gain = lambda seq, scores=scores: scores[seq.request_id][0]
        old.waiting = deque(objects[s.request_id] for s in waiting)
        old.swapped = deque(objects[s.request_id] for s in swapped)
        old.running = deque(objects[s.request_id] for s in running)
        for group, status in [(old.waiting, 0), (old.running, 1), (old.swapped, 2)]:
            for seq in group:
                seq.first_seq.status = status
        old.policy.interval_time = rng.choice([0.01, 0.02, 0.1])
        if strategy == "sliding_window":
            old._recompute_bins_sliding_window()
        elif not old._pending_swap_out_queue:
            old._recompute_bins()
        old._inject_current_bin()
        result = NS(blocks_to_swap_out=[], swapped_out=[], preempted=[])
        _, count, _ = old._preempt_pending(result)
        process, min_tbt, activate, _ = old._build_process_list(result, count)
        budget = NS(max_num_seqs=max(len(running), old.scheduler_config.max_num_seqs))
        old._adjust_batch_limits(process, min_tbt, budget)
        plan = new.plan(waiting, swapped, running, scores, old.policy.interval_time)
        ids = lambda seqs: [s.request_id for s in seqs]
        assert plan.activate == ids(activate)
        assert plan.preempt == ids(result.swapped_out)
        assert plan.process == ids(process)
        assert plan.batch_budget == budget.max_num_seqs
        assert plan.admission_limit == old.scheduler_config.max_num_seqs
        assert new.bounds == old._current_bounds
        assert new.is_last == old._current_is_last
        assert list(new.pending) == ids(old._pending_swap_out_queue)
        assert list(new.current) == ids(old._current_bin_queue)
        # Advance the external queue state, including departures/cancellations.
        active = set(plan.activate)
        victims = set(plan.preempt)
        departed = running[:1] if round_id % 3 == 0 else []
        running = [
            s for s in running if s.request_id not in victims and s not in departed
        ] + [s for s in waiting + swapped if s.request_id in active]
        waiting = [s for s in waiting if s.request_id not in active] + departed
        swapped = [s for s in swapped if s.request_id not in active] + [
            s for s in states if s.request_id in victims
        ]
        for state in running:
            state.output_len += 1


def test_cpu_watermark_hysteresis_matches_source(reference):
    old = reference.SLOScheduler.__new__(reference.SLOScheduler)
    old.prev_no_swap_space = False
    current = NS(free=100)
    old.block_manager = NS(
        num_total_cpu_blocks=100,
        cpu_watermarks=0.2,
        get_num_free_cpu_blocks=lambda: current.free,
    )
    new = GroupController(4, 2, 1, "length_bin")
    for free in [100, 20, 19, 50, 80, 81, 20, 0, 80, 100]:
        current.free = free
        assert new.enough_swap_space(free / 100) == old._test_enough_swap_space()
