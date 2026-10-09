# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JITServe reward and length-group selection, independent of engine execution.

Adapted from UIUC-MLSys/JITServe at 84584a1 (Apache-2.0).
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class RequestType(IntEnum):
    LATENCY = 0
    THROUGHPUT = 1
    COLLECTIVE = 2


@dataclass(frozen=True)
class SLO:
    request_type: RequestType
    deadline: float
    output_len: int
    collection_id: str
    ttft: float = 0.0
    tbt: float = 0.0

    @classmethod
    def parse(cls, extra: dict[str, Any] | None) -> "SLO":
        data = extra or {}
        required = ("request_type", "deadline", "output_len", "collection_id")
        missing = [name for name in required if "jitserve_" + name not in data]
        if missing:
            raise ValueError("Missing JITServe parameters: " + ", ".join(missing))
        kind = data["jitserve_request_type"]
        if type(kind) is not int:
            raise ValueError("jitserve_request_type must be an integer: 0, 1 or 2")
        kind = RequestType(kind)
        deadline = float(data["jitserve_deadline"])
        output_len = data["jitserve_output_len"]
        if type(output_len) is not int or output_len <= 0:
            raise ValueError("jitserve_output_len must be a positive integer estimate")
        if not math.isfinite(deadline):
            raise ValueError("jitserve_deadline must be finite Unix epoch seconds")
        collection = data["jitserve_collection_id"]
        if not isinstance(collection, str) or not collection:
            raise ValueError("jitserve_collection_id must be a nonempty string")
        ttft = float(data.get("jitserve_ttft", 0.0))
        tbt = float(data.get("jitserve_tbt", 0.0))
        if not all(math.isfinite(x) and x >= 0 for x in (ttft, tbt)):
            raise ValueError("JITServe TTFT/TBT must be finite and nonnegative")
        if kind == RequestType.LATENCY and (ttft <= 0 or tbt <= 0):
            raise ValueError("Latency requests require positive jitserve_ttft/tbt")
        return cls(kind, deadline, output_len, collection, ttft, tbt)


@dataclass
class RequestState:
    request_id: str
    slo: SLO
    arrival: float
    prompt_len: int
    output_len: int = 0
    predicted_len: int = 0
    prediction_version: int = 0
    ttft: float | None = None
    last_output_time: float | None = None
    service_gain: float = 0.0
    observed_prompt_len: int = 0
    observed_output_len: int = 0

    def __post_init__(self) -> None:
        self.predicted_len = self.slo.output_len

    @property
    def length(self) -> int:
        return self.prompt_len + self.output_len

    def observe(self, output_len: int, now: float, penalty: float) -> None:
        delta_input = self.prompt_len - self.observed_prompt_len
        delta = output_len - self.observed_output_len
        if output_len > 0 and self.ttft is None:
            self.ttft = now - self.arrival
        self.output_len = output_len
        gain = 0.0
        if self.slo.request_type == RequestType.LATENCY:
            if self.service_gain == 0 and self.ttft is not None:
                gain += ratio(self.slo.ttft, self.ttft) ** penalty * max(delta_input, 0)
            desired = (
                (now - self.arrival - min(self.ttft, self.slo.ttft)) / self.slo.tbt
                if self.ttft is not None
                else 0.0
            )
            if delta > 0:
                gain += delta * (
                    ratio(output_len, desired) ** penalty * 8 if desired > 0 else 1
                )
        self.service_gain = gain
        self.observed_prompt_len = self.prompt_len
        self.observed_output_len = output_len
        if delta > 0:
            self.last_output_time = now


def ratio(numerator: float, denominator: float) -> float:
    return 1.0 if denominator <= 0 else max(1e-6, min(1.0, numerator / denominator))


class Policy:
    def __init__(self, penalty: float = 1.0, interval: int = 50):
        self.penalty = penalty
        self.interval = interval
        self.unit_time = 0.02
        self._samples: deque[float] = deque(maxlen=10)
        self._last_time: float | None = None
        self.cache: dict[str, tuple[float, float]] = {}

    def tick(self, step: int, now: float) -> bool:
        if step % self.interval:
            return False
        self._samples.append(
            0.02 if self._last_time is None else (now - self._last_time) / self.interval
        )
        self.unit_time = sum(self._samples) / len(self._samples)
        self._last_time = now
        self.cache.clear()
        return True

    def priorities(
        self,
        states: list[RequestState],
        now: float,
        query: list[RequestState] | None = None,
    ) -> dict[str, tuple[float, float]]:
        groups: dict[str, list[RequestState]] = defaultdict(list)
        for state in states:
            groups[state.slo.collection_id].append(state)
        result = self.cache
        requested = states if query is None else query
        for state in requested:
            if state.request_id in result:
                continue
            peers = groups[state.slo.collection_id]
            if state.slo.request_type != RequestType.COLLECTIVE:
                gain, cost = self._reward(state, peers, now)
                result[state.request_id] = (-gain / cost, cost)
            else:
                rewards = [self._reward(peer, peers, now) for peer in peers]
                cost = max(cost for _, cost in rewards)
                priority = (-sum(gain for gain, _ in rewards) / cost, cost)
                result.update((peer.request_id, priority) for peer in peers)
        return {state.request_id: result[state.request_id] for state in requested}

    def _reward(
        self, state: RequestState, peers: list[RequestState], now: float
    ) -> tuple[float, float]:
        remain = state.predicted_len - state.output_len
        if state.slo.request_type == RequestType.COLLECTIVE:
            cost = max(
                # The original SLO policy refreshes reward predictions but its
                # collective allocation estimate retains the ingress prediction.
                sum(self.unit_time * (p.slo.output_len - p.output_len) for p in peers),
                1e-6,
            )
        else:
            cost = max(self.unit_time * remain, 1e-6)
        served = now - state.arrival
        budget = state.slo.deadline - state.arrival
        if state.slo.request_type == RequestType.LATENCY:
            if state.ttft is not None and state.service_gain > 0:
                gain = (
                    state.service_gain
                    + remain * 8 * ratio(budget, served + cost) ** self.penalty
                )
            else:
                gain = (
                    state.prompt_len
                    * ratio(state.slo.ttft, served + self.unit_time) ** self.penalty
                )
                gain += (
                    state.predicted_len
                    * 8
                    * ratio(budget, served + cost) ** self.penalty
                )
        else:
            gain = (state.prompt_len + state.predicted_len * 8) * ratio(
                budget, served + cost
            ) ** self.penalty
        return gain, cost

    def select(
        self,
        states: list[RequestState],
        now: float,
        batch_size: int,
        top_k: int,
        strategy: str,
    ) -> tuple[list[str], dict[str, tuple[float, float]]]:
        scores = self.priorities(states, now)
        groups = GroupController(batch_size, top_k, 0, strategy)
        groups._recompute(states, [], scores)
        return groups.selected, scores


@dataclass
class SchedulePlan:
    activate: list[str]
    preempt: list[str]
    process: list[str]
    batch_budget: int
    admission_limit: int


class GroupController:
    """Stateful selection/activation from JITServe's executed SLO path.

    Allocation, token chunking and physical worker capacity belong to V1.
    No bin rotation or collective-demand expansion is executed in 84584a1.
    """

    def __init__(self, batch_size: int, top_k: int, max_swaps: int, strategy: str):
        self.batch_size = batch_size
        self.top_k = top_k
        self.max_swaps = max_swaps
        self.strategy = strategy
        self.admission_limit = batch_size
        self.selected: list[str] = []
        self.current: deque[str] = deque()
        self.pending: deque[str] = deque()
        self.bounds = (0, 0)
        self.is_last = False
        self.no_swap_space = False

    def enough_swap_space(self, free_fraction: float) -> bool:
        # Original block manager's cpu_watermarks is 0.2. Here capacity is bytes.
        if self.no_swap_space:
            if free_fraction > 0.8:
                self.no_swap_space = False
            return not self.no_swap_space
        if free_fraction < 0.2:
            self.no_swap_space = True
        return not self.no_swap_space

    def contains(self, length: int) -> bool:
        lower, upper = self.bounds
        return lower <= length and (self.is_last or length < upper)

    def _recompute(self, candidates, running, scores) -> None:
        batch = (
            self.batch_size
            if self.strategy == "sliding_window"
            else self.admission_limit
        )
        ranked = sorted(candidates, key=lambda s: scores[s.request_id][0])[
            : batch * self.top_k
        ]
        ranked.sort(key=lambda s: s.length)
        if not ranked:
            self.current.clear()
            self.pending.clear()
            self.selected = []
            return
        if self.strategy == "sliding_window":
            best = 0
            current_sum = best_sum = sum(
                scores[s.request_id][0] for s in ranked[:batch]
            )
            for i in range(batch, len(ranked)):
                current_sum += scores[ranked[i].request_id][0]
                current_sum -= scores[ranked[i - batch].request_id][0]
                if current_sum < best_sum:
                    best_sum, best = current_sum, i - batch + 1
            chosen = ranked[best : best + batch]
            self.is_last = True  # Source intentionally has an open upper bound.
        else:
            width = max(batch, len(ranked) // self.top_k)
            bins = [ranked[i : i + width] for i in range(0, len(ranked), width)]
            best = min(
                range(len(bins)),
                key=lambda i: sum(scores[s.request_id][0] for s in bins[i]),
            )
            chosen = list(bins[best])
            for i in sorted(range(len(bins)), key=lambda i: (abs(i - best), i)):
                if len(chosen) >= batch:
                    break
                if i != best:
                    chosen.extend(bins[i])
            self.is_last = max(s.length for s in chosen) == ranked[-1].length
        self.bounds = (min(s.length for s in chosen), max(s.length for s in chosen) + 1)
        self.selected = [s.request_id for s in chosen]
        self.current = deque(self.selected)
        seen = set(self.current)
        self.pending.clear()
        for state in running:
            if not self.contains(state.length):
                self.pending.append(state.request_id)
            elif state.request_id not in seen:
                self.current.append(state.request_id)
                seen.add(state.request_id)

    def plan(
        self,
        waiting: list[RequestState],
        swapped: list[RequestState],
        running: list[RequestState],
        scores: dict[str, tuple[float, float]],
        unit_time: float,
    ) -> SchedulePlan:
        if self.strategy == "sliding_window" or not self.pending:
            self._recompute(waiting + swapped + running, running, scores)
        seen = set(self.current)
        for state in waiting + swapped:
            if state.request_id not in seen and self.contains(state.length):
                self.current.appendleft(state.request_id)
                seen.add(state.request_id)
        active = {s.request_id for s in running}
        pending: list[str] = []
        while self.pending and len(pending) < self.max_swaps:
            key = self.pending.popleft()
            if key in active:
                pending.append(key)
        victims = pending if self.strategy == "length_bin" else []
        queued = {s.request_id for s in waiting + swapped}
        activate: list[str] = []
        while self.current and len(activate) < len(pending):
            key = self.current.popleft()
            if key in queued:
                activate.append(key)
        remaining = [s for s in running if s.request_id not in victims]
        process = activate + [s.request_id for s in remaining]
        budget = max(len(running), self.admission_limit)
        if self.strategy == "sliding_window":
            budget = max(self.batch_size, len(process))
            self.admission_limit = budget
            if unit_time > min((s.slo.tbt for s in remaining), default=0.001):
                self.admission_limit -= self.max_swaps
        return SchedulePlan(activate, victims, process, budget, self.admission_limit)
