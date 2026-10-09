# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JITServe's ordered SLO budget pass, independent of physical KV allocation.

Adapted from _schedule_process_list in JITServe 84584a1. GPU cost represents
the whole footprint of each candidate, against total usable pool capacity.
It is not incremental allocation cost against the currently free pool.
"""

from dataclasses import dataclass, field


@dataclass
class Candidate:
    request_id: str
    status: str  # waiting, swapped, running
    remaining: int
    prefill: bool
    blocks: int


@dataclass
class ExecutionPlan:
    reserved: dict[str, int] = field(default_factory=dict)
    preempt: list[str] = field(default_factory=list)
    commit_order: list[str] = field(default_factory=list)
    stopped_at: str | None = None
    remaining_tokens: int = 0


def reserve(
    candidates: list[Candidate], token_budget: int, batch_budget: int, block_budget: int
) -> ExecutionPlan:
    plan = ExecutionPlan(remaining_tokens=token_budget)
    for item in candidates:
        tokens = min(item.remaining, plan.remaining_tokens)
        if item.prefill:
            tokens = min(tokens, 2)
        no_update = item.remaining == 0
        if (
            plan.stopped_at is None
            and tokens > 0
            and len(plan.reserved) < batch_budget
            and item.blocks <= block_budget
        ):
            plan.reserved[item.request_id] = tokens
            plan.remaining_tokens -= tokens
            block_budget -= item.blocks
            continue
        if not no_update and plan.stopped_at is None:
            plan.stopped_at = item.request_id
        if item.status == "running" and tokens > 0:
            plan.preempt.append(item.request_id)
    plan.commit_order = [
        item.request_id
        for status in ("swapped", "running", "waiting")
        for item in candidates
        if item.status == status and item.request_id in plan.reserved
    ]
    return plan
