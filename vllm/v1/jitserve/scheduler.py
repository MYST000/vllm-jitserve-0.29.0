# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JITServe scheduling on the vLLM 0.29.0 synchronous V1 engine."""

import math
import time
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.jitserve.connector import JITServeConnector, SwapMetadata
from vllm.v1.jitserve.execution import Candidate, reserve
from vllm.v1.jitserve.policy import SLO, GroupController, Policy, RequestState
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class AllocationContext:
    blocks: KVCacheBlocks
    local: int
    external: int
    computed: int
    remaining: int


class JITServeScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config
        # JITServe supplies its own dynamic ordering, including the victim order.
        self.policy = SchedulingPolicy.FCFS
        if self.use_v2_model_runner:
            raise ValueError("JITServe restore requires VLLM_USE_V2_MODEL_RUNNER=0")
        if config.scheduler_config.async_scheduling:
            raise ValueError(
                "JITServe CPU swap currently requires --no-async-scheduling"
            )
        if config.speculative_config is not None:
            raise ValueError(
                "JITServe CPU swap does not yet implement speculative state"
            )
        parallel = config.parallel_config
        if (
            parallel.pipeline_parallel_size,
            parallel.data_parallel_size,
            parallel.decode_context_parallel_size,
            parallel.prefill_context_parallel_size,
        ) != (1, 1, 1, 1):
            raise ValueError(
                "JITServe migration supports TP; PP/DP/DCP/PCP require validation"
            )
        if not isinstance(self.connector, JITServeConnector):
            raise ValueError(
                "JITServeScheduler requires JITServeConnector for CPU swap"
            )
        self.jit_connector = self.connector
        additional = config.additional_config
        if not isinstance(additional, dict):
            raise ValueError("JITServe additional_config must be a dictionary")
        options = additional.get("jitserve", {})
        self.interval = int(options.get("schedule_interval", 50))
        self.top_k = int(options.get("top_k_selection", 1))
        self.max_swaps = int(options.get("max_swaps_per_iter", 3))
        penalty = float(options.get("penalty_factor", 1.0))
        if (
            min(self.interval, self.top_k) <= 0
            or self.max_swaps < 0
            or not math.isfinite(penalty)
            or penalty <= 0
        ):
            raise ValueError("JITServe parameters must be positive; max_swaps may be 0")
        self.strategy = options.get("search_strategy", "sliding_window")
        if self.strategy not in {"sliding_window", "length_bin"}:
            raise ValueError("Unknown JITServe search strategy")
        self.jit_policy = Policy(penalty, self.interval)
        self.worker_capacity = self.max_num_running_reqs
        batch_size = int(options.get("max_batch_size", self.worker_capacity))
        if not 0 < batch_size <= self.worker_capacity:
            raise ValueError("jitserve.max_batch_size must be in [1, --max-num-seqs]")
        self.groups = GroupController(
            batch_size, self.top_k, self.max_swaps, self.strategy
        )
        self.jit_states: dict[str, RequestState] = {}
        self.jit_step = 0
        self.selected: list[str] = []
        self.priorities: dict[str, tuple[float, float]] = {}
        self.proactive_preemptions = 0
        self.slo_rounds = 0
        self.capacity_limited_rounds = 0
        self.allocation_stopped_rounds = 0
        self.last_execution = None

    def jitserve_stats(self) -> dict[str, int]:
        return {
            "proactive_preemptions": self.proactive_preemptions,
            "swap_out_submissions": self.jit_connector.num_stores,
            "swap_in_submissions": self.jit_connector.num_loads,
            "swap_out_submitted_bytes": self.jit_connector.stored_bytes,
            "swap_in_submitted_bytes": self.jit_connector.loaded_bytes,
            "cpu_reserved_bytes": self.jit_connector.used_bytes,
            "cpu_snapshots": len(self.jit_connector.snapshots),
            "live_requests": len(self.jit_states),
            "slo_rounds": self.slo_rounds,
            "logical_admission_limit": self.groups.admission_limit,
            "worker_capacity": self.worker_capacity,
            "capacity_limited_rounds": self.capacity_limited_rounds,
            "allocation_stopped_rounds": self.allocation_stopped_rounds,
        }

    def add_request(self, request):
        if request.sampling_params is None:
            raise ValueError("JITServe requires a generation request")
        if request.has_encoder_inputs:
            raise ValueError(
                "JITServe SLO execution currently supports text-only requests"
            )
        slo = SLO.parse(request.sampling_params.extra_args)
        self.jit_states[request.request_id] = RequestState(
            request.request_id, slo, request.arrival_time, request.num_prompt_tokens
        )
        super().add_request(request)

    def update_prediction(self, request_id: str, output_len: int, version: int) -> bool:
        if type(output_len) is not int or output_len <= 0:
            raise ValueError("output_len must be a positive integer")
        if type(version) is not int or version <= 0:
            raise ValueError("version must be a positive integer")
        state = self.jit_states.get(request_id)
        if state is None or version <= state.prediction_version:
            return False
        state.predicted_len, state.prediction_version = output_len, version
        return True

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        if request.num_computed_tokens:
            self.jit_connector.save_request(
                request, self.kv_cache_manager.get_blocks(request.request_id)
            )
        super()._preempt_request(request, timestamp, drop_stale_output)

    def _get_local_prefix_cache_hit(self, request):
        if request.request_id in self.jit_connector.snapshots:
            return self.kv_cache_manager.empty_kv_cache_blocks, 0, 0, False
        return super()._get_local_prefix_cache_hit(request)

    def _allocation_context(self, request):
        blocks = self.kv_cache_manager.empty_kv_cache_blocks
        local = external = 0
        if request.num_computed_tokens == 0:
            blocks, local, request.shared_prefix_boundary, _ = (
                self._get_local_prefix_cache_hit(request)
            )
            external, asynchronous = self.jit_connector.get_num_new_matched_tokens(
                request, local
            )
            assert external is not None and not asynchronous
        computed = request.num_computed_tokens + local + external
        return AllocationContext(
            blocks, local, external, computed, request.num_tokens - computed
        )

    def _candidate_footprint(self, request, context):
        manager = self.kv_cache_manager
        required = manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=request.num_tokens,
            new_computed_blocks=context.blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=context.computed,
            num_local_computed_tokens=request.num_computed_tokens + context.local,
            num_tokens_main_model=request.num_tokens,
            apply_admission_cap=True,
        )
        resident = {
            b.block_id
            for group in manager.get_blocks(request.request_id).blocks
            for b in group
            if not b.is_null
        }
        # The coordinator already includes unpinned prefix hits in its free-pool
        # demand. Add pinned hits and this request's current physical footprint.
        resident.update(
            b.block_id
            for group in context.blocks.blocks
            for b in group
            if not b.is_null and b.ref_cnt > 0
        )
        return required + len(resident)

    def _slo_preempt(self, request):
        self.running.remove(request)
        self._preempt_request(request, time.monotonic())
        self.proactive_preemptions += 1

    def _execute_slo(self, group_plan):
        self.current_step += 1
        manager = self.kv_cache_manager
        manager.new_step_starts()
        for key in group_plan.preempt:
            self._slo_preempt(self.requests[key])

        candidates = []
        for key in group_plan.process:
            request = self.requests[key]
            if self._is_blocked_waiting_status(
                request.status
            ) and not self._try_promote_blocked_waiting_request(request):
                continue
            status = (
                "running"
                if request.status == RequestStatus.RUNNING
                else "swapped"
                if key in self.jit_connector.snapshots
                else "waiting"
            )
            manager.remove_skipped_blocks(
                key, request.num_computed_tokens, request.num_prompt_tokens
            )
            context = self._allocation_context(request)
            candidates.append(
                Candidate(
                    key,
                    status,
                    context.remaining,
                    context.computed < request.num_prompt_tokens,
                    self._candidate_footprint(request, context),
                )
            )

        execution = reserve(
            candidates,
            min(
                self.max_num_scheduled_tokens,
                self.scheduler_config.max_num_batched_tokens,
            ),
            min(group_plan.batch_budget, self.worker_capacity),
            manager.block_pool.num_gpu_blocks - 1 - manager.watermark_blocks,
        )
        self.last_execution = execution
        self.allocation_stopped_rounds += execution.stopped_at is not None
        for key in execution.preempt:
            self._slo_preempt(self.requests[key])

        # V0 can drop a running request from all queues when its token budget
        # reaches zero. Preserve that work. If admissions need its fixed V1
        # slot, save its state instead of silently losing the request.
        admissions = sum(
            self.requests[key].status != RequestStatus.RUNNING
            for key in execution.reserved
        )
        for request in reversed(list(self.running)):
            if len(self.running) + admissions <= self.worker_capacity:
                break
            if request.request_id not in execution.reserved:
                self._slo_preempt(request)

        by_id = {c.request_id: c for c in candidates}
        scheduled: dict[str, list[Request]] = {
            status: [] for status in ("waiting", "swapped", "running")
        }
        blocks_by_id = {}
        tokens_by_id = {}
        sync_loads = False
        scheduled_loras: set[int] = set()
        for key in execution.commit_order:
            request = self.requests[key]
            candidate = by_id[key]
            # Earlier commits can evict an unpinned prefix hit. Resolve it at
            # acquisition time; never retain stale physical block references.
            context = self._allocation_context(request)
            tokens = min(execution.reserved[key], context.remaining)
            if candidate.status == "waiting":
                extra = min(context.remaining - tokens, execution.remaining_tokens)
                tokens += extra
                execution.remaining_tokens -= extra
            threshold = self.scheduler_config.long_prefill_token_threshold
            if 0 < threshold < tokens:
                tokens = threshold
            if self.need_mamba_block_aligned_split:
                tokens = self._mamba_block_aligned_split(
                    request,
                    tokens,
                    context.local,
                    context.external,
                    allow_sub_block_chunk=True,
                )
            if tokens <= 0:
                continue
            lora_id = request.lora_request.lora_int_id if request.lora_request else 0
            if (
                self.lora_config
                and lora_id > 0
                and (
                    lora_id not in scheduled_loras
                    and len(scheduled_loras) == self.lora_config.max_loras
                )
            ):
                continue
            new_blocks = manager.allocate_slots(
                request,
                tokens,
                num_new_computed_tokens=context.local,
                new_computed_blocks=context.blocks
                if candidate.status != "running"
                else None,
                num_external_computed_tokens=context.external,
                full_sequence_must_fit=(candidate.status == "waiting"),
                has_scheduled_reqs=True,
            )
            if new_blocks is None:
                # Match can_append/can_allocate/can_swap_in: keep the request
                # queued without evicting a different already accepted request.
                continue
            if candidate.status != "running":
                self.jit_connector.update_state_after_alloc(
                    request, manager.get_blocks(key), context.external
                )
                if request in self.waiting:
                    self.waiting.remove_request(request)
                else:
                    self.skipped_waiting.remove_request(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, time.monotonic()
                    )
                if request.prefill_stats and request.num_preemptions <= 0:
                    request.prefill_stats.set(
                        num_prompt_tokens=request.num_prompt_tokens,
                        num_local_cached_tokens=context.local,
                        num_external_cached_tokens=context.external,
                    )
                manager.record_prefix_cache_stats(request, context.local)
                if self.connector_prefix_cache_stats is not None:
                    self.connector_prefix_cache_stats.record(
                        num_tokens=request.num_tokens - context.local,
                        num_hits=context.external,
                        preempted=request.num_preemptions > 0,
                    )
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = context.computed
                self.running.append(request)
                new_blocks = manager.get_blocks(key)
                if context.computed + tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
            if lora_id > 0:
                scheduled_loras.add(lora_id)
            scheduled[candidate.status].append(request)
            tokens_by_id[key] = tokens
            blocks_by_id[key] = new_blocks
            sync_loads |= context.external > 0

        ordered = [
            request
            for status in ("swapped", "running", "waiting")
            for prefill in (False, True)
            for request in scheduled[status]
            if by_id[request.request_id].prefill == prefill
        ]
        self.running = ordered + [
            r for r in self.running if r.request_id not in tokens_by_id
        ]
        return self._make_scheduler_output(
            scheduled_new_reqs=scheduled["waiting"],
            scheduled_resumed_reqs=scheduled["swapped"],
            scheduled_running_reqs=scheduled["running"],
            req_to_new_blocks=blocks_by_id,
            num_scheduled_tokens=tokens_by_id,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            has_sync_kv_loads=sync_loads,
        )

    def schedule(self, throttle_prefills=False):
        if self.pause_state != PauseState.UNPAUSED:
            return super().schedule(throttle_prefills)
        now = time.time()
        self.jit_step += 1
        live = [state for key, state in self.jit_states.items() if key in self.requests]
        for state in live:
            state.output_len = self.requests[state.request_id].num_output_tokens
        slo_round = self.jit_policy.tick(self.jit_step, now)
        slo_round = slo_round and self.groups.enough_swap_space(
            (self.jit_connector.capacity - self.jit_connector.used_bytes)
            / self.jit_connector.capacity
        )
        queued = list(self.waiting) + list(self.skipped_waiting)
        queued_ids = {r.request_id for r in queued}
        # The V0 swap queue is FIFO, separate from priority-sorted fresh work.
        swapped = [
            self.requests[key]
            for key in self.jit_connector.snapshots
            if key in queued_ids
        ]
        swapped_ids = {r.request_id for r in swapped}
        waiting = [r for r in queued if r.request_id not in swapped_ids]
        query = waiting + swapped + self.running if slo_round else waiting
        self.priorities = self.jit_policy.priorities(
            live, now, [self.jit_states[r.request_id] for r in query]
        )
        admission = self.groups.admission_limit
        if slo_round:
            self.slo_rounds += 1
            plan = self.groups.plan(
                [self.jit_states[r.request_id] for r in waiting],
                [self.jit_states[r.request_id] for r in swapped],
                [self.jit_states[r.request_id] for r in self.running],
                self.priorities,
                self.jit_policy.unit_time,
            )
            self.selected = list(self.groups.selected)
            admission = plan.batch_budget
            if admission > self.worker_capacity:
                # V1 has fixed worker arrays; expose saturation in experiment stats.
                if self.capacity_limited_rounds == 0:
                    logger.info(
                        "JITServe logical batch %d exceeds V1 worker capacity %d; "
                        "recording capacity_limited_rounds. Increase --max-num-seqs "
                        "relative to jitserve.max_batch_size for expansion headroom.",
                        admission,
                        self.worker_capacity,
                    )
                self.capacity_limited_rounds += 1
            output = self._execute_slo(plan)
        else:
            ordered = swapped + sorted(
                waiting, key=lambda r: self.priorities[r.request_id]
            )
            self.waiting = create_request_queue(self.policy)
            self.skipped_waiting = create_request_queue(self.policy)
            for request in ordered:
                self.waiting.add_request(request)
            self.max_num_running_reqs = max(
                len(self.running), min(admission, self.worker_capacity)
            )
            try:
                output = super().schedule(throttle_prefills)
            finally:
                self.max_num_running_reqs = self.worker_capacity
        # Source observes only the post-schedule running queue. In particular,
        # prefilling without a new token resets the per-observation service gain.
        observed_at = time.time()
        for request in self.running:
            self.jit_states[request.request_id].observe(
                request.num_output_tokens, observed_at, self.jit_policy.penalty
            )
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        output = super().update_from_output(scheduler_output, model_runner_output)
        metadata = scheduler_output.kv_connector_metadata
        assert isinstance(metadata, SwapMetadata)
        self.jit_connector.complete_step(metadata)
        return output

    def _free_request(self, request, delay_free_blocks=False):
        self.jit_states.pop(request.request_id, None)
        self.jit_policy.cache.pop(request.request_id, None)
        return super()._free_request(request, delay_free_blocks)
