# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real V1 scheduler and hybrid block allocator without a GPU.

The model output is a supplied token, as in upstream scheduler tests; these
tests validate allocation and progress accounting, not model inference.
"""

import time

import pytest
import torch
from transformers import Qwen2Config

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.jitserve.policy import SchedulePlan
from vllm.v1.jitserve.scheduler import JITServeScheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


@pytest.fixture(params=[16, 64], ids=["same_block_size", "mixed_block_sizes"])
def scheduler(tmp_path, request):
    if isinstance(request.param, tuple):
        mamba_block_size, capacity, num_blocks = request.param
    else:
        mamba_block_size, capacity, num_blocks = request.param, 1, 128
    Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, max_model_len=1024
        ),
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(
            block_size=mamba_block_size,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=capacity,
            max_num_batched_tokens=64,
            max_model_len=1024,
            async_scheduling=False,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="JITServeConnector",
            kv_connector_module_path="vllm.v1.jitserve.connector",
            kv_role="kv_both",
        ),
        additional_config={
            "jitserve": {
                "schedule_interval": 100,
                "search_strategy": "length_bin",
                "top_k_selection": 2,
                "max_swaps_per_iter": 1,
            }
        },
    )
    config.cache_config.num_gpu_blocks = num_blocks
    kv_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=4, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["gdn"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=((8,), (4, 4)),
                    dtypes=(torch.bfloat16, torch.float32),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    register_all_kvcache_specs(config)
    init_none_hash(sha256)
    return JITServeScheduler(
        vllm_config=config,
        kv_cache_config=kv_config,
        block_size=mamba_block_size,
        hash_block_size=16,
        structured_output_manager=StructuredOutputManager(config),
        log_stats=True,
    )


def request(name, prompt_len=33, prediction=10000):
    return Request(
        request_id=name,
        prompt_token_ids=[1] * prompt_len,
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
        sampling_params=SamplingParams(
            max_tokens=100,
            extra_args={
                "jitserve_request_type": 1,
                "jitserve_deadline": time.time() + 600,
                "jitserve_output_len": prediction,
                "jitserve_collection_id": name,
            },
        ),
    )


def finish_step(scheduler, output):
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={key: i for i, key in enumerate(ids)},
            sampled_token_ids=[
                [2]
                if scheduler.requests[key].num_computed_tokens
                >= scheduler.requests[key].num_prompt_tokens
                else []
                for key in ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def slo_step(scheduler):
    # Isolate allocation/restore assertions from cadence; cadence has its own test.
    scheduler.jit_step = scheduler.jit_policy.interval - 1
    return scheduler.schedule()


@pytest.mark.parametrize("prompt_len", [31, 32, 33, 47, 48, 49, 63, 64, 65])
def test_preempted_hybrid_request_resumes_at_exact_watermark(scheduler, prompt_len):
    long = request("long", prompt_len)
    scheduler.add_request(long)
    while not long.num_output_tokens:
        finish_step(scheduler, scheduler.schedule())
    watermark = long.num_computed_tokens
    assert watermark == prompt_len
    urgent = request("urgent", 10, 1)
    scheduler.add_request(urgent)
    output = slo_step(scheduler)
    assert "long" in output.kv_connector_metadata.stores
    assert "long" in output.preempted_req_ids
    snapshot = scheduler.jit_connector.snapshots["long"]
    assert snapshot.computed_tokens == watermark
    assert long.status == RequestStatus.PREEMPTED
    finish_step(scheduler, output)
    scheduler.finish_requests(["urgent"], RequestStatus.FINISHED_ABORTED)
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {"long": 1}
    assert resumed.has_sync_kv_loads
    target = resumed.kv_connector_metadata.loads["long"]
    # The latest GDN state must have an allocated destination even at a boundary.
    last_gdn_position = max(
        i for i, block in enumerate(snapshot.block_ids[1]) if block >= 0
    )
    assert target[1][last_gdn_position] >= 0
    assert long.num_computed_tokens == watermark + 1
    assert scheduler.jit_connector.used_bytes > 0  # copy not completed yet
    finish_step(scheduler, resumed)
    assert scheduler.jit_connector.used_bytes == 0


def test_abort_offloaded_request_releases_budget_after_worker_step(scheduler):
    long = request("long")
    scheduler.add_request(long)
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(request("urgent", 10, 1))
    finish_step(scheduler, slo_step(scheduler))
    assert scheduler.jit_connector.used_bytes > 0
    scheduler.finish_requests(["long", "urgent"], RequestStatus.FINISHED_ABORTED)
    cleanup = scheduler.schedule()
    assert "long" in cleanup.kv_connector_metadata.drops
    assert scheduler.jit_connector.used_bytes > 0
    finish_step(scheduler, cleanup)
    assert scheduler.jit_connector.used_bytes == 0
    assert not scheduler.jit_connector.reservations
    assert not scheduler.jit_states


def test_prediction_update_rejects_stale_and_finished_requests(scheduler):
    req = request("r")
    scheduler.add_request(req)
    assert scheduler.update_prediction("r", 42, 2)
    assert not scheduler.update_prediction("r", 1, 2)
    assert not scheduler.update_prediction("r", 1, 1)
    assert scheduler.jit_states["r"].predicted_len == 42
    scheduler.finish_requests(["r"], RequestStatus.FINISHED_ABORTED)
    assert not scheduler.update_prediction("r", 1, 3)


def test_proactive_swap_only_occurs_on_slo_round(scheduler):
    scheduler.jit_policy.interval = 4
    long = request("long")
    scheduler.add_request(long)
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(request("urgent", 10, 1))
    for _ in range(2):
        output = scheduler.schedule()
        assert not output.kv_connector_metadata.stores
        assert long.status == RequestStatus.RUNNING
        finish_step(scheduler, output)
    output = scheduler.schedule()
    assert set(output.kv_connector_metadata.stores) == {"long"}
    assert set(output.num_scheduled_tokens) == {"urgent"}


def test_sliding_window_does_not_swap_longer_running_request(scheduler):
    scheduler.groups.strategy = "sliding_window"
    long = request("long")
    scheduler.add_request(long)
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(request("urgent", 10, 1))
    output = slo_step(scheduler)
    assert scheduler.selected == ["urgent"]
    assert set(output.num_scheduled_tokens) == {"long"}
    assert not output.kv_connector_metadata.stores


def test_sliding_expansion_obeys_worker_capacity_and_reports_it(scheduler):
    scheduler.groups.strategy = "sliding_window"
    short = request("short", 10)
    scheduler.add_request(short)
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(request("urgent", 33, 1))
    output = slo_step(scheduler)
    assert set(output.num_scheduled_tokens) == {"urgent"}
    assert set(output.kv_connector_metadata.stores) == {"short"}
    stats = scheduler.jitserve_stats()
    assert stats["logical_admission_limit"] == 1
    assert stats["capacity_limited_rounds"] == 1
    assert stats["worker_capacity"] == 1


def test_interval_one_preserves_source_cold_start_behavior(scheduler):
    scheduler.jit_policy.interval = 1
    scheduler.add_request(request("r"))
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert scheduler.jitserve_stats()["live_requests"] == 1


@pytest.mark.parametrize("scheduler", [(16, 3, 128), (64, 3, 128)], indirect=True)
def test_slo_reserves_activation_before_running_prefill_consumes_budget(scheduler):
    scheduler.max_num_scheduled_tokens = 129
    scheduler.scheduler_config.max_num_batched_tokens = 129
    decode, prefill = request("decode", 10), request("prefill", 192)
    scheduler.add_request(decode)
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(prefill)
    finish_step(scheduler, scheduler.schedule())
    previous = prefill.num_computed_tokens
    assert 0 < previous < prefill.num_prompt_tokens
    scheduler.add_request(request("activate", 33, 1))
    output = scheduler._execute_slo(
        SchedulePlan(["activate"], [], ["activate", "prefill", "decode"], 3, 3)
    )
    assert scheduler.last_execution.reserved == {
        "activate": 2,
        "prefill": 2,
        "decode": 1,
    }
    assert output.num_scheduled_tokens["prefill"] == 2
    assert output.num_scheduled_tokens["activate"] > 0
    assert prefill.num_computed_tokens == previous + 2
    assert list(output.num_scheduled_tokens) == ["prefill", "decode", "activate"]
    assert [r.request_id for r in scheduler.running] == [
        "decode",
        "prefill",
        "activate",
    ]
    finish_step(scheduler, output)


@pytest.mark.parametrize("scheduler", [(16, 3, 8), (64, 3, 8)], indirect=True)
def test_failed_first_candidate_stops_backfill_and_swaps_running_in_order(scheduler):
    for key in ["a", "b"]:
        scheduler.add_request(request(key, 2))
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(request("large", 256, 1))
    scheduler.add_request(request("small", 2, 1))
    output = scheduler._execute_slo(
        SchedulePlan(["large", "small"], [], ["large", "small", "a", "b"], 4, 4)
    )
    assert scheduler.last_execution.stopped_at == "large"
    assert scheduler.last_execution.preempt == ["a", "b"]
    assert list(output.kv_connector_metadata.stores) == ["a", "b"]
    assert not output.num_scheduled_tokens
    assert set(scheduler.requests) == {"large", "small", "a", "b"}
    assert not scheduler.running
    finish_step(scheduler, output)


@pytest.mark.parametrize("scheduler", [(16, 3, 128)], indirect=True)
def test_exhausted_token_budget_keeps_unscheduled_running_requests(scheduler):
    for key in ["a", "b"]:
        scheduler.add_request(request(key, 2))
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(request("new", 10))
    scheduler.max_num_scheduled_tokens = 1
    output = scheduler._execute_slo(SchedulePlan(["new"], [], ["new", "a", "b"], 3, 3))
    assert output.num_scheduled_tokens == {"new": 1}
    assert {r.request_id for r in scheduler.running} == {"new", "a", "b"}
    assert not output.kv_connector_metadata.stores
    finish_step(scheduler, output)


@pytest.mark.parametrize("scheduler", [(16, 2, 128), (64, 2, 128)], indirect=True)
def test_slo_restores_snapshot_before_running_commit(scheduler):
    for key in ["restore", "running"]:
        scheduler.add_request(request(key, 10))
    finish_step(scheduler, scheduler.schedule())
    request_to_restore = scheduler.requests["restore"]
    watermark = request_to_restore.num_computed_tokens
    scheduler._slo_preempt(request_to_restore)
    output = scheduler._execute_slo(
        SchedulePlan(["restore"], [], ["restore", "running"], 2, 2)
    )
    assert list(output.num_scheduled_tokens) == ["restore", "running"]
    assert output.num_scheduled_tokens["restore"] == 1
    assert request_to_restore.num_computed_tokens == watermark + 1
    assert output.has_sync_kv_loads
    assert "restore" in output.kv_connector_metadata.loads
    finish_step(scheduler, output)
    assert scheduler.jit_connector.used_bytes == 0


def test_insufficient_cpu_space_is_an_explicit_error(scheduler):
    long = request("long")
    scheduler.add_request(long)
    finish_step(scheduler, scheduler.schedule())
    scheduler.jit_connector.capacity = 1
    scheduler.add_request(request("urgent", 10, 1))
    with pytest.raises(MemoryError, match="CPU swap capacity exhausted"):
        slo_step(scheduler)


def test_restoring_snapshot_keeps_cpu_budget_until_copies_finish(scheduler):
    req = request("r")
    scheduler.add_request(req)
    finish_step(scheduler, scheduler.schedule())
    blocks = scheduler.kv_cache_manager.get_blocks("r")
    connector = scheduler.jit_connector
    connector.save_request(req, blocks)
    connector.capacity = connector.used_bytes
    ids = connector.snapshots["r"].block_ids
    connector.update_state_after_alloc(req, blocks, req.num_computed_tokens)
    assert not connector.can_store(ids)
    connector.complete_step(connector.build_connector_meta(None))
    assert connector.can_store(ids)


@pytest.mark.parametrize(
    "scheduler_cls",
    [
        "vllm.v1.jitserve.scheduler.JITServeScheduler",
        JITServeScheduler,
    ],
)
def test_bad_slo_is_rejected_before_engine_submission(scheduler_cls):
    from types import SimpleNamespace

    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.input_processor import InputProcessor

    processor = InputProcessor.__new__(InputProcessor)
    processor.scheduler_config = SimpleNamespace(scheduler_cls=scheduler_cls)
    with pytest.raises(VLLMValidationError, match="Missing JITServe parameters"):
        processor.process_inputs("r", "prompt", SamplingParams(), ("generate",))


def test_swap_metadata_survives_scheduler_worker_serialization(scheduler):
    import pickle

    from vllm.v1.jitserve.connector import SwapMetadata

    req = request("r")
    scheduler.add_request(req)
    finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(request("urgent", 10, 1))
    output = slo_step(scheduler)
    # MultiprocExecutor's shared-memory MessageQueue uses this protocol.
    result = pickle.loads(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL))
    assert isinstance(result.kv_connector_metadata, SwapMetadata)
    assert result.kv_connector_metadata.stores == output.kv_connector_metadata.stores
