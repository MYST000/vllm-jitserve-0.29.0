# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy contracts and real byte-copy round trips, without model execution."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.jitserve.connector import JITServeConnector, Snapshot, SwapMetadata
from vllm.v1.jitserve.policy import SLO, Policy, RequestState, RequestType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)


def test_collective_members_share_aggregate_reward():
    states = [
        RequestState(str(i), SLO(RequestType.COLLECTIVE, 100, 10, "job"), 0, 20)
        for i in range(2)
    ]
    policy = Policy()
    scores = policy.priorities(states, 1)
    assert scores["0"] == scores["1"] == (-500.0, 0.4)


def test_length_window_uses_reward_density_before_length_grouping():
    states = [
        RequestState(str(i), SLO(RequestType.THROUGHPUT, 100, n, str(i)), 0, p)
        for i, (p, n) in enumerate([(100, 1), (200, 2), (10, 100), (300, 3)])
    ]
    selected, scores = Policy().select(states, 1, 2, 2, "sliding_window")
    assert selected == ["0", "1"]
    assert scores["0"][0] < scores["2"][0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_len", 0),
        ("output_len", 2.5),
        ("output_len", True),
        ("request_type", 1.5),
        ("deadline", float("nan")),
        ("collection_id", ""),
        ("collection_id", None),
    ],
)
def test_invalid_slo_is_rejected(field, value):
    extra = dict(
        jitserve_request_type=1,
        jitserve_output_len=10,
        jitserve_deadline=100,
        jitserve_collection_id="job",
    )
    extra["jitserve_" + field] = value
    with pytest.raises(ValueError):
        SLO.parse(extra)


def test_ingress_trace_preserves_internal_identity_without_payloads():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.jitserve.trace import ingress_record

    request = EngineCoreRequest(
        request_id="chatcmpl-attempt-randomized",
        external_req_id="chatcmpl-attempt",
        prompt_token_ids=[12, 34],
        mm_features=None,
        sampling_params=SamplingParams(
            extra_args={"jitserve_output_len": 123, "private": "never-log"}
        ),
        pooling_params=None,
        arrival_time=1,
        lora_request=None,
        cache_salt="private-salt",
        data_parallel_rank=None,
        trace_headers={"authorization": "never-log"},
    )
    assert ingress_record(request) == {
        "engine_request_id": "chatcmpl-attempt-randomized",
        "external_request_id": "chatcmpl-attempt",
        "slo_parameters": {"jitserve_output_len": 123},
    }


def make_connector(capacity=4096):
    attention = FullAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=2, dtype=torch.float16
    )
    recurrent = MambaSpec(block_size=2, shapes=((8,),), dtypes=(torch.float32,))
    # Both groups alias one allocation; layers are interleaved within blocks.
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(
                size=256, layers=["a", "b"], layer_stride=16, block_stride=32
            ),
            KVCacheTensor(size=256, layers=["m"], layer_stride=32, block_stride=32),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["a", "b"], attention),
            KVCacheGroupSpec(["m"], recurrent),
        ],
    )
    options = SimpleNamespace(
        additional_config={"jitserve": {"cpu_bytes_to_use": capacity}},
        parallel_config=SimpleNamespace(world_size=2),
        kv_transfer_config=SimpleNamespace(is_kv_producer=False),
    )
    return JITServeConnector(options, KVConnectorRole.WORKER, config)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_snapshot_survives_page_reuse_across_attention_and_recurrent_groups(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU result is not GPU validation")
    connector = make_connector()
    backing = torch.arange(256, dtype=torch.uint8, device=device)
    connector.register_kv_caches({name: backing for name in ("a", "b", "m")})
    expected = {name: pages.clone() for name, pages in connector.pages.items()}
    ids = ([1, 2], [-1, 3])
    assert connector.snapshot_bytes(ids) == 192  # 96 real bytes per TP rank
    connector.handle_preemptions(SwapMetadata(stores={"r": Snapshot(3, ids, 192)}))
    backing.zero_()  # the freed physical pages have been overwritten
    metadata = SwapMetadata(loads={"r": ([4, 5], [-1, 6])})
    connector.load_kv_before_preprocess(metadata)
    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(None)
    for name in ("a", "b"):
        assert torch.equal(connector.pages[name][4:6], expected[name][1:3])
    assert torch.equal(connector.pages["m"][6], expected["m"][3])
    assert not connector.cpu


def test_swap_budget_accounts_for_real_pages_and_tp_ranks():
    connector = make_connector(capacity=191)
    assert not connector.can_store(([1, 2], [-1, 3]))
    assert connector.can_store(([1], [-1, 3]))


def test_cancel_discards_cpu_snapshot():
    connector = make_connector()
    backing = torch.arange(256, dtype=torch.uint8)
    connector.register_kv_caches({name: backing for name in ("a", "b", "m")})
    connector.handle_preemptions(
        SwapMetadata(stores={"r": Snapshot(3, ([1, 2], [-1, 3]), 192)})
    )
    connector.handle_preemptions(SwapMetadata(drops={"r"}))
    assert not connector.cpu


def test_missing_recurrent_resume_page_is_an_error():
    connector = make_connector()
    backing = torch.arange(256, dtype=torch.uint8)
    connector.register_kv_caches({name: backing for name in ("a", "b", "m")})
    connector.handle_preemptions(
        SwapMetadata(stores={"r": Snapshot(3, ([1, 2], [-1, 3]), 192)})
    )
    with pytest.raises(RuntimeError, match="lost required page"):
        connector.load_kv_before_preprocess(
            SwapMetadata(loads={"r": ([4, 5], [-1, -1])})
        )
    assert connector.completed_loads == 0


def test_restore_must_run_before_recurrent_preprocessing():
    connector = make_connector()
    connector.bind_connector_metadata(SwapMetadata())
    with pytest.raises(RuntimeError, match="pre-preprocessing restore hook"):
        connector.start_load_kv(None)
