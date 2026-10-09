# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata-only ingress evidence for stock/JITServe comparisons."""

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

SLO_FIELDS = (
    "jitserve_request_type",
    "jitserve_deadline",
    "jitserve_output_len",
    "jitserve_collection_id",
    "jitserve_ttft",
    "jitserve_tbt",
    "jitserve_client_request_id",
)


def ingress_record(request: EngineCoreRequest) -> dict:
    params = request.sampling_params
    extra = (params.extra_args or {}) if isinstance(params, SamplingParams) else {}
    return {
        "engine_request_id": request.request_id,
        "external_request_id": request.external_req_id,
        "slo_parameters": {name: extra[name] for name in SLO_FIELDS if name in extra},
    }
