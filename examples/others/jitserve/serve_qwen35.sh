#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
model_path="${JITSERVE_MODEL:-/docker/data/HF_MODELS/Qwen3.5-27B}"
tp_size="${JITSERVE_TP:-4}"
backend="${JITSERVE_BACKEND:-jitserve-v1-port}"
case "$backend" in
    jitserve-v1-port|stock-v1) ;;
    *) echo "Unknown JITSERVE_BACKEND: $backend" >&2; exit 2 ;;
esac

"$repo_root/.venv/bin/python" "$script_dir/inspect_qwen35.py" \
    --model "$model_path" --tensor-parallel-size "$tp_size"

additional_config="${JITSERVE_ADDITIONAL_CONFIG:-}"
if [[ -z "$additional_config" ]]; then
    additional_config='{"jitserve":{"schedule_interval":50,"top_k_selection":1,"max_swaps_per_iter":3,"cpu_bytes_to_use":8589934592}}'
fi
command=(env VLLM_USE_V2_MODEL_RUNNER=0 "$repo_root/.venv/bin/vllm" serve "$model_path" \
    --served-model-name Qwen3.5-27B-jitserve \
    --host 127.0.0.1 --port "${JITSERVE_PORT:-8011}" \
    --tensor-parallel-size "$tp_size" --dtype bfloat16 \
    --gpu-memory-utilization "${JITSERVE_GPU_UTIL:-0.85}" \
    --max-model-len "${JITSERVE_MAX_MODEL_LEN:-8192}" \
    --max-num-seqs "${JITSERVE_MAX_NUM_SEQS:-16}" \
    --max-num-batched-tokens "${JITSERVE_BATCH_TOKENS:-512}" \
    --no-async-scheduling --enforce-eager \
    --enable-prefix-caching --mamba-cache-mode align \
    --no-disable-hybrid-kv-cache-manager \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --additional-config "$additional_config")
if [[ "$backend" == "jitserve-v1-port" ]]; then
    command+=(--scheduler-cls vllm.v1.jitserve.scheduler.JITServeScheduler
        --kv-transfer-config '{"kv_connector":"JITServeConnector","kv_connector_module_path":"vllm.v1.jitserve.connector","kv_role":"kv_both"}')
fi
command+=("$@")
if "$dry_run"; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPUs reserved for this run}"
exec "${command[@]}"
