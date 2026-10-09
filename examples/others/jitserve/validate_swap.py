# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare stock V1 and JITServe tokens with real mid-generation preemption.

Run the baseline and JITServe modes in separate processes using the same model,
GPU, and execution flags. Output contains tokens and counters, not prompt text.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

from vllm import EngineArgs, LLMEngine, SamplingParams


def worker_swap_stats(worker):
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    return get_kv_transfer_group().worker_stats()


def create_engine(
    args,
    *,
    max_model_len=1024,
    schedule_interval=2,
    max_num_seqs=1,
    jitserve_options=None,
):
    kwargs = {}
    if args.baseline is not None:
        kwargs.update(
            scheduler_cls="vllm.v1.jitserve.scheduler.JITServeScheduler",
            kv_transfer_config={
                "kv_connector": "JITServeConnector",
                "kv_connector_module_path": "vllm.v1.jitserve.connector",
                "kv_role": "kv_both",
            },
            additional_config={
                "jitserve": {
                    "schedule_interval": schedule_interval,
                    "search_strategy": "length_bin",
                    "top_k_selection": 2,
                    "max_swaps_per_iter": 1,
                    **(jitserve_options or {}),
                }
            },
        )
    return LLMEngine.from_engine_args(
        EngineArgs(
            model=args.model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=128 + max_num_seqs - 1,
            enforce_eager=True,
            async_scheduling=False,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            disable_hybrid_kv_cache_manager=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            tensor_parallel_size=args.tensor_parallel_size,
            seed=42,
            gpu_memory_utilization=args.gpu_memory_utilization,
            **kwargs,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    args = parser.parse_args()
    jit = args.baseline is not None
    engine = create_engine(args)
    tokenizer = engine.get_tokenizer()
    texts = {
        "long": "List the integers from 1 to 100, separated by commas.",
        "urgent": "Reply with only the word OK.",
    }

    def add(name, predicted_len):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": texts[name]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        params = SamplingParams(
            temperature=0,
            max_tokens=96,
            extra_args={
                "jitserve_request_type": 1,
                "jitserve_deadline": time.time() + 600,
                "jitserve_output_len": predicted_len,
                "jitserve_collection_id": name,
            },
        )
        engine.add_request(name, prompt, params)

    add("long", 10000)
    outputs = {}
    step = 0
    while engine.has_unfinished_requests():
        if step == 8:
            add("urgent", 4)
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = {
                    "tokens": list(output.outputs[0].token_ids),
                    "finish_reason": output.outputs[0].finish_reason,
                }
        step += 1
    assert set(outputs) == {"long", "urgent"}, "Both requests must run"
    result = {"mode": "jitserve" if jit else "stock", "outputs": outputs}
    if jit:
        core = engine.engine_core.engine_core
        result["scheduler"] = core.jitserve_stats()
        result["workers"] = engine.collective_rpc(worker_swap_stats)
        assert result["scheduler"]["proactive_preemptions"] > 0
        assert result["scheduler"]["swap_out_submissions"] > 0
        assert result["scheduler"]["swap_in_submissions"] > 0
        assert result["scheduler"]["cpu_reserved_bytes"] == 0
        assert result["scheduler"]["live_requests"] == 0
        assert len(result["workers"]) == args.tensor_parallel_size
        for stats in result["workers"]:
            assert stats["completed_stores"] > 0
            assert stats["completed_loads"] > 0
            assert stats["cpu_snapshots"] == 0
        baseline = json.loads(args.baseline.read_text())
        assert baseline["model"] == str(Path(args.model).resolve())
        assert baseline["tensor_parallel_size"] == args.tensor_parallel_size
        assert outputs == baseline["outputs"], "Swap changed output tokens or stopping"
        result["tokens_match_stock"] = True
    result["model"] = str(Path(args.model).resolve())
    result["tensor_parallel_size"] = args.tensor_parallel_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}))


if __name__ == "__main__":
    main()
