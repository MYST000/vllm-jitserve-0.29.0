# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real model checks with natural SLO cadence for both aligned strategies."""

import argparse
import json
import time
from pathlib import Path

from validate_swap import create_engine, worker_swap_stats

from vllm import SamplingParams
from vllm.v1.jitserve.policy import GroupController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    args = parser.parse_args()
    jit = args.baseline is not None
    engine = create_engine(
        args,
        max_model_len=4096,
        max_num_seqs=3,
        schedule_interval=4,
        jitserve_options={"max_batch_size": 1},
    )
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    tokenizer = engine.get_tokenizer()
    baseline = json.loads(args.baseline.read_text()) if jit else None
    result = {
        "model": str(Path(args.model).resolve()),
        "tp": args.tensor_parallel_size,
        "mode": "jitserve" if jit else "stock",
        "cases": {},
    }
    if baseline is not None:
        assert baseline["model"] == result["model"]
        assert baseline["tp"] == result["tp"]

    def add(name, length, prediction, short=False):
        content = (
            "Reply with only the word OK."
            if short
            else "List the integers from 1 to 100, separated by commas."
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
        tokens = [tokenizer.encode("Context", add_special_tokens=False)[0]] * (
            length - len(tokens)
        ) + tokens
        assert len(tokens) == length
        engine.add_request(
            name,
            {"prompt_token_ids": tokens},
            SamplingParams(
                temperature=0,
                max_tokens=96,
                extra_args={
                    "jitserve_request_type": 1,
                    "jitserve_deadline": time.time() + 600,
                    "jitserve_output_len": prediction,
                    "jitserve_collection_id": name,
                    "jitserve_tbt": 10.0,
                },
            ),
        )

    for name in [
        "sliding_open_upper",
        "length_bin_swap",
        "sliding_expand",
        "cancel_offloaded",
    ]:
        assert engine.reset_prefix_cache()
        if jit:
            strategy = "sliding_window" if name.startswith("sliding") else "length_bin"
            scheduler.groups = GroupController(
                2 if name == "sliding_expand" else 1, 2, 1, strategy
            )
            scheduler.jit_policy.cache.clear()
        before = core.jitserve_stats() if jit else {}
        workers_before = engine.collective_rpc(worker_swap_stats) if jit else []
        outputs, events, batches = {}, [], []
        cancelled = False
        expanding = name == "sliding_expand"
        add(f"{name}-initial", 48 if expanding else 128, 10000)
        # Observe immutable metadata emitted by the real scheduler. No counter,
        # queue, token or time is edited to force a scheduling decision.
        original_schedule = scheduler.schedule

        def observe(
            *a,
            original_schedule=original_schedule,
            events=events,
            batches=batches,
            **kw,
        ):
            running_prefills = {
                r.request_id
                for r in scheduler.running
                if r.num_computed_tokens < r.num_prompt_tokens
            }
            previous_execution = scheduler.last_execution if jit else None
            output = original_schedule(*a, **kw)
            if jit:
                for key, snapshot in output.kv_connector_metadata.stores.items():
                    events.append(
                        {
                            "kind": "store",
                            "request": key,
                            "watermark": snapshot.computed_tokens,
                            "step": scheduler.jit_step,
                        }
                    )
                for key in output.kv_connector_metadata.loads:
                    events.append(
                        {"kind": "load", "request": key, "step": scheduler.jit_step}
                    )
                batches.append(
                    {
                        "step": scheduler.jit_step,
                        "requests": list(output.num_scheduled_tokens),
                        "admission": scheduler.groups.admission_limit,
                        "tokens": dict(output.num_scheduled_tokens),
                        "running_prefills_before": sorted(running_prefills),
                        "execution": (
                            {
                                "reserved": dict(scheduler.last_execution.reserved),
                                "commit_order": scheduler.last_execution.commit_order,
                                "stopped_at": scheduler.last_execution.stopped_at,
                                "preempt": scheduler.last_execution.preempt,
                            }
                            if scheduler.last_execution is not previous_execution
                            else None
                        ),
                    }
                )
            return output

        scheduler.schedule = observe
        try:
            for step in range(4096):
                if step == 4:
                    add(
                        f"{name}-urgent",
                        512 if expanding else 48,
                        1000 if expanding else 1,
                        short=not expanding,
                    )
                if expanding and step == 12:
                    add(f"{name}-third", 1024, 100)
                if expanding and step == 20:
                    add(f"{name}-fourth", 1536, 1, short=True)
                for output in engine.step():
                    if output.finished:
                        outputs[output.request_id] = {
                            "tokens": list(output.outputs[0].token_ids),
                            "finish_reason": output.outputs[0].finish_reason,
                        }
                if (
                    name == "cancel_offloaded"
                    and not cancelled
                    and (bool(events) if jit else step == 8)
                ):
                    engine.abort_request([f"{name}-initial", f"{name}-urgent"])
                    cancelled = True
                if (
                    step >= 20
                    and not engine.has_unfinished_requests()
                    and not scheduler.has_requests()
                ):
                    break
            else:
                raise AssertionError(f"{name}: engine did not drain")
        finally:
            scheduler.schedule = original_schedule
        case = {
            "outputs": outputs,
            "events": events,
            "batches": batches,
            "cancelled": cancelled,
        }
        result["cases"][name] = case
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        if jit:
            stats = core.jitserve_stats()
            workers = engine.collective_rpc(worker_swap_stats)
            case["scheduler"] = stats
            case["workers_delta"] = [
                {key: current[key] - prev[key] for key in current}
                for current, prev in zip(workers, workers_before, strict=True)
            ]
            assert (
                stats["cpu_reserved_bytes"]
                == stats["cpu_snapshots"]
                == stats["live_requests"]
                == 0
            )
            assert all(w["cpu_snapshots"] == 0 for w in workers)
            if name == "sliding_open_upper":
                assert not events, (
                    "Open upper bound must retain the longer running request"
                )
            else:
                assert any(e["kind"] == "store" for e in events)
                assert all(e["step"] % 4 == 0 for e in events if e["kind"] == "store")
                assert all(w["completed_stores"] > 0 for w in case["workers_delta"])
            if name in {"length_bin_swap", "sliding_expand"}:
                assert any(e["kind"] == "load" for e in events)
                assert all(w["completed_loads"] > 0 for w in case["workers_delta"])
            if expanding:
                assert any(b["admission"] == 3 for b in batches)
                assert any(b["admission"] == 4 for b in batches)
                assert (
                    stats["capacity_limited_rounds"] > before["capacity_limited_rounds"]
                )
                assert any(
                    b["execution"]
                    and any(
                        b["execution"]["reserved"].get(key) == b["tokens"].get(key) == 2
                        for key in b["running_prefills_before"]
                    )
                    for b in batches
                ), "No running prefill executed its two-token SLO reservation"
                assert any(
                    b["execution"] and b["execution"]["stopped_at"] for b in batches
                ), "Execution never exercised stop_allocate"
            if not cancelled:
                assert outputs == baseline["cases"][name]["outputs"], name
                case["tokens_match_stock"] = True
        if not cancelled:
            assert len(outputs) == (4 if expanding else 2)
        case["passed"] = True
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({"case": name, "passed": True, "events": events}), flush=True)


if __name__ == "__main__":
    main()
