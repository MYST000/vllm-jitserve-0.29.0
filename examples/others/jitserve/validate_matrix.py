# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-model swap checks at recurrent-state boundaries and during prefill.

Run stock and JITServe in separate processes. Only scheduling metadata and
generated token IDs are recorded; prompts and model state are not logged.
"""

import argparse
import json
import time
from pathlib import Path

from validate_swap import create_engine, worker_swap_stats

from vllm import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    jit = args.baseline is not None
    baseline = json.loads(args.baseline.read_text()) if jit else None
    engine = create_engine(
        args, max_model_len=4096, schedule_interval=2, max_num_seqs=args.max_num_seqs
    )
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    tokenizer = engine.get_tokenizer()
    block_size = engine.vllm_config.cache_config.block_size
    result = {
        "mode": "jitserve" if jit else "stock",
        "model": str(Path(args.model).resolve()),
        "tensor_parallel_size": args.tensor_parallel_size,
        "block_size": block_size,
        "max_num_seqs": args.max_num_seqs,
        "cases": {},
    }
    if baseline is not None:
        for key in ("model", "tensor_parallel_size", "block_size"):
            assert result[key] == baseline[key], key
        assert args.max_num_seqs == baseline.get("max_num_seqs", 1)

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def prompt(length=None, urgent=False):
        content = (
            "Reply with only the word OK."
            if urgent
            else "List the integers from 1 to 100, separated by commas."
        )
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Ignore filler text in this message."},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
        if length is not None:
            assert length >= len(tokens)
            index = tokens.index(tokenizer.convert_tokens_to_ids("<|im_end|>"))
            filler = tokenizer.encode(" filler", add_special_tokens=False)[0]
            tokens[index:index] = [filler] * (length - len(tokens))
        return {"prompt_token_ids": tokens}

    def add(name, tokens, prediction, max_tokens=96):
        return engine.add_request(
            name,
            tokens,
            SamplingParams(
                temperature=0,
                max_tokens=max_tokens,
                extra_args={
                    "jitserve_request_type": 1,
                    "jitserve_deadline": time.time() + 600,
                    "jitserve_output_len": prediction,
                    "jitserve_collection_id": name,
                },
            ),
        )

    original_schedule = scheduler.schedule
    events = []
    saved = {}
    reused = {}
    max_scheduled_requests = 0

    def observe_schedule(*a, **kw):
        nonlocal max_scheduled_requests
        output = original_schedule(*a, **kw)
        max_scheduled_requests = max(
            max_scheduled_requests, len(output.num_scheduled_tokens)
        )
        if jit:
            metadata = output.kv_connector_metadata
            for rid, snapshot in metadata.stores.items():
                saved[rid] = snapshot
                reused[rid] = 0
                events.append({"kind": "store", "tokens": snapshot.computed_tokens})
            for rid, snapshot in saved.items():
                for other in output.num_scheduled_tokens:
                    if other == rid:
                        continue
                    other_ids = scheduler.kv_cache_manager.get_blocks(
                        other
                    ).get_block_ids()
                    # HMA groups overlay the same allocation: a freed attention
                    # block can be overwritten by a recurrent-state group.
                    source_blocks = {b for group in snapshot.block_ids for b in group}
                    dest_blocks = {b for group in other_ids for b in group}
                    reused[rid] += len(
                        source_blocks.intersection(dest_blocks) - {-1, 0}
                    )
            for rid, ids in metadata.loads.items():
                snapshot = saved.pop(rid)
                relocated = sum(
                    old >= 0 and pos < len(dest) and dest[pos] >= 0 and old != dest[pos]
                    for source, dest in zip(snapshot.block_ids, ids, strict=True)
                    for pos, old in enumerate(source)
                )
                events.append(
                    {
                        "kind": "load",
                        "tokens": snapshot.computed_tokens,
                        "scheduled_tokens": output.num_scheduled_tokens[rid],
                        "relocated_blocks": relocated,
                        "blocks_reused_by_other_requests": reused.pop(rid),
                    }
                )
            for rid in metadata.drops:
                saved.pop(rid, None)
                reused.pop(rid, None)
        return output

    scheduler.schedule = observe_schedule
    short_len = max(
        len(prompt()["prompt_token_ids"]), 128 if args.max_num_seqs == 2 else 0
    )
    cases = [
        ("decode_repeat", short_len, [short_len + n for n in (7, 15, 23)], False),
        *[
            (f"boundary_{delta:+d}", block_size - 8, [block_size + delta], False)
            for delta in (-1, 0, 1)
        ],
        ("prefill_repeat", 2 * block_size + 64, [128, 256, block_size], False),
        ("cancel_offloaded", short_len, [short_len + 7], True),
    ]
    for name, length, watermarks, cancel in cases:
        assert engine.reset_prefix_cache()
        events.clear()
        max_scheduled_requests = 0
        before = engine.collective_rpc(worker_swap_stats) if jit else None
        background = []
        if args.max_num_seqs == 2:
            background = [f"{name}-background"]
            background_id = add(background[0], prompt(), 1, max_tokens=1024)
            # A source-aligned SLO round may admit nothing from a cold queue.
            # Prime actual decode, not merely one call to engine.step().
            for _ in range(4096):
                engine.step()
                if scheduler.requests[background_id].num_output_tokens:
                    break
            else:
                raise AssertionError("Background request did not reach decode")
        long_id = add(name, prompt(length), 10000)
        next_urgent = 0
        outputs = {}
        background_tokens = []
        cancelled = False
        for _ in range(4096):
            req = scheduler.requests.get(long_id)
            if (
                req is not None
                and next_urgent < len(watermarks)
                and req.num_computed_tokens >= watermarks[next_urgent]
            ):
                assert req.num_computed_tokens == watermarks[next_urgent], (
                    name,
                    req.num_computed_tokens,
                    watermarks[next_urgent],
                )
                add(f"{name}-urgent-{next_urgent}", prompt(urgent=True), 4)
                if jit:
                    # This matrix isolates exact restore watermarks. Align the
                    # policy round to the injection; natural cadence is covered
                    # separately in validate_policy.py and scheduler regressions.
                    scheduler.jit_step += (
                        -scheduler.jit_step - 1
                    ) % scheduler.jit_policy.interval
                next_urgent += 1
            for output in engine.step():
                if output.request_id in background:
                    assert not output.finished, "Background must remain running"
                    background_tokens = list(output.outputs[0].token_ids)
                if output.finished:
                    outputs[output.request_id] = {
                        "tokens": list(output.outputs[0].token_ids),
                        "finish_reason": output.outputs[0].finish_reason,
                    }
            if cancel and next_urgent and not cancelled:
                if jit:
                    assert long_id in scheduler.jit_connector.snapshots
                engine.abort_request([name, f"{name}-urgent-0", *background])
                cancelled = True
            if background and len(outputs) == len(watermarks) + 1:
                engine.abort_request(background)
            if not engine.has_unfinished_requests() and not scheduler.has_requests():
                break
        else:
            raise AssertionError(f"{name}: engine did not drain")
        assert next_urgent == len(watermarks), "Missing a requested injection point"
        case = {
            "prompt_tokens": length,
            "watermarks": watermarks,
            "outputs": outputs,
            "events": list(events),
            "cancelled": cancelled,
            "max_scheduled_requests": max_scheduled_requests,
            "background_tokens": background_tokens,
        }
        result["cases"][name] = case
        assert max_scheduled_requests == args.max_num_seqs
        if jit:
            case["scheduler"] = core.jitserve_stats()
            workers = engine.collective_rpc(worker_swap_stats)
            case["workers_delta"] = [
                {k: stats[k] - prev[k] for k in stats}
                for stats, prev in zip(workers, before, strict=True)
            ]
            save()
            assert case["scheduler"]["cpu_reserved_bytes"] == 0
            assert case["scheduler"]["cpu_snapshots"] == 0
            assert case["scheduler"]["live_requests"] == 0
            assert len(workers) == args.tensor_parallel_size
            for stats in case["workers_delta"]:
                assert stats["completed_stores"] == len(watermarks)
                assert stats["completed_loads"] == (0 if cancel else len(watermarks))
                assert stats["cpu_snapshots"] == 0
            assert [e["tokens"] for e in events if e["kind"] == "store"] == watermarks
            if not cancel:
                assert all(
                    e["relocated_blocks"] > 0
                    or e["blocks_reused_by_other_requests"] > 0
                    for e in events
                    if e["kind"] == "load"
                )
                assert outputs == baseline["cases"][name]["outputs"], name
                case["tokens_match_stock"] = True
            if background:
                reference = baseline["cases"][name]["background_tokens"]
                common = min(len(reference), len(background_tokens))
                assert common > 0 and reference[:common] == background_tokens[:common]
                case["background_common_tokens_match"] = common
        if not cancel:
            assert len(outputs) == len(watermarks) + 1
        case["passed"] = True
        save()
        print(json.dumps({"case": name, "passed": True, "events": events}), flush=True)


if __name__ == "__main__":
    main()
