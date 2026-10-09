# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate real OpenAI HTTP/SSE, Tool calls, SLO errors and disconnect cleanup."""

import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="Qwen3.5-27B-jitserve")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = httpx.Client(base_url=args.base_url, timeout=120)
    client.get("/health").raise_for_status()
    result = {}

    def record(name, value):
        result[name] = value
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({name: value}), flush=True)

    def body():
        return {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with only the word OK."}],
            "temperature": 0,
            "max_tokens": 96,
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {
                "jitserve_request_type": 1,
                "jitserve_deadline": time.time() + 600,
                "jitserve_output_len": 96,
                "jitserve_collection_id": "http-validation",
            },
        }

    def post(payload, route="/v1/chat/completions"):
        response = client.post(route, json=payload)
        response.raise_for_status()
        return response.json()

    def stream(payload, route="/v1/chat/completions"):
        events, done = [], False
        with client.stream("POST", route, json={**payload, "stream": True}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    continue
                event = json.loads(data)
                assert "error" not in event, event
                events.append(event)
        return events, done

    for name, value in (("missing", None), ("invalid", -1)):
        payload = body()
        if value is None:
            del payload["vllm_xargs"]
        else:
            payload["vllm_xargs"]["jitserve_output_len"] = value
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400, response.status_code
        assert "JITServe" in response.text or "jitserve" in response.text
        record(f"slo_{name}", {"status": response.status_code})

    plain = post(body())
    choice = plain["choices"][0]
    assert choice["message"]["content"].strip() == "OK"
    assert choice["finish_reason"] == "stop"
    payload = body()
    payload["stream_options"] = {"include_usage": True}
    chunks, done = stream(payload)
    text = "".join(
        c["delta"].get("content") or "" for e in chunks for c in e["choices"]
    )
    reasons = [
        c["finish_reason"] for e in chunks for c in e["choices"] if c["finish_reason"]
    ]
    assert done and reasons == ["stop"]
    assert text == choice["message"]["content"]
    assert (
        chunks[-1]["usage"]["completion_tokens"] == plain["usage"]["completion_tokens"]
    )
    record("chat_sse", {"matches_nonstream": True, "done": done, "chunks": len(chunks)})

    payload = body()
    payload["messages"] = [
        {
            "role": "user",
            "content": "What is 17 plus 25? Use add_integers to compute it.",
        }
    ]
    payload["max_tokens"] = 256
    payload["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "add_integers",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        }
    ]
    payload["tool_choice"] = "auto"
    tool_response = post(payload)["choices"][0]
    assert tool_response["finish_reason"] == "tool_calls"
    calls = tool_response["message"]["tool_calls"]
    assert len(calls) == 1 and calls[0]["id"]
    function = calls[0]["function"]
    assert function["name"] == "add_integers"
    arguments = json.loads(function["arguments"])
    assert arguments == {"a": 17, "b": 25}
    chunks, done = stream(payload)
    streamed_name, streamed_args, call_ids, reasons = "", "", [], []
    for event in chunks:
        for c in event["choices"]:
            if c["finish_reason"]:
                reasons.append(c["finish_reason"])
            for call in c["delta"].get("tool_calls") or []:
                assert call["index"] == 0
                if call.get("id"):
                    call_ids.append(call["id"])
                f = call.get("function") or {}
                streamed_name += f.get("name") or ""
                streamed_args += f.get("arguments") or ""
    assert done and reasons == ["tool_calls"] and len(call_ids) == 1
    assert streamed_name == function["name"]
    assert json.loads(streamed_args) == arguments
    # Execute the actual local arithmetic Tool, then supply its matching result.
    tool_result = arguments["a"] + arguments["b"]
    payload["messages"].extend(
        [
            tool_response["message"],
            {
                "role": "tool",
                "tool_call_id": calls[0]["id"],
                "content": str(tool_result),
            },
        ]
    )
    answer = post(payload)["choices"][0]
    assert answer["finish_reason"] == "stop"
    assert str(tool_result) in answer["message"]["content"]
    record(
        "tool_roundtrip", {"calls": 1, "stream_matches": True, "local_execution": True}
    )

    payload = body()
    payload["input"] = payload.pop("messages")
    payload["max_output_tokens"] = payload.pop("max_tokens")
    response = post(payload, "/v1/responses")
    assert response["status"] == "completed"
    chunks, _ = stream(payload, "/v1/responses")
    assert chunks[-1]["type"] == "response.completed"
    text = "".join(
        e["delta"] for e in chunks if e["type"] == "response.output_text.delta"
    )
    assert text.strip() == "OK"
    record("responses_sse", {"completed": True, "text_matches": True})

    def metrics():
        response = client.get("/metrics")
        response.raise_for_status()
        rows = [line for line in response.text.splitlines() if not line.startswith("#")]

        def total(prefix, label=""):
            values = [
                float(row.rsplit(" ", 1)[1])
                for row in rows
                if row.startswith(prefix + "{") and label in row
            ]
            assert values, f"Missing metric: {prefix}/{label}"
            return sum(values)

        return {
            "running": total("vllm:num_requests_running"),
            "waiting": total("vllm:num_requests_waiting"),
            "aborts": total("vllm:request_success_total", 'finished_reason="abort"'),
            "preemptions": total("vllm:num_preemptions_total"),
            "generation_tokens": total("vllm:generation_tokens_total"),
            "normal_finishes": sum(
                total("vllm:request_success_total", f'finished_reason="{reason}"')
                for reason in ("stop", "length", "error", "repetition")
            ),
        }

    payload = body()
    payload["messages"][0]["content"] = (
        "List the integers from 1 to 100, separated by commas."
    )
    payload["vllm_xargs"]["jitserve_output_len"] = 10000
    reference = post(payload)["choices"][0]
    before = metrics()
    text, reasons, injected, done = "", [], False, False
    with client.stream(
        "POST", "/v1/chat/completions", json={**payload, "stream": True}
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                done = True
                continue
            for c in json.loads(line[6:])["choices"]:
                content = c["delta"].get("content") or ""
                text += content
                if c["finish_reason"]:
                    reasons.append(c["finish_reason"])
                if content and not injected:
                    urgent = body()
                    urgent["vllm_xargs"]["jitserve_output_len"] = 4
                    assert (
                        post(urgent)["choices"][0]["message"]["content"].strip() == "OK"
                    )
                    injected = True
    assert injected and done
    assert text == reference["message"]["content"]
    assert reasons == [reference["finish_reason"]]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        after = metrics()
        if after["preemptions"] > before["preemptions"]:
            break
        time.sleep(0.5)
    else:
        raise AssertionError("Concurrent HTTP request did not trigger preemption")
    record(
        "http_preemption",
        {
            "preemption_delta": after["preemptions"] - before["preemptions"],
            "text_and_stopping_match": True,
        },
    )

    before = metrics()
    payload = body()
    payload["messages"][0]["content"] = (
        "List the integers from 1 to 1000, separated by commas."
    )
    payload.update(stream=True, max_tokens=2048)
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: {"):
                event = json.loads(line[6:])
                if any(c["delta"].get("content") for c in event["choices"]):
                    break
        else:
            raise AssertionError("No generated token before cancellation")
        active = metrics()
        assert active["running"] > 0, "Cancellation must target a running request"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        after = metrics()
        if after["running"] == after["waiting"] == 0:
            break
        time.sleep(0.5)
    else:
        raise AssertionError(f"Disconnect did not drain: {before} -> {after}")
    # Frontend abort removes OutputProcessor state before terminal stats are
    # recorded, so request_success{finished_reason="abort"} does not increment.
    assert after["normal_finishes"] == before["normal_finishes"]
    generated = after["generation_tokens"] - before["generation_tokens"]
    assert 0 < generated < payload["max_tokens"]
    time.sleep(1)
    settled = metrics()
    assert settled["generation_tokens"] == after["generation_tokens"]
    assert settled["running"] == settled["waiting"] == 0
    record(
        "disconnect",
        {
            "running_before_disconnect": active["running"],
            "running_after_disconnect": settled["running"],
            "waiting_after_disconnect": settled["waiting"],
            "generated_tokens": generated,
            "normal_finish_delta": after["normal_finishes"] - before["normal_finishes"],
            "abort_metric_delta": after["aborts"] - before["aborts"],
            "generation_stopped": True,
        },
    )
    client.get("/health").raise_for_status()
    record("passed", True)
    client.close()


if __name__ == "__main__":
    main()
