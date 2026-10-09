# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run paired real OpenHands workflows against isolated stock/JITServe servers."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    gpu_state = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    (args.output / "gpus-before.txt").write_text(gpu_state)
    selected = set(args.gpus.split(","))
    for line in gpu_state.splitlines():
        index, memory = (value.strip() for value in line.split(","))
        if index in selected and int(memory) > 10:
            raise RuntimeError(f"GPU {index} is in use ({memory} MiB)")
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "HF_HUB_OFFLINE": "1",
        "JITSERVE_MAX_NUM_SEQS": "4",
        "JITSERVE_MAX_MODEL_LEN": "16384",
        "JITSERVE_ADDITIONAL_CONFIG": json.dumps(
            {
                "jitserve": {
                    "schedule_interval": 4,
                    "search_strategy": "sliding_window",
                    "top_k_selection": 2,
                    "max_swaps_per_iter": 1,
                    "max_batch_size": 2,
                    "cpu_bytes_to_use": 8589934592,
                    "trace_requests": True,
                }
            }
        ),
    }
    paired = {}
    for backend in ("stock-v1", "jitserve-v1-port"):
        phase = args.output / backend
        phase.mkdir()
        environment["JITSERVE_BACKEND"] = backend
        command = [
            "bash",
            "examples/others/jitserve/serve_qwen35.sh",
            "--kv-cache-memory-bytes",
            "6442450944",
        ]
        (phase / "launch.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "environment": {
                        key: value
                        for key, value in environment.items()
                        if key.startswith("JITSERVE_")
                        or key in {"CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE"}
                    },
                },
                indent=2,
            )
        )
        with (phase / "server.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                with httpx.Client(timeout=2, trust_env=False) as client:
                    ready_by = time.monotonic() + 300
                    while time.monotonic() < ready_by:
                        if process.poll() is not None:
                            raise RuntimeError(f"{backend} exited during startup")
                        try:
                            response = client.get("http://127.0.0.1:8011/health")
                            if response.status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(1)
                    else:
                        raise TimeoutError(f"{backend} did not become ready")
                client_command = [
                    str(args.sdk / ".venv/bin/python"),
                    str(args.sdk / "benchmarks/flowpilot/scripts/validate_jitserve.py"),
                    "--backend",
                    backend,
                    "--output",
                    str(phase / "workflows"),
                    "--rounds",
                    str(args.rounds),
                    "--concurrency",
                    *(str(value) for value in args.concurrency),
                ]
                client_environment = {
                    **os.environ,
                    "PYTHONPATH": str(args.sdk / "benchmarks/flowpilot/src"),
                    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                    "OPENHANDS_SUPPRESS_BANNER": "1",
                }
                with (phase / "client.log").open("w") as client_log:
                    subprocess.run(
                        client_command,
                        cwd=args.sdk,
                        env=client_environment,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                paired[backend] = json.loads(
                    (phase / "workflows/summary.json").read_text()
                )
            finally:
                process.terminate()
                process.wait(timeout=60)
        ingress = []
        for line in (phase / "server.log").read_text().splitlines():
            if "JITServe ingress " in line:
                ingress.append(json.loads(line.split("JITServe ingress ", 1)[1]))
        by_external = {row["external_request_id"]: row for row in ingress}
        links = []
        for event_path in (phase / "workflows").glob(
            "concurrency-*/round-*/events.jsonl"
        ):
            events = [json.loads(line) for line in event_path.read_text().splitlines()]
            identities = {
                e["jitserve"]["request_id"]: e["jitserve"]
                for e in events
                if e["event"] == "jitserve_request_identity"
            }
            for event in events:
                if event["event"] != "llm_response":
                    continue
                identity = identities[event["request_id"]]
                observed = by_external[event["llm_response_id"]]
                if backend == "jitserve-v1-port":
                    assert observed["slo_parameters"] == identity["slo_parameters"]
                else:
                    assert observed["slo_parameters"] == {}
                links.append({**identity, **observed})
        assert links
        (phase / "engine-identity.json").write_text(json.dumps(links, indent=2))
        print(
            json.dumps(
                {
                    "backend": backend,
                    "mapped_engine_requests": len(links),
                    "passed": True,
                }
            ),
            flush=True,
        )
    (args.output / "summary.json").write_text(
        json.dumps({"passed": True, "backends": paired}, indent=2)
    )


if __name__ == "__main__":
    main()
