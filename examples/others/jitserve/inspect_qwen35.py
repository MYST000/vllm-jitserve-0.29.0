# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect local Qwen3.5 weights/configuration without loading or using a GPU."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("/docker/data/HF_MODELS/Qwen3.5-27B")
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    args = parser.parse_args()
    data = (args.model / "config.json").read_bytes()
    config = json.loads(data)
    text = config["text_config"]
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    missing = sorted(shard for shard in shards if not (args.model / shard).is_file())
    if missing:
        raise FileNotFoundError(f"Missing weight shards: {missing}")
    for name in (
        "num_attention_heads",
        "num_key_value_heads",
        "linear_num_key_heads",
        "linear_num_value_heads",
    ):
        if args.tensor_parallel_size <= 0 or text[name] % args.tensor_parallel_size:
            raise ValueError(f"TP size must divide {name}={text[name]}")
    result = {
        "model": str(args.model.resolve()),
        "config_sha256": hashlib.sha256(data).hexdigest(),
        "architecture": config["architectures"],
        "layers": dict(Counter(text["layer_types"])),
        "checkpoint_bytes": index["metadata"]["total_size"],
        "weight_shards": len(shards),
        "tensor_parallel_size": args.tensor_parallel_size,
        "validation": "configuration_and_shard_presence_only",
        "gpu_inference": "pending",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
