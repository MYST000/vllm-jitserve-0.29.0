# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-owned CPU snapshots for JITServe active-request preemption.

Stores execute before freed GPU pages can be overwritten. Loads execute before
the resumed forward. CPU snapshots are consumed, never treated as a prefix cache.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

logger = init_logger(__name__)


@dataclass
class Snapshot:
    computed_tokens: int
    block_ids: tuple[list[int], ...]
    size_bytes: int


@dataclass
class SwapMetadata(KVConnectorMetadata):
    stores: dict[str, Snapshot] = field(default_factory=dict)
    loads: dict[str, tuple[list[int], ...]] = field(default_factory=dict)
    drops: set[str] = field(default_factory=set)


class JITServeConnector(KVConnectorBase_V1, SupportsHMA):
    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.config = kv_cache_config
        additional = vllm_config.additional_config
        if not isinstance(additional, dict):
            raise ValueError("JITServe additional_config must be a dictionary")
        options = additional.get("jitserve", {})
        self.capacity = int(options.get("cpu_bytes_to_use", 8 * 1024**3))
        self.world_size = vllm_config.parallel_config.world_size
        if self.capacity <= 0:
            raise ValueError("JITServe cpu_bytes_to_use must be positive")
        self.snapshots: dict[str, Snapshot] = {}
        self.reservations: dict[str, int] = {}
        self.pending = SwapMetadata()
        self.used_bytes = 0
        self.pages: dict[str, torch.Tensor] = {}
        self.cpu: dict[str, dict[str, tuple[list[int], torch.Tensor]]] = {}
        self.page_sizes: dict[str, int] = {}
        for group in self.config.kv_cache_groups:
            for name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                self.page_sizes[name] = spec.page_size_bytes
        self.num_stores = 0
        self.num_loads = 0
        self.stored_bytes = 0
        self.loaded_bytes = 0
        self.completed_stores = 0
        self.completed_loads = 0
        self.completed_store_bytes = 0
        self.completed_load_bytes = 0
        self._prepared_metadata: SwapMetadata | None = None

    def worker_stats(self) -> dict[str, int]:
        return {
            "completed_stores": self.completed_stores,
            "completed_loads": self.completed_loads,
            "completed_store_bytes": self.completed_store_bytes,
            "completed_load_bytes": self.completed_load_bytes,
            "cpu_snapshots": len(self.cpu),
        }

    def snapshot_bytes(self, block_ids: tuple[list[int], ...]) -> int:
        return (
            sum(
                sum(block >= 0 for block in ids)
                * sum(self.page_sizes[name] for name in group.layer_names)
                for ids, group in zip(
                    block_ids, self.config.kv_cache_groups, strict=True
                )
            )
            * self.world_size
        )

    def can_store(self, block_ids: tuple[list[int], ...]) -> bool:
        return self.used_bytes + self.snapshot_bytes(block_ids) <= self.capacity

    def save_request(self, request, blocks) -> None:
        ids = tuple(
            [block.block_id if not block.is_null else -1 for block in group]
            for group in blocks.blocks
        )
        size = self.snapshot_bytes(ids)
        if self.used_bytes + size > self.capacity:
            raise MemoryError("JITServe CPU swap capacity exhausted")
        if request.request_id in self.reservations:
            raise RuntimeError("Cannot overwrite an unconsumed JITServe snapshot")
        if request.num_in_flight_tokens or request.num_output_placeholders:
            raise RuntimeError("JITServe swap requires completed GPU steps")
        snapshot = Snapshot(request.num_computed_tokens, ids, size)
        self.snapshots[request.request_id] = snapshot
        self.reservations[request.request_id] = size
        self.pending.stores[request.request_id] = snapshot
        self.used_bytes += size
        self.num_stores += 1
        self.stored_bytes += size

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        snapshot = self.snapshots.get(request.request_id)
        if snapshot is None:
            return 0, False
        if num_computed_tokens:
            raise RuntimeError("JITServe snapshots require a private resume allocation")
        return snapshot.computed_tokens, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        snapshot = self.snapshots.get(request.request_id)
        if snapshot is None:
            return
        if num_external_tokens != snapshot.computed_tokens:
            raise RuntimeError(
                "JITServe resume lost the saved computed-token watermark"
            )
        self.pending.loads[request.request_id] = tuple(
            [block.block_id if not block.is_null else -1 for block in group]
            for group in blocks.blocks
        )
        self.num_loads += 1
        self.loaded_bytes += snapshot.size_bytes
        del self.snapshots[request.request_id]

    def complete_step(self, metadata: SwapMetadata) -> None:
        # Stores run before loads on the worker. Keep load/drop reservations
        # through the whole synchronous step, including the peak host usage.
        for request_id in metadata.loads.keys() | metadata.drops:
            self.used_bytes -= self.reservations.pop(request_id, 0)

    def build_connector_meta(self, scheduler_output):
        metadata, self.pending = self.pending, SwapMetadata()
        return metadata

    def request_finished(self, request, block_ids):
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        self.snapshots.pop(request.request_id, None)
        self.pending.drops.add(request.request_id)
        return False, None

    def register_kv_caches(self, kv_caches):
        # KVCacheTensor describes physical byte layout, including packed layers
        # and Mamba states; tensor axis conventions are backend-specific.
        layouts = {
            name: (tensor, index)
            for tensor in self.config.kv_cache_tensors
            for index, name in enumerate(tensor.layers)
        }
        for name, page_size in self.page_sizes.items():
            tensor = kv_caches[name]
            layout, layer_index = layouts[name]
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                tensor.untyped_storage()
            )
            self.pages[name] = torch.as_strided(
                raw,
                size=(self.config.num_blocks, page_size),
                stride=(layout.block_stride, 1),
                storage_offset=layout.offset + layer_index * layout.layer_stride,
            )

    def handle_preemptions(self, kv_connector_metadata):
        assert isinstance(kv_connector_metadata, SwapMetadata)
        for request_id, snapshot in kv_connector_metadata.stores.items():
            if request_id in self.cpu:
                raise RuntimeError("JITServe worker snapshot already exists")
            start = time.perf_counter()
            saved = {}
            size = 0
            for ids, group in zip(
                snapshot.block_ids, self.config.kv_cache_groups, strict=True
            ):
                positions = [i for i, block in enumerate(ids) if block >= 0]
                for name in group.layer_names:
                    pages = self.pages[name]
                    host = torch.empty(
                        (len(positions), pages.shape[1]),
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=pages.is_cuda,
                    )
                    for row, position in enumerate(positions):
                        host[row].copy_(pages[ids[position]], non_blocking=True)
                    saved[name] = (positions, host)
                    size += host.numel()
            self._synchronize()
            self.cpu[request_id] = saved
            self.completed_stores += 1
            self.completed_store_bytes += size
            logger.info(
                "JITServe D2H request=%s bytes=%d seconds=%.6f",
                request_id,
                size,
                time.perf_counter() - start,
            )
        for request_id in kv_connector_metadata.drops:
            self.cpu.pop(request_id, None)

    def load_kv_before_preprocess(self, metadata):
        assert isinstance(metadata, SwapMetadata)
        for request_id, ids_by_group in metadata.loads.items():
            saved = self.cpu[request_id]
            start = time.perf_counter()
            size = 0
            for ids, group in zip(
                ids_by_group, self.config.kv_cache_groups, strict=True
            ):
                for name in group.layer_names:
                    positions, host = saved[name]
                    spec = group.kv_cache_spec
                    if isinstance(spec, UniformTypeKVCacheSpecs):
                        spec = spec.kv_cache_specs[name]
                    needs_all = isinstance(spec, FullAttentionSpec)
                    for row, position in enumerate(positions):
                        if position < len(ids) and ids[position] >= 0:
                            self.pages[name][ids[position]].copy_(
                                host[row], non_blocking=True
                            )
                            size += host.shape[1]
                        elif needs_all or position == positions[-1]:
                            raise RuntimeError(
                                f"JITServe resume lost required page: {name}/{position}"
                            )
            self._synchronize()
            del self.cpu[request_id]
            self.completed_loads += 1
            self.completed_load_bytes += size
            logger.info(
                "JITServe H2D request=%s bytes=%d seconds=%.6f",
                request_id,
                size,
                time.perf_counter() - start,
            )
        self._prepared_metadata = metadata

    def start_load_kv(self, forward_context, **kwargs: Any):
        metadata = self._get_connector_metadata()
        if metadata is not self._prepared_metadata:
            raise RuntimeError("JITServe needs the pre-preprocessing restore hook")

    def clear_connector_metadata(self):
        super().clear_connector_metadata()
        self._prepared_metadata = None

    def _synchronize(self):
        devices = {page.device for page in self.pages.values() if page.is_cuda}
        for device in devices:
            torch.cuda.current_stream(device).synchronize()

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def get_finished(self, finished_req_ids):
        for request_id in finished_req_ids:
            self.cpu.pop(request_id, None)
        return None, None
