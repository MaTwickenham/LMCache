from __future__ import annotations

import cloudpickle
import copy
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

from vllm import LLM, SamplingParams


TierName = Literal["cpu", "gpu", "miss"]

LOCAL_CPU_BACKEND = "LocalCPUBackend"
LOCAL_GPU_BACKEND = "LocalGPUBackend"
FRAGMENT_REQUEST_CONFIG = {"lmcache.request_kind": "fragment_prefill"}


def _worker_probe_lmcache_engine(_worker_wrapper: Any) -> dict[str, Any]:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if engine is None:
        return {
            "engine_available": False,
            "storage_manager_available": False,
            "backends": [],
        }

    storage_manager = engine.storage_manager
    backend_names = []
    if storage_manager is not None:
        backend_names = sorted(storage_manager.storage_backends.keys())

    return {
        "engine_available": True,
        "storage_manager_available": storage_manager is not None,
        "backends": backend_names,
    }


def _worker_move_fragment(
    _worker_wrapper: Any,
    prompt_ids: list[int],
    src: str,
    dst: str,
    lookup_id: str,
    request_configs: dict[str, Any],
    remove_src: bool = True,
) -> dict[str, Any]:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if engine is None or engine.storage_manager is None:
        raise RuntimeError("LMCache engine/storage manager is not available in worker.")

    found = engine.lookup(
        list(prompt_ids),
        search_range=[src],
        lookup_id=lookup_id,
        pin=True,
        request_configs=request_configs,
    )
    if not found:
        engine.lookup_unpin(lookup_id)
        raise RuntimeError(f"Cannot move fragment: not found in {src}.")

    block_mapping = engine.lookup_pins.get(lookup_id, {})
    keys = list(block_mapping.get(src, []))
    if not keys:
        engine.lookup_unpin(lookup_id)
        raise RuntimeError(f"Cannot move fragment: no pinned keys in {src}.")

    memory_objs = engine.storage_manager.batched_get(keys=keys, location=src)
    if any(memory_obj is None for memory_obj in memory_objs):
        engine.lookup_unpin(lookup_id)
        raise RuntimeError(
            f"Cannot move fragment: failed to fetch memory objects from {src}."
        )

    engine.storage_manager.batched_put(
        keys=keys,
        memory_objs=memory_objs,  # type: ignore[arg-type]
        location=dst,
    )
    engine.lookup_unpin(lookup_id)
    if remove_src:
        engine.storage_manager.batched_remove(keys, locations=[src])

    return {
        "num_tokens": int(found),
        "num_keys": len(keys),
        "src": src,
        "dst": dst,
    }


def _worker_evict_fragment(
    _worker_wrapper: Any,
    prompt_ids: list[int],
    location: str,
    request_configs: dict[str, Any],
) -> int:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if engine is None:
        raise RuntimeError("LMCache engine is not available in worker.")

    return int(
        engine.clear(
            tokens=list(prompt_ids),
            locations=[location],
            request_configs=request_configs,
        )
    )


@dataclass
class QueryReuseSnapshot:
    locations: dict[str, TierName]
    hit_tokens: int
    gpu_hit_tokens: int
    cpu_hit_tokens: int
    miss_tokens: int
    hit_fragments: int
    gpu_hit_fragments: int
    cpu_hit_fragments: int
    missed_fragments: list[str]


@dataclass
class MaintenanceStats:
    wall_s: float = 0.0
    materialize_s: float = 0.0
    move_s: float = 0.0
    evict_s: float = 0.0
    admitted_chunks: int = 0
    promoted_chunks: int = 0
    demoted_chunks: int = 0
    evicted_chunks: int = 0
    admitted_tokens: int = 0
    promoted_tokens: int = 0
    demoted_tokens: int = 0
    evicted_tokens: int = 0

    def absorb(self, other: "MaintenanceStats") -> None:
        self.wall_s += float(other.wall_s)
        self.materialize_s += float(other.materialize_s)
        self.move_s += float(other.move_s)
        self.evict_s += float(other.evict_s)
        self.admitted_chunks += int(other.admitted_chunks)
        self.promoted_chunks += int(other.promoted_chunks)
        self.demoted_chunks += int(other.demoted_chunks)
        self.evicted_chunks += int(other.evicted_chunks)
        self.admitted_tokens += int(other.admitted_tokens)
        self.promoted_tokens += int(other.promoted_tokens)
        self.demoted_tokens += int(other.demoted_tokens)
        self.evicted_tokens += int(other.evicted_tokens)


class LMCacheFragmentRuntime:
    def __init__(self, llm: LLM, base_sampling_params: SamplingParams):
        self.llm = llm
        self.base_sampling_params = base_sampling_params
        self._worker_engine_verified = False

    def _collective_rpc(self, func: Any, *args: Any, **kwargs: Any) -> list[Any]:
        return self.llm.collective_rpc(
            cloudpickle.dumps(func),
            args=args,
            kwargs=kwargs or None,
        )

    def _ensure_worker_engine(self) -> None:
        if self._worker_engine_verified:
            return
        results = self._collective_rpc(_worker_probe_lmcache_engine)
        if not results:
            raise RuntimeError("LMCache worker probe returned no results.")

        bad_results = [
            result
            for result in results
            if (
                not bool(result.get("engine_available"))
                or not bool(result.get("storage_manager_available"))
            )
        ]
        if bad_results:
            raise RuntimeError(
                "LMCache worker engine is not available on all workers: "
                f"{bad_results}"
            )
        self._worker_engine_verified = True

    def make_sampling_params(
        self,
        *,
        request_kind: str,
        store_location: str | None = None,
        skip_save: bool | None = None,
        execution_mode: str | None = None,
    ) -> SamplingParams:
        cloned = copy.deepcopy(self.base_sampling_params)
        extra_args = dict(cloned.extra_args or {})
        kv_transfer_params = dict(extra_args.get("kv_transfer_params") or {})
        kv_transfer_params["lmcache.request_kind"] = request_kind
        if store_location is not None:
            kv_transfer_params["lmcache.store_location"] = store_location
        if skip_save is not None:
            kv_transfer_params["lmcache.skip_save"] = bool(skip_save)
        if execution_mode is not None:
            kv_transfer_params["lmcache.execution_mode"] = str(execution_mode)
        extra_args["kv_transfer_params"] = kv_transfer_params
        cloned.extra_args = extra_args
        return cloned

    def materialize_fragment(self, prefill_record: Any, *, location: str) -> float:
        start = time.perf_counter()
        self.llm.generate(
            prompts={"prompt_token_ids": list(prefill_record.prompt_ids)},
            sampling_params=self.make_sampling_params(
                request_kind="fragment_prefill",
                store_location=location,
            ),
            use_tqdm=False,
        )
        return time.perf_counter() - start

    def move_fragment(
        self,
        prefill_record: Any,
        *,
        src: str,
        dst: str,
        remove_src: bool = True,
    ) -> float:
        self._ensure_worker_engine()
        request_id = f"explicit-move-{uuid.uuid4().hex}"
        start = time.perf_counter()
        self._collective_rpc(
            _worker_move_fragment,
            list(prefill_record.prompt_ids),
            src,
            dst,
            request_id,
            FRAGMENT_REQUEST_CONFIG,
            remove_src,
        )
        return time.perf_counter() - start

    def evict_fragment(self, prefill_record: Any, *, location: str) -> float:
        self._ensure_worker_engine()
        start = time.perf_counter()
        self._collective_rpc(
            _worker_evict_fragment,
            list(prefill_record.prompt_ids),
            location,
            FRAGMENT_REQUEST_CONFIG,
        )
        return time.perf_counter() - start


class ExplicitFragmentPool:
    def __init__(
        self,
        *,
        runtime: LMCacheFragmentRuntime,
        prefill_records: dict[str, Any],
        fragment_tokens: dict[str, int],
        cpu_budget_tokens: int,
        gpu_budget_tokens: int,
        cpu_enabled: bool,
        gpu_enabled: bool,
        admit_misses_to: Literal["auto", "cpu", "gpu", "none"] = "auto",
        gpu_lookahead: int = 0,
    ):
        self.runtime = runtime
        self.prefill_records = prefill_records
        self.fragment_tokens = fragment_tokens
        self.cpu_budget_tokens = max(int(cpu_budget_tokens), 0)
        self.gpu_budget_tokens = max(int(gpu_budget_tokens), 0)
        self.cpu_enabled = bool(cpu_enabled and self.cpu_budget_tokens > 0)
        self.gpu_enabled = bool(gpu_enabled and self.gpu_budget_tokens > 0)
        self.admit_misses_to = admit_misses_to
        self.gpu_lookahead = max(int(gpu_lookahead), 0)

        self.cpu_lru: OrderedDict[str, None] = OrderedDict()
        self.gpu_lru: OrderedDict[str, None] = OrderedDict()
        self.cpu_used_tokens = 0
        self.gpu_used_tokens = 0

    def _touch(self, chunk_id: str) -> None:
        if chunk_id in self.gpu_lru:
            self.gpu_lru.move_to_end(chunk_id)
        elif chunk_id in self.cpu_lru:
            self.cpu_lru.move_to_end(chunk_id)

    def _location_of(self, chunk_id: str) -> TierName:
        if chunk_id in self.gpu_lru:
            return "gpu"
        if chunk_id in self.cpu_lru:
            return "cpu"
        return "miss"

    def _tier_backend_name(self, tier: TierName) -> str:
        if tier == "gpu":
            return LOCAL_GPU_BACKEND
        if tier == "cpu":
            return LOCAL_CPU_BACKEND
        raise ValueError(f"Unsupported tier: {tier}")

    def _tier_budget(self, tier: TierName) -> int:
        return self.gpu_budget_tokens if tier == "gpu" else self.cpu_budget_tokens

    def _tier_used(self, tier: TierName) -> int:
        return self.gpu_used_tokens if tier == "gpu" else self.cpu_used_tokens

    def _set_tier_used(self, tier: TierName, value: int) -> None:
        if tier == "gpu":
            self.gpu_used_tokens = value
        else:
            self.cpu_used_tokens = value

    def _tier_lru(self, tier: TierName) -> OrderedDict[str, None]:
        return self.gpu_lru if tier == "gpu" else self.cpu_lru

    def _can_use_tier(self, tier: TierName) -> bool:
        if tier == "gpu":
            return self.gpu_enabled
        if tier == "cpu":
            return self.cpu_enabled
        return False

    def _remove_from_tier_state(self, chunk_id: str, tier: TierName) -> None:
        tokens = int(self.fragment_tokens[chunk_id])
        lru = self._tier_lru(tier)
        if chunk_id in lru:
            del lru[chunk_id]
            self._set_tier_used(tier, max(self._tier_used(tier) - tokens, 0))

    def _insert_into_tier_state(self, chunk_id: str, tier: TierName) -> None:
        lru = self._tier_lru(tier)
        lru[chunk_id] = None
        lru.move_to_end(chunk_id)
        self._set_tier_used(
            tier, self._tier_used(tier) + int(self.fragment_tokens[chunk_id])
        )

    def _evict_one_cpu(self) -> MaintenanceStats:
        if not self.cpu_lru:
            return MaintenanceStats()
        victim, _ = self.cpu_lru.popitem(last=False)
        wall_s = self.runtime.evict_fragment(
            self.prefill_records[victim],
            location=LOCAL_CPU_BACKEND,
        )
        tokens = int(self.fragment_tokens[victim])
        self.cpu_used_tokens = max(self.cpu_used_tokens - tokens, 0)
        return MaintenanceStats(
            wall_s=wall_s,
            evict_s=wall_s,
            evicted_chunks=1,
            evicted_tokens=tokens,
        )

    def _make_room_cpu(self, needed_tokens: int) -> MaintenanceStats:
        stats = MaintenanceStats()
        while (
            self.cpu_enabled
            and self.cpu_used_tokens + needed_tokens > self.cpu_budget_tokens
            and self.cpu_lru
        ):
            stats.absorb(self._evict_one_cpu())
        return stats

    def _demote_one_gpu(self) -> MaintenanceStats:
        if not self.gpu_lru:
            return MaintenanceStats()
        victim, _ = self.gpu_lru.popitem(last=False)
        tokens = int(self.fragment_tokens[victim])
        stats = MaintenanceStats()

        if self.cpu_enabled and tokens <= self.cpu_budget_tokens:
            stats.absorb(self._make_room_cpu(tokens))
            if self.cpu_used_tokens + tokens <= self.cpu_budget_tokens:
                wall_s = self.runtime.move_fragment(
                    self.prefill_records[victim],
                    src=LOCAL_GPU_BACKEND,
                    dst=LOCAL_CPU_BACKEND,
                    remove_src=True,
                )
                self.gpu_used_tokens = max(self.gpu_used_tokens - tokens, 0)
                self._insert_into_tier_state(victim, "cpu")
                stats.absorb(
                    MaintenanceStats(
                        wall_s=wall_s,
                        move_s=wall_s,
                        demoted_chunks=1,
                        demoted_tokens=tokens,
                    )
                )
                return stats

        wall_s = self.runtime.evict_fragment(
            self.prefill_records[victim],
            location=LOCAL_GPU_BACKEND,
        )
        self.gpu_used_tokens = max(self.gpu_used_tokens - tokens, 0)
        stats.absorb(
            MaintenanceStats(
                wall_s=wall_s,
                evict_s=wall_s,
                evicted_chunks=1,
                evicted_tokens=tokens,
            )
        )
        return stats

    def _make_room_gpu(self, needed_tokens: int) -> MaintenanceStats:
        stats = MaintenanceStats()
        while (
            self.gpu_enabled
            and self.gpu_used_tokens + needed_tokens > self.gpu_budget_tokens
            and self.gpu_lru
        ):
            stats.absorb(self._demote_one_gpu())
        return stats

    def capture_query_reuse(self, chunk_ids: list[str]) -> QueryReuseSnapshot:
        locations: dict[str, TierName] = {}
        hit_tokens = 0
        gpu_hit_tokens = 0
        cpu_hit_tokens = 0
        miss_tokens = 0
        hit_fragments = 0
        gpu_hit_fragments = 0
        cpu_hit_fragments = 0
        missed_fragments: list[str] = []

        for chunk_id in chunk_ids:
            location = self._location_of(chunk_id)
            locations[chunk_id] = location
            tokens = int(self.fragment_tokens[chunk_id])
            if location == "gpu":
                hit_tokens += tokens
                gpu_hit_tokens += tokens
                hit_fragments += 1
                gpu_hit_fragments += 1
            elif location == "cpu":
                hit_tokens += tokens
                cpu_hit_tokens += tokens
                hit_fragments += 1
                cpu_hit_fragments += 1
            else:
                miss_tokens += tokens
                missed_fragments.append(chunk_id)

        return QueryReuseSnapshot(
            locations=locations,
            hit_tokens=hit_tokens,
            gpu_hit_tokens=gpu_hit_tokens,
            cpu_hit_tokens=cpu_hit_tokens,
            miss_tokens=miss_tokens,
            hit_fragments=hit_fragments,
            gpu_hit_fragments=gpu_hit_fragments,
            cpu_hit_fragments=cpu_hit_fragments,
            missed_fragments=missed_fragments,
        )

    def note_query_access(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._touch(chunk_id)

    def seed_initial_resident_pool(
        self,
        *,
        chunk_order: list[str],
        tier_order: list[TierName],
    ) -> MaintenanceStats:
        stats = MaintenanceStats()
        for chunk_id in chunk_order:
            tokens = int(self.fragment_tokens[chunk_id])
            if self._location_of(chunk_id) != "miss":
                continue
            for tier in tier_order:
                if tier == "miss" or not self._can_use_tier(tier):
                    continue
                if tokens > self._tier_budget(tier):
                    continue
                if self._tier_used(tier) + tokens > self._tier_budget(tier):
                    continue
                location = self._tier_backend_name(tier)
                wall_s = self.runtime.materialize_fragment(
                    self.prefill_records[chunk_id],
                    location=location,
                )
                self._insert_into_tier_state(chunk_id, tier)
                stats.absorb(
                    MaintenanceStats(
                        wall_s=wall_s,
                        materialize_s=wall_s,
                        admitted_chunks=1,
                        admitted_tokens=tokens,
                    )
                )
                break
        return stats

    def _resolve_admission_tier(self) -> TierName:
        if self.admit_misses_to == "cpu":
            return "cpu"
        if self.admit_misses_to == "gpu":
            return "gpu"
        if self.admit_misses_to == "none":
            return "miss"
        if self.cpu_enabled:
            return "cpu"
        if self.gpu_enabled:
            return "gpu"
        return "miss"

    def materialize_or_move(
        self,
        chunk_id: str,
        *,
        target_tier: TierName,
    ) -> MaintenanceStats:
        current = self._location_of(chunk_id)
        tokens = int(self.fragment_tokens[chunk_id])
        if target_tier == "miss" or current == target_tier:
            self._touch(chunk_id)
            return MaintenanceStats()
        if not self._can_use_tier(target_tier):
            return MaintenanceStats()
        if tokens > self._tier_budget(target_tier):
            return MaintenanceStats()

        stats = MaintenanceStats()
        if target_tier == "gpu":
            stats.absorb(self._make_room_gpu(tokens))
            if self.gpu_used_tokens + tokens > self.gpu_budget_tokens:
                return stats
        else:
            stats.absorb(self._make_room_cpu(tokens))
            if self.cpu_used_tokens + tokens > self.cpu_budget_tokens:
                return stats

        target_backend = self._tier_backend_name(target_tier)
        if current == "miss":
            wall_s = self.runtime.materialize_fragment(
                self.prefill_records[chunk_id],
                location=target_backend,
            )
            self._insert_into_tier_state(chunk_id, target_tier)
            stats.absorb(
                MaintenanceStats(
                    wall_s=wall_s,
                    materialize_s=wall_s,
                    admitted_chunks=1,
                    admitted_tokens=tokens,
                )
            )
            return stats

        src_backend = self._tier_backend_name(current)
        wall_s = self.runtime.move_fragment(
            self.prefill_records[chunk_id],
            src=src_backend,
            dst=target_backend,
            remove_src=True,
        )
        self._remove_from_tier_state(chunk_id, current)
        self._insert_into_tier_state(chunk_id, target_tier)
        if current == "cpu" and target_tier == "gpu":
            stats.absorb(
                MaintenanceStats(
                    wall_s=wall_s,
                    move_s=wall_s,
                    promoted_chunks=1,
                    promoted_tokens=tokens,
                )
            )
        else:
            stats.absorb(
                MaintenanceStats(
                    wall_s=wall_s,
                    move_s=wall_s,
                    demoted_chunks=1,
                    demoted_tokens=tokens,
                )
            )
        return stats

    def run_inter_query_maintenance(
        self,
        *,
        current_chunk_ids: list[str],
        next_chunk_ids: list[str] | None = None,
    ) -> MaintenanceStats:
        start = time.perf_counter()
        stats = MaintenanceStats()

        self.note_query_access(current_chunk_ids)
        target_tier = self._resolve_admission_tier()
        if target_tier != "miss":
            for chunk_id in current_chunk_ids:
                if self._location_of(chunk_id) != "miss":
                    continue
                stats.absorb(
                    self.materialize_or_move(chunk_id, target_tier=target_tier)
                )

        if self.gpu_enabled and self.gpu_lookahead > 0 and next_chunk_ids:
            for chunk_id in next_chunk_ids:
                if self._location_of(chunk_id) == "gpu":
                    continue
                stats.absorb(
                    self.materialize_or_move(chunk_id, target_tier="gpu")
                )

        stats.wall_s = time.perf_counter() - start
        return stats

    def resident_counts(self) -> dict[str, int]:
        return {
            "cpu_fragments": len(self.cpu_lru),
            "gpu_fragments": len(self.gpu_lru),
            "cpu_tokens": self.cpu_used_tokens,
            "gpu_tokens": self.gpu_used_tokens,
        }
