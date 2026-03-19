from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2


def tokens_to_bytes(tokens: int) -> int:
    return max(int(tokens), 0) * BYTES_PER_TOKEN


@dataclass(frozen=True)
class CostModel:
    recompute_fixed_ms: float = 0.0
    recompute_ms_per_token: float = 0.0
    transfer_fixed_ms: float = 0.0
    transfer_gib_per_s: float = 12.0

    def recompute_ms(self, tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        return self.recompute_fixed_ms + self.recompute_ms_per_token * float(tokens)

    def transfer_ms(self, bytes_size: int) -> float:
        if bytes_size <= 0:
            return 0.0
        gib = float(bytes_size) / float(1024**3)
        bw = max(self.transfer_gib_per_s, 1e-9)
        return self.transfer_fixed_ms + (gib / bw) * 1000.0

    @staticmethod
    def from_spec(spec: str) -> "CostModel":
        if not spec:
            return CostModel()
        payload = json.loads(spec)
        if not isinstance(payload, dict):
            raise ValueError("utility_cost_model must be a JSON object")
        allowed = {f.name for f in CostModel.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: payload[k] for k in payload if k in allowed}
        return CostModel(**filtered)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _log1p_safe(x: object) -> float:
    try:
        value = float(x)  # type: ignore[arg-type]
    except Exception:
        return 0.0
    if value <= 0:
        return 0.0
    return float(math.log1p(value))


def _float_or(x: object, default: float = 0.0) -> float:
    try:
        value = float(x)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    if math.isnan(value) or math.isinf(value):
        return float(default)
    return float(value)


@dataclass(frozen=True)
class UtilityPlannerConfig:
    cost_model: CostModel
    tail_lambda: float = 0.0
    w_layer: float = 1.0
    w_hotness: float = 0.2
    w_importance: float = 1.0
    w_graph: float = 0.2
    w_tokens: float = 0.0
    gpu_then_cpu_penalty_ms: float = 0.0

    @staticmethod
    def from_args(
        *,
        utility_cost_model: str = "",
        tail_lambda: float = 0.0,
        gpu_then_cpu_penalty_ms: float = 0.0,
    ) -> "UtilityPlannerConfig":
        return UtilityPlannerConfig(
            cost_model=CostModel.from_spec(utility_cost_model),
            tail_lambda=float(tail_lambda or 0.0),
            gpu_then_cpu_penalty_ms=float(gpu_then_cpu_penalty_ms or 0.0),
        )


def _layer_score(layer: str) -> float:
    layer = (layer or "unknown").lower()
    if layer == "system":
        layer = "long_term"
    priors = {
        "long_term": 2.0,
        "mid_term": 1.0,
        "short_term": 0.0,
        "unknown": 1.0,
        "question": 0.0,
        "memoryos": 1.0,
        "amem": 1.0,
        "memos": 1.0,
        "skillsbench": 1.0,
    }
    return float(priors.get(layer, 1.0))


def reuse_score_from_hints(
    *,
    layer: str,
    tokens: int,
    hints: dict[str, Any],
    cfg: UtilityPlannerConfig,
) -> float:
    score = 0.0
    score += cfg.w_layer * _layer_score(layer)
    score += cfg.w_hotness * _log1p_safe(hints.get("prior_use_count"))
    score += cfg.w_importance * _float_or(hints.get("importance_score"), 0.0)
    score += cfg.w_graph * _log1p_safe(hints.get("graph_in_degree"))
    if cfg.w_tokens:
        score -= cfg.w_tokens * _log1p_safe(tokens)
    return float(score)


def p_reuse_from_score(score: float) -> float:
    return _sigmoid(score)


def choose_admit_action(
    *,
    layer: str,
    tokens: int,
    hints: dict[str, Any],
    cfg: UtilityPlannerConfig,
    gpu_free_tokens: int = 0,
    writeback_async: bool = False,
) -> tuple[str, float, dict[str, float]]:
    token_count = max(int(tokens or 0), 0)
    if token_count <= 0:
        return "drop", 0.0, {"p_reuse": 0.0}

    score = reuse_score_from_hints(
        layer=layer,
        tokens=token_count,
        hints=hints,
        cfg=cfg,
    )
    p_reuse = p_reuse_from_score(score)

    miss_ms = float(cfg.cost_model.recompute_ms(token_count))
    cpu_ms = float(cfg.cost_model.transfer_ms(tokens_to_bytes(token_count)))
    saved_by_cpu = max(miss_ms - cpu_ms, 0.0)
    saved_by_gpu = max(miss_ms, 0.0)

    benefit_cpu = p_reuse * saved_by_cpu
    benefit_gpu = p_reuse * saved_by_gpu

    writeback_ms = float(cfg.cost_model.transfer_ms(tokens_to_bytes(token_count)))
    cost_cpu = writeback_ms
    if writeback_async:
        async_scale = 0.1 + 0.4 * _clamp(cfg.tail_lambda, 0.0, 1.0)
        cost_cpu = writeback_ms * async_scale

    util_cpu = benefit_cpu - cost_cpu
    util_gpu = float("-inf")
    gpu_free_tokens = max(int(gpu_free_tokens or 0), 0)
    if gpu_free_tokens >= token_count:
        util_gpu = benefit_gpu - float(cfg.gpu_then_cpu_penalty_ms)

    util_drop = 0.0
    best = max(util_drop, util_cpu, util_gpu)
    if best <= 0.0:
        return (
            "drop",
            float(best),
            {
                "score": float(score),
                "p_reuse": float(p_reuse),
                "util_cpu": float(util_cpu),
                "util_gpu": float(util_gpu),
            },
        )

    if util_gpu >= util_cpu:
        return (
            "gpu_then_cpu",
            float(util_gpu),
            {
                "score": float(score),
                "p_reuse": float(p_reuse),
                "util_cpu": float(util_cpu),
                "util_gpu": float(util_gpu),
            },
        )

    return (
        "cpu",
        float(util_cpu),
        {
            "score": float(score),
            "p_reuse": float(p_reuse),
            "util_cpu": float(util_cpu),
            "util_gpu": float(util_gpu),
        },
    )


@dataclass(frozen=True)
class PrefillPlacement:
    chunk_id: str
    action: str
    utility: float
    target_location: str | None


def plan_prefill_placements(
    *,
    unique_chunks: dict[str, dict[str, Any]],
    prefill_order: list[str],
    gpu_budget_tokens: int,
    cpu_budget_tokens: int,
    cfg: UtilityPlannerConfig,
    writeback_async: bool = False,
) -> list[PrefillPlacement]:
    placements: list[PrefillPlacement] = []
    gpu_free_tokens = max(int(gpu_budget_tokens or 0), 0)
    cpu_free_tokens = max(int(cpu_budget_tokens or 0), 0)

    for chunk_id in prefill_order:
        chunk = unique_chunks[chunk_id]
        token_count = int(chunk.get("tokens", 0) or 0)
        layer = str(chunk.get("layer", "unknown"))
        hints = dict(chunk.get("hints") or {})
        action, utility, _debug = choose_admit_action(
            layer=layer,
            tokens=token_count,
            hints=hints,
            cfg=cfg,
            gpu_free_tokens=gpu_free_tokens,
            writeback_async=writeback_async,
        )

        target_location: str | None = None
        if action == "gpu_then_cpu":
            if token_count <= gpu_free_tokens:
                gpu_free_tokens -= token_count
                target_location = "LocalGPUBackend"
            elif token_count <= cpu_free_tokens:
                cpu_free_tokens -= token_count
                target_location = "LocalCPUBackend"
                action = "cpu"
            else:
                action = "drop"
        elif action == "cpu":
            if token_count <= cpu_free_tokens:
                cpu_free_tokens -= token_count
                target_location = "LocalCPUBackend"
            else:
                action = "drop"

        placements.append(
            PrefillPlacement(
                chunk_id=chunk_id,
                action=action,
                utility=float(utility),
                target_location=target_location,
            )
        )

    return placements


def attach_prior_use_count(
    *,
    unique_chunks: dict[str, dict[str, Any]],
    query_chunk_ids: list[list[str]],
) -> None:
    counts: dict[str, int] = {}
    for chunk_ids in query_chunk_ids:
        for chunk_id in chunk_ids:
            counts[chunk_id] = counts.get(chunk_id, 0) + 1

    for chunk_id, chunk in unique_chunks.items():
        hints = dict(chunk.get("hints") or {})
        hints["prior_use_count"] = int(counts.get(chunk_id, 0))
        chunk["hints"] = hints


def _memoryos_importance(chunk_type: str) -> float:
    chunk_type = (chunk_type or "").lower()
    if chunk_type == "user_profile":
        return 1.0
    if chunk_type in {"assistant_knowledge", "knowledge"}:
        return 0.9
    if chunk_type == "retrieved_page":
        return 0.7
    if chunk_type == "page":
        return 0.4
    return 0.5


def enrich_memoryos_hints(unique_chunks: dict[str, dict[str, Any]]) -> None:
    for chunk in unique_chunks.values():
        hints = dict(chunk.get("hints") or {})
        hints.setdefault("importance_score", _memoryos_importance(str(chunk.get("type"))))
        hints.setdefault("graph_in_degree", 0)
        chunk["hints"] = hints


def enrich_id_backed_hints(
    *,
    workload_kind: str,
    unique_chunks: dict[str, dict[str, Any]],
    data_root: Path,
    datasets: list[str],
) -> None:
    memory_records: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        if workload_kind == "amem":
            path = data_root / "amem" / f"amem_{dataset}_memorys.json"
        elif workload_kind == "memos":
            path = data_root / "memos" / f"memos_{dataset}_memorys.json"
        else:
            raise ValueError(f"Unsupported workload_kind={workload_kind!r}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        for item in loaded:
            memory_id = str(item.get("memory_id", "")).strip()
            if memory_id:
                memory_records[memory_id] = item

    incoming_edges: dict[str, int] = {}
    if workload_kind == "amem":
        for record in memory_records.values():
            for target in record.get("links") or []:
                target_id = str(target).strip()
                if target_id:
                    incoming_edges[target_id] = incoming_edges.get(target_id, 0) + 1

    for chunk in unique_chunks.values():
        memory_id = str(chunk.get("memory_id", "")).strip()
        if not memory_id:
            continue
        record = memory_records.get(memory_id)
        if record is None:
            continue

        hints = dict(chunk.get("hints") or {})
        if workload_kind == "amem":
            hints["importance_score"] = _float_or(record.get("importance_score"), 0.5)
            hints["tags"] = list(record.get("tags") or [])
            hints["keywords"] = list(record.get("keywords") or [])
            hints["graph_in_degree"] = int(incoming_edges.get(memory_id, 0))
            hints["link_memory_ids"] = [str(v) for v in (record.get("links") or []) if v]
        else:
            meta = record.get("meta") or {}
            confidence = _float_or(meta.get("confidence"), 0.5)
            use_count = int(meta.get("use_count", 0) or 0)
            status = str(meta.get("status", "")).lower()
            importance = confidence
            if status == "activated":
                importance += 0.15
            if str(meta.get("memory_type", "")).lower() == "usermemory":
                importance += 0.1
            hints["importance_score"] = _clamp(importance, 0.0, 1.0)
            hints["tags"] = list(meta.get("tags") or [])
            hints["graph_in_degree"] = 0
            hints["prior_use_count"] = max(int(hints.get("prior_use_count", 0) or 0), use_count)
        chunk["hints"] = hints


def enrich_skillsbench_hints(unique_chunks: dict[str, dict[str, Any]]) -> None:
    kind_bias = {
        "skill_md": 0.75,
        "reference": 0.65,
        "script": 0.55,
    }
    for chunk in unique_chunks.values():
        hints = dict(chunk.get("hints") or {})
        task_frequency = max(
            int(chunk.get("task_frequency", 0) or hints.get("task_frequency", 0) or 1),
            1,
        )
        chunk_type = str(chunk.get("type", "") or "fragment")
        base_importance = kind_bias.get(chunk_type, 0.6)
        shared_bonus = min(0.25, 0.08 * _log1p_safe(task_frequency))
        hints["importance_score"] = _clamp(base_importance + shared_bonus, 0.0, 1.0)
        hints["graph_in_degree"] = max(task_frequency - 1, 0)
        hints["task_frequency"] = task_frequency
        hints["skill_names"] = list(chunk.get("skill_names") or [])
        hints["source_tasks"] = list(chunk.get("source_tasks") or [])
        chunk["hints"] = hints
