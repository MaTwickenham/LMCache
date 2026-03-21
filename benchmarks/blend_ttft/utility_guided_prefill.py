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


def _centered_importance(x: object) -> float:
    value = _clamp(_float_or(x, 0.5), 0.0, 1.0)
    return float((2.0 * value) - 1.0)


def _centered_hotness(x: object, *, reference_count: float) -> float:
    baseline = math.log1p(max(float(reference_count or 0.0), 0.0))
    return float(_log1p_safe(x) - baseline)


def _kind_score(kind: str) -> float:
    kind = (kind or "").strip().lower()
    priors = {
        "user_profile": 0.95,
        "assistant_knowledge": 0.65,
        "knowledge": 0.35,
        "retrieved_page": 0.2,
        "page": 0.05,
        "usermemory": 1.0,
        "longtermmemory": 0.2,
        "workingmemory": -0.5,
        "skill_md": 0.6,
        "tool_doc": 0.4,
        "example": 0.15,
    }
    return float(priors.get(kind, 0.0))


@dataclass(frozen=True)
class UtilityPlannerConfig:
    cost_model: CostModel
    tail_lambda: float = 0.0
    enable_semantic_hints: bool = True
    w_layer: float = 0.75
    w_hotness: float = 0.45
    w_importance: float = 0.8
    w_kind: float = 0.9
    w_graph: float = 0.2
    w_runtime_access: float = 0.75
    w_runtime_recency: float = 0.35
    w_tokens: float = 0.0
    runtime_recency_window_queries: float = 4.0
    hotness_reference_count: float = 8.0
    reuse_score_bias: float = -2.0
    gpu_then_cpu_penalty_ms: float = 0.0
    cpu_admission_overhead_ms: float = 0.5

    @staticmethod
    def from_args(
        *,
        utility_cost_model: str = "",
        tail_lambda: float = 0.0,
        enable_semantic_hints: bool = True,
        gpu_then_cpu_penalty_ms: float = 0.0,
    ) -> "UtilityPlannerConfig":
        return UtilityPlannerConfig(
            cost_model=CostModel.from_spec(utility_cost_model),
            tail_lambda=float(tail_lambda or 0.0),
            enable_semantic_hints=bool(enable_semantic_hints),
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
    kind: str,
    tokens: int,
    hints: dict[str, Any],
    cfg: UtilityPlannerConfig,
) -> float:
    score = 0.0
    score += cfg.w_layer * _layer_score(layer)
    if cfg.enable_semantic_hints:
        score += cfg.w_hotness * _centered_hotness(
            hints.get("prior_use_count"),
            reference_count=cfg.hotness_reference_count,
        )
        score += cfg.w_importance * _centered_importance(
            hints.get("importance_score")
        )
        score += cfg.w_kind * _kind_score(kind)
        score += cfg.w_graph * _log1p_safe(hints.get("graph_in_degree"))
    if cfg.w_tokens:
        score -= cfg.w_tokens * _log1p_safe(tokens)
    return float(score)


def reuse_score_from_runtime(
    *,
    access_count: int = 0,
    queries_since_access: int | None = None,
    cfg: UtilityPlannerConfig,
) -> float:
    score = cfg.w_runtime_access * _log1p_safe(access_count)
    if queries_since_access is None:
        return float(score)

    distance = max(int(queries_since_access), 0)
    window = max(float(cfg.runtime_recency_window_queries), 1.0)
    recency = 1.0 / (1.0 + (float(distance) / window))
    score += cfg.w_runtime_recency * recency
    return float(score)


def reuse_score(
    *,
    layer: str,
    kind: str,
    tokens: int,
    hints: dict[str, Any],
    cfg: UtilityPlannerConfig,
    access_count: int = 0,
    queries_since_access: int | None = None,
) -> float:
    return float(
        reuse_score_from_hints(
            layer=layer,
            kind=kind,
            tokens=tokens,
            hints=hints,
            cfg=cfg,
        )
        + reuse_score_from_runtime(
            access_count=access_count,
            queries_since_access=queries_since_access,
            cfg=cfg,
        )
        + float(cfg.reuse_score_bias)
    )


def p_reuse_from_score(score: float) -> float:
    return _sigmoid(score)


@dataclass(frozen=True)
class FragmentTierUtility:
    score: float
    p_reuse: float
    keep_cpu: float
    keep_gpu: float
    admit_cpu: float
    admit_gpu: float
    gpu_premium_keep: float
    gpu_premium_admit: float
    writeback_ms: float
    cpu_saved_ms: float
    gpu_saved_ms: float


def estimate_fragment_tier_utilities(
    *,
    layer: str,
    kind: str = "",
    tokens: int,
    hints: dict[str, Any],
    cfg: UtilityPlannerConfig,
    access_count: int = 0,
    queries_since_access: int | None = None,
    writeback_async: bool = False,
    has_cpu_fallback: bool = True,
) -> FragmentTierUtility:
    token_count = max(int(tokens or 0), 0)
    if token_count <= 0:
        return FragmentTierUtility(
            score=0.0,
            p_reuse=0.0,
            keep_cpu=0.0,
            keep_gpu=float("-inf"),
            admit_cpu=0.0,
            admit_gpu=float("-inf"),
            gpu_premium_keep=float("-inf"),
            gpu_premium_admit=float("-inf"),
            writeback_ms=0.0,
            cpu_saved_ms=0.0,
            gpu_saved_ms=0.0,
        )

    score = reuse_score(
        layer=layer,
        kind=kind,
        tokens=token_count,
        hints=hints,
        cfg=cfg,
        access_count=access_count,
        queries_since_access=queries_since_access,
    )
    p_reuse = p_reuse_from_score(score)

    miss_ms = float(cfg.cost_model.recompute_ms(token_count))
    cpu_ms = float(cfg.cost_model.transfer_ms(tokens_to_bytes(token_count)))
    saved_by_cpu = max(miss_ms - cpu_ms, 0.0)
    saved_by_gpu = max(miss_ms, 0.0)

    keep_cpu = p_reuse * saved_by_cpu
    keep_gpu = (p_reuse * saved_by_gpu) - float(cfg.gpu_then_cpu_penalty_ms)

    writeback_ms = float(cfg.cost_model.transfer_ms(tokens_to_bytes(token_count)))
    cost_cpu = writeback_ms
    if writeback_async:
        async_scale = 0.1 + 0.4 * _clamp(cfg.tail_lambda, 0.0, 1.0)
        cost_cpu = writeback_ms * async_scale
    cost_cpu += float(cfg.cpu_admission_overhead_ms)

    admit_cpu = keep_cpu - cost_cpu
    admit_gpu = keep_gpu

    keep_fallback = 0.0
    admit_fallback = 0.0
    if has_cpu_fallback:
        keep_fallback = max(keep_cpu, 0.0)
        admit_fallback = max(admit_cpu, 0.0)

    return FragmentTierUtility(
        score=float(score),
        p_reuse=float(p_reuse),
        keep_cpu=float(keep_cpu),
        keep_gpu=float(keep_gpu),
        admit_cpu=float(admit_cpu),
        admit_gpu=float(admit_gpu),
        gpu_premium_keep=float(keep_gpu - keep_fallback),
        gpu_premium_admit=float(admit_gpu - admit_fallback),
        writeback_ms=float(writeback_ms),
        cpu_saved_ms=float(saved_by_cpu),
        gpu_saved_ms=float(saved_by_gpu),
    )


def choose_admit_action(
    *,
    layer: str,
    kind: str = "",
    tokens: int,
    hints: dict[str, Any],
    cfg: UtilityPlannerConfig,
    gpu_free_tokens: int = 0,
    writeback_async: bool = False,
    access_count: int = 0,
    queries_since_access: int | None = None,
) -> tuple[str, float, dict[str, float]]:
    token_count = max(int(tokens or 0), 0)
    if token_count <= 0:
        return "drop", 0.0, {"p_reuse": 0.0}

    utility = estimate_fragment_tier_utilities(
        layer=layer,
        kind=kind,
        tokens=token_count,
        hints=hints,
        cfg=cfg,
        access_count=access_count,
        queries_since_access=queries_since_access,
        writeback_async=writeback_async,
        has_cpu_fallback=True,
    )
    util_cpu = utility.admit_cpu
    util_gpu = float("-inf")
    gpu_free_tokens = max(int(gpu_free_tokens or 0), 0)
    if gpu_free_tokens >= token_count:
        util_gpu = utility.admit_gpu

    util_drop = 0.0
    best = max(util_drop, util_cpu, util_gpu)
    if best <= 0.0:
        return (
            "drop",
            float(best),
            {
                "score": float(utility.score),
                "p_reuse": float(utility.p_reuse),
                "util_cpu": float(util_cpu),
                "util_gpu": float(util_gpu),
                "gpu_premium_admit": float(utility.gpu_premium_admit),
            },
        )

    if util_gpu >= util_cpu:
        return (
            "gpu_then_cpu",
            float(util_gpu),
            {
                "score": float(utility.score),
                "p_reuse": float(utility.p_reuse),
                "util_cpu": float(util_cpu),
                "util_gpu": float(util_gpu),
                "gpu_premium_admit": float(utility.gpu_premium_admit),
            },
        )

    return (
        "cpu",
        float(util_cpu),
        {
            "score": float(utility.score),
            "p_reuse": float(utility.p_reuse),
            "util_cpu": float(util_cpu),
            "util_gpu": float(util_gpu),
            "gpu_premium_admit": float(utility.gpu_premium_admit),
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
        kind = str(chunk.get("type") or chunk.get("memory_type") or "")
        hints = dict(chunk.get("hints") or {})
        action, utility, _debug = choose_admit_action(
            layer=layer,
            kind=kind,
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
        return 0.9 if chunk_type == "assistant_knowledge" else 0.78
    if chunk_type == "retrieved_page":
        return 0.55
    if chunk_type == "page":
        return 0.25
    return 0.5


def _memos_importance(meta: dict[str, Any]) -> float:
    memory_type = str(meta.get("memory_type", "")).strip().lower()
    confidence = _clamp(_float_or(meta.get("confidence"), 0.5), 0.0, 1.0)
    status = str(meta.get("status", "")).strip().lower()

    type_prior = {
        "usermemory": 0.9,
        "longtermmemory": 0.65,
        "workingmemory": 0.35,
    }.get(memory_type, 0.5)
    status_bonus = 0.05 if status == "activated" else 0.0
    importance = (0.8 * float(type_prior)) + (0.15 * float(confidence)) + status_bonus
    return _clamp(importance, 0.0, 1.0)


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
            use_count = int(meta.get("use_count", 0) or 0)
            hints["importance_score"] = _memos_importance(meta)
            hints["tags"] = list(meta.get("tags") or [])
            hints["graph_in_degree"] = 0
            hints["prior_use_count"] = max(int(hints.get("prior_use_count", 0) or 0), use_count)
        chunk["hints"] = hints


def enrich_skillsbench_hints(unique_chunks: dict[str, dict[str, Any]]) -> None:
    kind_bias = {
        "skill_md": 0.56,
        "reference": 0.42,
        "script": 0.28,
    }
    size_penalty_scale = {
        "skill_md": 0.05,
        "reference": 0.09,
        "script": 0.08,
    }
    for chunk in unique_chunks.values():
        hints = dict(chunk.get("hints") or {})
        task_frequency = max(
            int(chunk.get("task_frequency", 0) or hints.get("task_frequency", 0) or 1),
            1,
        )
        prior_use_count = max(int(hints.get("prior_use_count", 0) or 0), 0)
        tokens = max(int(chunk.get("tokens", 0) or 0), 0)
        chunk_type = str(chunk.get("type", "") or "fragment")
        base_importance = kind_bias.get(chunk_type, 0.6)
        shared_bonus = min(0.14, 0.05 * _log1p_safe(task_frequency))
        trace_bonus = min(0.16, 0.07 * _log1p_safe(prior_use_count))
        size_scale = max(float(tokens) / 768.0, 1.0)
        size_penalty = min(
            0.28,
            float(size_penalty_scale.get(chunk_type, 0.06))
            * _log1p_safe(size_scale),
        )
        importance = base_importance + shared_bonus + trace_bonus - size_penalty
        hints["importance_score"] = _clamp(importance, 0.0, 1.0)
        hints["graph_in_degree"] = max(min(task_frequency - 1, 4), 0)
        hints["task_frequency"] = task_frequency
        hints["skill_names"] = list(chunk.get("skill_names") or [])
        hints["source_tasks"] = list(chunk.get("source_tasks") or [])
        chunk["hints"] = hints
