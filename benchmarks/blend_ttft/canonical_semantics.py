from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utility_guided_prefill import (
    enrich_id_backed_hints,
    enrich_memoryos_hints,
    enrich_skillsbench_hints,
)


CANONICAL_SEMANTICS_VERSION = 1
SKILLS_WORKLOAD_KINDS = frozenset({"skillsbench", "swe_skillsbench"})
UNSAFE_TRACE_HINT_KEYS = frozenset(
    {
        "query_positions",
        "first_query_idx",
        "last_query_idx",
        "trace_total_use_count",
        "remaining_use_count",
        "task_frequency",
        "group_frequency",
    }
)
UNSAFE_RUNTIME_REUSE_HINT_KEYS = frozenset(
    {
        "estimated_future_use_count",
        "estimated_next_use_distance_queries",
        "online_estimated_future_use_count",
        "online_estimated_next_use_distance_queries",
    }
)
UNSAFE_MEMGAS_TRACE_HINT_PREFIXES = (
    "memgas_avg_",
    "memgas_top",
    "memgas_ranked_appear_count",
    "memgas_graph_support_rate",
    "memgas_seed_support_rate",
    "memgas_dominant_",
)


def canonical_workload_family(workload_kind: str) -> str:
    if workload_kind in SKILLS_WORKLOAD_KINDS:
        return "skillsbench"
    return str(workload_kind)


def is_skills_workload(workload_kind: str) -> bool:
    return canonical_workload_family(workload_kind) == "skillsbench"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _float_or(x: object, default: float = 0.0) -> float:
    try:
        value = float(x)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    if math.isnan(value) or math.isinf(value):
        return float(default)
    return float(value)


def _log1p_safe(x: object) -> float:
    value = _float_or(x, 0.0)
    if value <= 0.0:
        return 0.0
    return float(math.log1p(value))


def _normalized_log_count(x: object, *, reference: float) -> float:
    baseline = math.log1p(max(float(reference), 1.0))
    return _clamp(_log1p_safe(x) / baseline, 0.0, 1.0)


@dataclass(frozen=True)
class CanonicalSemantics:
    version: int = CANONICAL_SEMANTICS_VERSION
    workload_kind: str = ""
    workload_family: str = ""
    layer: str = "unknown"
    kind: str = ""
    importance_score: float = 0.5
    stability_score: float = 0.5
    sharedness_score: float = 0.0
    relation_degree: float = 0.0
    retention_prior: float = 0.5
    affinity_score: float = 0.0
    future_use_reference_count: float = 1.0
    next_use_window_queries: float = 3.0
    semantic_groups: tuple[str, ...] = field(default_factory=tuple)
    raw_evidence: dict[str, Any] = field(default_factory=dict)

    def to_hint_dict(
        self,
        *,
        base_hints: dict[str, Any],
    ) -> dict[str, Any]:
        hints = dict(base_hints)
        hints["canonical_version"] = int(self.version)
        hints["canonical_workload_kind"] = str(self.workload_kind)
        hints["canonical_workload_family"] = str(self.workload_family)
        hints["importance_score"] = float(self.importance_score)
        hints["stability_score"] = float(self.stability_score)
        hints["sharedness_score"] = float(self.sharedness_score)
        hints["relation_degree"] = float(self.relation_degree)
        hints["retention_prior"] = float(self.retention_prior)
        hints["affinity_score"] = float(self.affinity_score)
        hints["future_use_reference_count"] = float(self.future_use_reference_count)
        hints["next_use_window_queries"] = float(self.next_use_window_queries)
        hints["graph_in_degree"] = int(round(float(self.relation_degree) * 4.0))
        if self.semantic_groups:
            hints["semantic_groups"] = list(self.semantic_groups)
        return hints


def apply_canonical_semantics(
    *,
    workload_kind: str,
    unique_chunks: dict[str, dict[str, Any]],
    query_chunk_ids: list[list[str]],
    data_root: Path,
    datasets: list[str],
) -> None:
    family = canonical_workload_family(workload_kind)
    if family == "memoryos":
        enrich_memoryos_hints(unique_chunks)
    elif family == "skillsbench":
        enrich_skillsbench_hints(unique_chunks)
    else:
        enrich_id_backed_hints(
            workload_kind=family,
            unique_chunks=unique_chunks,
            data_root=data_root,
            datasets=datasets,
        )

    for chunk in unique_chunks.values():
        base_hints = _sanitize_base_hints(dict(chunk.get("hints") or {}))
        semantics = _canonicalize_chunk(
            workload_kind=workload_kind,
            chunk=chunk,
            hints=base_hints,
        )
        chunk["canonical_semantics"] = asdict(semantics)
        chunk["semantic_evidence_raw"] = dict(semantics.raw_evidence)
        chunk["hints"] = semantics.to_hint_dict(
            base_hints=base_hints,
        )


def _normalize_query_positions(raw_positions: object) -> tuple[int, ...]:
    if not isinstance(raw_positions, (list, tuple)):
        return ()
    values: list[int] = []
    for item in raw_positions:
        try:
            value = int(item)  # type: ignore[arg-type]
        except Exception:
            continue
        if value >= 0:
            values.append(value)
    values.sort()
    return tuple(values)


def _sanitize_base_hints(base_hints: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for raw_key, value in base_hints.items():
        key = str(raw_key)
        if key in UNSAFE_TRACE_HINT_KEYS:
            continue
        if any(key.startswith(prefix) for prefix in UNSAFE_MEMGAS_TRACE_HINT_PREFIXES):
            continue
        sanitized[key] = value
    return sanitized


def strip_past_only_runtime_hints(
    *,
    unique_chunks: dict[str, dict[str, Any]],
) -> None:
    for chunk in unique_chunks.values():
        hints = dict(chunk.get("hints") or {})
        family = canonical_workload_family(
            str(
                hints.get("canonical_workload_family")
                or hints.get("canonical_workload_kind")
                or hints.get("workload_kind")
                or ""
            ).strip()
        )
        for key in UNSAFE_TRACE_HINT_KEYS:
            hints.pop(key, None)
        for key in UNSAFE_RUNTIME_REUSE_HINT_KEYS:
            hints.pop(key, None)
        if family == "skillsbench":
            hints.pop("prior_use_count", None)
            hints.pop("static_prior_use_count", None)
        chunk["hints"] = hints


def _canonicalize_chunk(
    *,
    workload_kind: str,
    chunk: dict[str, Any],
    hints: dict[str, Any],
) -> CanonicalSemantics:
    family = canonical_workload_family(workload_kind)
    if family == "memoryos":
        return _canonicalize_memoryos_chunk(
            workload_kind=workload_kind,
            chunk=chunk,
            hints=hints,
        )
    if family == "skillsbench":
        return _canonicalize_skillsbench_chunk(
            workload_kind=workload_kind,
            chunk=chunk,
            hints=hints,
        )
    if family == "memos":
        return _canonicalize_memos_chunk(
            workload_kind=workload_kind,
            chunk=chunk,
            hints=hints,
        )
    if family == "amem":
        return _canonicalize_amem_chunk(
            workload_kind=workload_kind,
            chunk=chunk,
            hints=hints,
        )
    if family == "dspy_locomo":
        return _canonicalize_dspy_locomo_chunk(
            workload_kind=workload_kind,
            chunk=chunk,
            hints=hints,
        )
    if family == "memgas":
        return _canonicalize_memgas_chunk(
            workload_kind=workload_kind,
            chunk=chunk,
            hints=hints,
        )
    return _canonicalize_generic_chunk(
        workload_kind=workload_kind,
        chunk=chunk,
        hints=hints,
    )


def _make_semantics(
    *,
    workload_kind: str,
    workload_family: str,
    layer: str,
    kind: str,
    importance_score: float,
    stability_score: float,
    sharedness_score: float,
    relation_degree: float,
    future_use_reference_count: float,
    next_use_window_queries: float,
    semantic_groups: list[str] | tuple[str, ...] | None = None,
    raw_evidence: dict[str, Any] | None = None,
) -> CanonicalSemantics:
    importance_score = _clamp(importance_score, 0.0, 1.0)
    stability_score = _clamp(stability_score, 0.0, 1.0)
    sharedness_score = _clamp(sharedness_score, 0.0, 1.0)
    relation_degree = _clamp(relation_degree, 0.0, 1.0)
    retention_prior = _clamp(
        (0.50 * importance_score)
        + (0.30 * stability_score)
        + (0.20 * sharedness_score),
        0.0,
        1.0,
    )
    affinity_score = _clamp(
        (0.65 * sharedness_score) + (0.35 * relation_degree),
        0.0,
        1.0,
    )
    return CanonicalSemantics(
        workload_kind=str(workload_kind),
        workload_family=str(workload_family),
        layer=str(layer or "unknown"),
        kind=str(kind or ""),
        importance_score=float(importance_score),
        stability_score=float(stability_score),
        sharedness_score=float(sharedness_score),
        relation_degree=float(relation_degree),
        retention_prior=float(retention_prior),
        affinity_score=float(affinity_score),
        future_use_reference_count=max(float(future_use_reference_count), 1.0),
        next_use_window_queries=max(float(next_use_window_queries), 1.0),
        semantic_groups=tuple(str(item) for item in (semantic_groups or ()) if str(item)),
        raw_evidence=dict(raw_evidence or {}),
    )


def _canonicalize_memoryos_chunk(
    *,
    workload_kind: str,
    chunk: dict[str, Any],
    hints: dict[str, Any],
) -> CanonicalSemantics:
    kind = str(chunk.get("type") or "").strip().lower()
    importance = _float_or(hints.get("importance_score"), 0.5)
    stability = {
        "user_profile": 1.0,
        "assistant_knowledge": 0.82,
        "knowledge": 0.72,
        "retrieved_page": 0.38,
        "page": 0.18,
    }.get(kind, 0.45)
    sharedness = {
        "user_profile": 0.90,
        "assistant_knowledge": 0.72,
        "knowledge": 0.58,
        "retrieved_page": 0.24,
        "page": 0.10,
    }.get(kind, 0.20)
    return _make_semantics(
        workload_kind=workload_kind,
        workload_family="memoryos",
        layer=str(chunk.get("layer") or "memoryos"),
        kind=str(chunk.get("type") or ""),
        importance_score=importance,
        stability_score=stability,
        sharedness_score=sharedness,
        relation_degree=0.0,
        future_use_reference_count=_float_or(
            hints.get("future_use_reference_count"),
            1.0,
        ),
        next_use_window_queries=_float_or(
            hints.get("next_use_window_queries"),
            3.0,
        ),
        raw_evidence={
            "memory_type": str(chunk.get("type") or ""),
            "conversation_id": str(chunk.get("conversation_id") or ""),
        },
    )


def _canonicalize_memos_chunk(
    *,
    workload_kind: str,
    chunk: dict[str, Any],
    hints: dict[str, Any],
) -> CanonicalSemantics:
    kind = str(chunk.get("type") or "").strip().lower()
    use_count = max(int(hints.get("prior_use_count", 0) or 0), 0)
    importance = _float_or(hints.get("importance_score"), 0.5)
    stability = {
        "usermemory": 0.92,
        "longtermmemory": 0.80,
        "workingmemory": 0.35,
    }.get(kind, 0.50)
    sharedness = _clamp(
        {
            "usermemory": 0.70,
            "longtermmemory": 0.54,
            "workingmemory": 0.22,
        }.get(kind, 0.30)
        + (0.18 * _normalized_log_count(use_count, reference=64.0)),
        0.0,
        1.0,
    )
    return _make_semantics(
        workload_kind=workload_kind,
        workload_family="memos",
        layer=str(chunk.get("layer") or "memos"),
        kind=str(chunk.get("type") or ""),
        importance_score=importance,
        stability_score=stability,
        sharedness_score=sharedness,
        relation_degree=0.0,
        future_use_reference_count=_float_or(
            hints.get("future_use_reference_count"),
            1.0,
        ),
        next_use_window_queries=_float_or(
            hints.get("next_use_window_queries"),
            4.0,
        ),
        raw_evidence={
            "memory_type": str(chunk.get("type") or ""),
            "prior_use_count": use_count,
            "memory_id": str(chunk.get("memory_id") or ""),
        },
    )


def _canonicalize_amem_chunk(
    *,
    workload_kind: str,
    chunk: dict[str, Any],
    hints: dict[str, Any],
) -> CanonicalSemantics:
    keywords = [str(item) for item in (hints.get("keywords") or []) if str(item)]
    relation_degree = _clamp(
        int(hints.get("graph_in_degree", 0) or 0) / 4.0,
        0.0,
        1.0,
    )
    importance = _clamp(
        0.42 + min(0.18, 0.03 * len(keywords)),
        0.0,
        1.0,
    )
    return _make_semantics(
        workload_kind=workload_kind,
        workload_family="amem",
        layer=str(chunk.get("layer") or "amem"),
        kind=str(chunk.get("type") or ""),
        importance_score=importance,
        stability_score=0.24,
        sharedness_score=0.12,
        relation_degree=relation_degree,
        future_use_reference_count=0.75,
        next_use_window_queries=2.5,
        raw_evidence={
            "keywords": list(keywords),
            "relation_targets": list(hints.get("link_memory_ids") or []),
            "memory_id": str(chunk.get("memory_id") or ""),
        },
    )


def _canonicalize_dspy_locomo_chunk(
    *,
    workload_kind: str,
    chunk: dict[str, Any],
    hints: dict[str, Any],
) -> CanonicalSemantics:
    kind = str(chunk.get("type") or "").strip().lower()
    hint_role = str(hints.get("hint_role") or "").strip().lower()
    sharedness = {
        "summary_anchor": 0.72,
        "temporal_bridge": 0.55,
        "event_support": 0.35,
    }.get(hint_role, 0.40 if kind == "session_anchor" else 0.28)
    stability = 0.85 if kind == "session_anchor" else 0.42
    return _make_semantics(
        workload_kind=workload_kind,
        workload_family="dspy_locomo",
        layer=str(chunk.get("layer") or "dspy_locomo"),
        kind=str(chunk.get("type") or ""),
        importance_score=_float_or(hints.get("importance_score"), 0.5),
        stability_score=stability,
        sharedness_score=sharedness,
        relation_degree=0.0,
        future_use_reference_count=_float_or(
            hints.get("future_use_reference_count"),
            1.0,
        ),
        next_use_window_queries=_float_or(
            hints.get("next_use_window_queries"),
            4.5,
        ),
        raw_evidence={
            "hint_role": hint_role,
            "source_session_id": hints.get("source_session_id"),
            "temporal_index": hints.get("temporal_index"),
            "speaker_set": list(hints.get("speaker_set") or []),
            "source_turn_ids": list(hints.get("source_turn_ids") or []),
        },
    )


def _canonicalize_memgas_chunk(
    *,
    workload_kind: str,
    chunk: dict[str, Any],
    hints: dict[str, Any],
) -> CanonicalSemantics:
    keywords = [str(item) for item in (hints.get("keywords") or []) if str(item)]
    turn_count_signal = _normalized_log_count(
        hints.get("memgas_turn_count"),
        reference=48.0,
    )
    keyword_signal = _clamp(float(len(keywords)) / 8.0, 0.0, 1.0)
    speaker_signal = _clamp(
        float(len(hints.get("speaker_set") or [])) / 3.0,
        0.0,
        1.0,
    )
    summary_signal = 1.0 if str(hints.get("summary") or "").strip() else 0.0
    structural_stability = 0.0
    if bool(hints.get("memgas_has_source_session_summary")):
        structural_stability += 0.08
    if bool(hints.get("memgas_has_generated_summary")):
        structural_stability += 0.08
    if bool(hints.get("memgas_has_event_summary")):
        structural_stability += 0.06
    if bool(hints.get("memgas_has_observation")):
        structural_stability += 0.04
    if bool(hints.get("memgas_has_image_caption")):
        structural_stability += 0.02
    importance = _clamp(
        (0.58 * _float_or(hints.get("importance_score"), 0.5))
        + (0.18 * turn_count_signal)
        + (0.14 * keyword_signal)
        + (0.10 * summary_signal),
        0.0,
        1.0,
    )
    stability = _clamp(
        0.54 + (0.16 * turn_count_signal) + structural_stability,
        0.0,
        1.0,
    )
    sharedness = _clamp(
        0.18 + (0.18 * keyword_signal) + (0.14 * speaker_signal) + (0.12 * summary_signal),
        0.0,
        1.0,
    )
    relation_degree = _clamp(
        _float_or(hints.get("relation_degree"), 0.0),
        0.0,
        1.0,
    )
    semantic_groups = [str(item) for item in (hints.get("semantic_groups") or []) if str(item)]
    return _make_semantics(
        workload_kind=workload_kind,
        workload_family="memgas",
        layer=str(chunk.get("layer") or "memgas"),
        kind=str(chunk.get("type") or ""),
        importance_score=importance,
        stability_score=stability,
        sharedness_score=_clamp(
            sharedness + min(0.08, 0.015 * len(keywords)),
            0.0,
            1.0,
        ),
        relation_degree=relation_degree,
        future_use_reference_count=_float_or(
            hints.get("future_use_reference_count"),
            1.0,
        ),
        next_use_window_queries=_float_or(
            hints.get("next_use_window_queries"),
            4.0,
        ),
        semantic_groups=semantic_groups,
        raw_evidence={
            "keywords": list(keywords),
            "summary": str(hints.get("summary") or ""),
            "memory_id": str(chunk.get("memory_id") or ""),
            "turn_count": int(hints.get("memgas_turn_count", 0) or 0),
            "speaker_set": list(hints.get("speaker_set") or []),
            "conversation_id": str(hints.get("conversation_id") or ""),
            "session_index": int(hints.get("session_index", 0) or 0),
        },
    )


def _canonicalize_skillsbench_chunk(
    *,
    workload_kind: str,
    chunk: dict[str, Any],
    hints: dict[str, Any],
) -> CanonicalSemantics:
    kind = str(chunk.get("type") or "").strip().lower()
    role = str(hints.get("attachment_role") or "").strip().lower()
    semantic_groups = list(hints.get("semantic_groups") or [])
    skill_group_count = sum(
        1 for group in semantic_groups if str(group).startswith("skill::")
    )
    base_sharedness = {
        "primary": 0.12,
        "core_support": 0.68,
        "secondary_support": 0.46,
    }.get(role, 0.22)
    base_sharedness += {
        "skill_md": 0.10,
        "reference": 0.08,
        "asset": 0.04,
        "script": -0.02,
        "task_brief": -0.10,
    }.get(kind, 0.0)
    if skill_group_count >= 2:
        base_sharedness += 0.08
    elif skill_group_count == 1 and role != "primary":
        base_sharedness += 0.04
    relation_degree = {
        "primary": 0.06,
        "core_support": 0.64,
        "secondary_support": 0.36,
    }.get(role, 0.16)
    if skill_group_count >= 2:
        relation_degree += 0.08
    stability_score = {
        "skill_md": 0.88,
        "reference": 0.66,
        "asset": 0.56,
        "script": 0.18,
    }.get(kind, 0.50)
    return _make_semantics(
        workload_kind=workload_kind,
        workload_family="skillsbench",
        layer=str(chunk.get("layer") or "skillsbench"),
        kind=str(chunk.get("type") or ""),
        importance_score=_float_or(hints.get("importance_score"), 0.5),
        stability_score=stability_score,
        sharedness_score=_clamp(base_sharedness, 0.0, 1.0),
        relation_degree=relation_degree,
        future_use_reference_count=_float_or(
            hints.get("future_use_reference_count"),
            1.25,
        ),
        next_use_window_queries=_float_or(
            hints.get("next_use_window_queries"),
            4.0,
        ),
        semantic_groups=semantic_groups,
        raw_evidence={
            "attachment_role": role,
            "skill_names": list(hints.get("skill_names") or []),
            "source_tasks": list(hints.get("source_tasks") or []),
            "semantic_groups": semantic_groups,
        },
    )


def _canonicalize_generic_chunk(
    *,
    workload_kind: str,
    chunk: dict[str, Any],
    hints: dict[str, Any],
) -> CanonicalSemantics:
    return _make_semantics(
        workload_kind=workload_kind,
        workload_family=canonical_workload_family(workload_kind),
        layer=str(chunk.get("layer") or "unknown"),
        kind=str(chunk.get("type") or ""),
        importance_score=_float_or(hints.get("importance_score"), 0.5),
        stability_score=0.5,
        sharedness_score=0.25,
        relation_degree=_clamp(
            _float_or(hints.get("relation_degree"), 0.0),
            0.0,
            1.0,
        ),
        future_use_reference_count=_float_or(
            hints.get("future_use_reference_count"),
            1.0,
        ),
        next_use_window_queries=_float_or(
            hints.get("next_use_window_queries"),
            3.0,
        ),
        raw_evidence={},
    )
