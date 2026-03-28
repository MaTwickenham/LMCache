from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Literal

from utility_guided_prefill import CostModel, tokens_to_bytes


FetchAction = Literal["reuse_gpu", "load_cpu", "recompute"]
ExecutionMode = Literal["cacheblend", "native_prefix"]

HYBRID_GATE_BASE_MARGIN_MS = 0.0
HYBRID_GATE_CPU_LOAD_MARGIN_MS = 0.0
HYBRID_GATE_MISS_MARGIN_MS = 0.0
HYBRID_GATE_FRAGMENT_REF_TOKENS = 64.0
HYBRID_GATE_FRAGMENT_SCALE_MIN = 0.8
HYBRID_GATE_FRAGMENT_SCALE_MAX = 2.0
HYBRID_GATE_SMALL_CACHED_TOKEN_THRESHOLD = 768.0
HYBRID_GATE_SMALL_CACHED_MARGIN_MS = 10.0
SKILLS_MISS_BLEND_MIN_WEIGHTED_TOKENS = 384.0
SKILLS_MISS_BLEND_MAX_BONUS_MS = 900.0
MEMGAS_MISS_BLEND_MIN_WEIGHTED_TOKENS = 1024.0
MEMGAS_MISS_BLEND_MAX_BONUS_MS = 420.0
FRAGMENTED_MISS_FASTPATH_MIN_MISS_TOKENS = 1024.0
FRAGMENTED_MISS_FASTPATH_MIN_MISS_RATIO = 0.82
FRAGMENTED_MISS_FASTPATH_MIN_FRAGMENTS = 3
FRAGMENTED_MISS_FASTPATH_MIN_AVG_TOKENS = 512.0
FRAGMENTED_MISS_FASTPATH_MAX_AVG_TOKENS = 1536.0
FRAGMENTED_MISS_FASTPATH_MAX_FRAGMENTS = 4
FRAGMENTED_MISS_FASTPATH_MAX_BONUS_MS = 90.0
BLEND_QUERY_BASE_OVERHEAD_MS = 4.0
BLEND_FRAGMENT_OVERHEAD_MS = 2.0
BLEND_MISS_FRAGMENT_OVERHEAD_MS = 2.0
BLEND_CPU_LOAD_FRAGMENT_OVERHEAD_MS = 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _canonical_workload_family(fragment: dict[str, Any]) -> str:
    hints = dict(fragment.get("hints") or {})
    return str(
        hints.get("canonical_workload_family")
        or hints.get("canonical_workload_kind")
        or hints.get("workload_family")
        or ""
    ).strip().lower()


def _normalize_query_positions(raw_positions: object) -> tuple[int, ...]:
    if not isinstance(raw_positions, (list, tuple)):
        return ()
    positions: list[int] = []
    for item in raw_positions:
        try:
            value = int(item)  # type: ignore[arg-type]
        except Exception:
            continue
        if value >= 0:
            positions.append(value)
    if not positions:
        return ()
    positions.sort()
    return tuple(positions)


def _future_reuse_features(
    hints: dict[str, Any],
    *,
    current_query_idx: int | None,
) -> tuple[int, int | None]:
    del current_query_idx
    estimated_count = hints.get("estimated_future_use_count")
    if estimated_count is None:
        estimated_count = hints.get("online_estimated_future_use_count")

    estimated_distance = hints.get("estimated_next_use_distance_queries")
    if estimated_distance is None:
        estimated_distance = hints.get("online_estimated_next_use_distance_queries")

    remaining = max(int(estimated_count or hints.get("prior_use_count", 0) or 0), 0)
    if estimated_distance is None:
        return remaining, None
    try:
        distance = max(int(estimated_distance), 0)
    except Exception:
        distance = None
    return remaining, distance


def _semantic_groups_from_hints(hints: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item).strip().lower()
        for item in (hints.get("semantic_groups") or [])
        if str(item).strip()
    )


def _semantic_pattern_traits(
    *,
    fragment: dict[str, Any],
    hints: dict[str, Any],
) -> dict[str, Any]:
    kind = str(fragment.get("type") or "").strip().lower()
    role = str(hints.get("attachment_role", "") or "").strip().lower()
    tokens = max(int(fragment.get("tokens", 0) or 0), 0)
    retention = _clamp(float(hints.get("retention_prior", 0.5) or 0.5), 0.0, 1.0)
    stability = _clamp(float(hints.get("stability_score", 0.5) or 0.5), 0.0, 1.0)
    sharedness = _clamp(float(hints.get("sharedness_score", 0.0) or 0.0), 0.0, 1.0)
    importance = _clamp(float(hints.get("importance_score", 0.5) or 0.5), 0.0, 1.0)
    groups = _semantic_groups_from_hints(hints)
    summary_backed = ("summary_backed" in groups) or bool(
        str(hints.get("summary") or "").strip()
    )
    support_artifact = role in {"core_support", "secondary_support"} or (
        sharedness >= 0.68 and kind in {"skill_md", "reference", "asset", "script"}
    )
    coherent_context = summary_backed or kind in {
        "session_anchor",
        "dialogue_window",
        "assistant_knowledge",
        "knowledge",
        "user_profile",
    }
    cold_large_primary = (
        role == "primary"
        and tokens >= 1536
        and sharedness < 0.45
        and retention < 0.58
        and stability < 0.65
    )
    return {
        "kind": kind,
        "role": role,
        "tokens": tokens,
        "retention": retention,
        "stability": stability,
        "sharedness": sharedness,
        "importance": importance,
        "summary_backed": summary_backed,
        "support_artifact": support_artifact,
        "coherent_context": coherent_context,
        "cold_large_primary": cold_large_primary,
    }


def _skills_bootstrap_bonus_ms(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    locations: list[str],
    current_query_idx: int | None,
) -> tuple[float, dict[str, float]]:
    bonus_ms = 0.0
    shared_missing_fragments = 0
    shared_missing_tokens = 0
    reused_missing_fragments = 0

    for fragment_id, location in zip(fragment_ids, locations, strict=True):
        if str(location or "miss").lower() != "miss":
            continue

        fragment = fragments.get(fragment_id) or {}
        if _canonical_workload_family(fragment) != "skillsbench":
            continue

        hints = dict(fragment.get("hints") or {})
        remaining_use_count, next_use_distance = _future_reuse_features(
            hints,
            current_query_idx=current_query_idx,
        )
        seen_count = max(int(hints.get("online_seen_count", 0) or 0), 0)
        group_seen_count = max(int(hints.get("online_group_seen_count", 0) or 0), 0)
        traits = _semantic_pattern_traits(fragment=fragment, hints=hints)
        if remaining_use_count <= 0 and seen_count <= 0 and group_seen_count <= 1:
            continue

        role = str(traits["role"])
        tokens = int(traits["tokens"])

        if role == "core_support":
            role_weight = 1.00
            shared_missing_fragments += 1
            shared_missing_tokens += tokens
        elif role == "secondary_support":
            role_weight = 0.38
        elif role == "primary":
            if group_seen_count <= 1 and float(traits["sharedness"]) < 0.55:
                continue
            role_weight = 0.16
        else:
            role_weight = 0.20

        if next_use_distance is None and group_seen_count >= 2:
            proximity = 0.72
        elif next_use_distance is None:
            proximity = 0.45
        elif next_use_distance <= 1:
            proximity = 1.00
        elif next_use_distance <= 4:
            proximity = 0.80
        elif next_use_distance <= 8:
            proximity = 0.50
        else:
            proximity = 0.20

        reuse_strength = _clamp(
            float(max(remaining_use_count, seen_count, max(group_seen_count - 1, 0)))
            / 4.0,
            0.20,
            1.00,
        )
        sharing_strength = _clamp(
            float(max(group_seen_count, 1 + int(round(float(traits["sharedness"]) * 4.0))))
            / 6.0,
            0.15,
            1.00,
        )
        size_strength = _clamp(float(tokens) / 1536.0, 0.35, 1.00)

        fragment_bonus = role_weight * (
            20.0
            + (55.0 * reuse_strength)
            + (35.0 * proximity)
            + (25.0 * sharing_strength)
        ) * size_strength

        reused_missing_fragments += 1
        bonus_ms += float(fragment_bonus)

    if shared_missing_fragments <= 0:
        return 0.0, {
            "skills_bootstrap_bonus_ms": 0.0,
            "skills_bootstrap_shared_missing_fragments": 0.0,
            "skills_bootstrap_shared_missing_tokens": 0.0,
            "skills_bootstrap_reused_missing_fragments": float(reused_missing_fragments),
            "skills_bootstrap_applied": 0.0,
        }

    if shared_missing_fragments < 2 and shared_missing_tokens < 1024:
        return 0.0, {
            "skills_bootstrap_bonus_ms": 0.0,
            "skills_bootstrap_shared_missing_fragments": float(shared_missing_fragments),
            "skills_bootstrap_shared_missing_tokens": float(shared_missing_tokens),
            "skills_bootstrap_reused_missing_fragments": float(reused_missing_fragments),
            "skills_bootstrap_applied": 0.0,
        }

    bonus_ms = min(float(bonus_ms), 220.0)
    return float(bonus_ms), {
        "skills_bootstrap_bonus_ms": float(bonus_ms),
        "skills_bootstrap_shared_missing_fragments": float(shared_missing_fragments),
        "skills_bootstrap_shared_missing_tokens": float(shared_missing_tokens),
        "skills_bootstrap_reused_missing_fragments": float(reused_missing_fragments),
        "skills_bootstrap_applied": 1.0,
    }


def _skills_miss_blend_bonus_ms(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    locations: list[str],
    cfg: "UtilityPlannerConfig",
    current_query_idx: int | None,
) -> tuple[float, dict[str, float]]:
    weighted_tokens = 0.0
    candidate_fragments = 0
    shared_candidates = 0
    trace_candidates = 0
    large_candidates = 0
    candidate_tokens = 0
    cached_hit_fragments = 0
    cached_hit_tokens = 0

    kind_weight = {
        "skill_md": 1.00,
        "reference": 0.92,
        "script": 0.82,
        "asset": 0.70,
        "task_brief": 0.0,
    }

    for fragment_id, location in zip(fragment_ids, locations, strict=True):
        if str(location or "miss").lower() != "miss":
            fragment = fragments.get(fragment_id) or {}
            if _canonical_workload_family(fragment) == "skillsbench":
                tokens = max(int(fragment.get("tokens", 0) or 0), 0)
                if tokens > 0:
                    cached_hit_fragments += 1
                    cached_hit_tokens += tokens
            continue

        fragment = fragments.get(fragment_id) or {}
        if _canonical_workload_family(fragment) != "skillsbench":
            continue

        tokens = max(int(fragment.get("tokens", 0) or 0), 0)
        if tokens <= 0:
            continue

        hints = dict(fragment.get("hints") or {})
        kind = str(fragment.get("type") or "").strip().lower()
        kind_strength = float(kind_weight.get(kind, 0.55))
        if kind_strength <= 0.0:
            continue

        seen_count = max(int(hints.get("online_seen_count", 0) or 0), 0)
        group_seen_count = max(int(hints.get("online_group_seen_count", 0) or 0), 0)
        remaining_use_count, next_use_distance = _future_reuse_features(
            hints,
            current_query_idx=current_query_idx,
        )
        traits = _semantic_pattern_traits(fragment=fragment, hints=hints)
        role = str(traits["role"])

        role_strength = 0.0
        is_large_candidate = False
        if role == "core_support":
            role_strength = 1.00
            shared_candidates += 1
        elif role == "secondary_support":
            role_strength = 0.82
            if group_seen_count >= 2 or float(traits["sharedness"]) >= 0.55:
                shared_candidates += 1
        elif role == "primary":
            if remaining_use_count > 0 or seen_count > 0:
                role_strength = 0.80
                trace_candidates += 1
            elif group_seen_count >= 2 and float(traits["sharedness"]) >= 0.45:
                role_strength = 0.62
                shared_candidates += 1
            elif tokens >= 4096 and float(traits["sharedness"]) >= 0.35:
                role_strength = 0.48
                is_large_candidate = True
        else:
            if remaining_use_count > 0 or seen_count > 0 or group_seen_count >= 2:
                role_strength = 0.50
            elif tokens >= 4096:
                role_strength = 0.35
                is_large_candidate = True

        if role_strength <= 0.0:
            continue

        share_strength = _clamp(
            float(max(group_seen_count, seen_count + 1, int(round(float(traits["sharedness"]) * 5.0))))
            / 6.0,
            0.35,
            1.00,
        )
        if remaining_use_count > 0 or seen_count > 0:
            share_strength = max(
                share_strength,
                _clamp(
                    float(max(remaining_use_count, seen_count) + 1) / 6.0,
                    0.35,
                    1.00,
                ),
            )
        if (remaining_use_count > 0 or seen_count > 0) and (
            next_use_distance is None or next_use_distance <= 8
        ):
            trace_strength = 1.00
        elif remaining_use_count > 0 or seen_count > 0:
            trace_strength = 0.92
        else:
            trace_strength = 0.70
        size_strength = _clamp(float(tokens) / 1536.0, 0.40, 1.00)
        fragment_weighted_tokens = float(tokens) * (
            kind_strength
            * role_strength
            * share_strength
            * trace_strength
            * size_strength
        )
        if fragment_weighted_tokens < 192.0:
            continue

        weighted_tokens += float(fragment_weighted_tokens)
        candidate_fragments += 1
        candidate_tokens += tokens
        if is_large_candidate or tokens >= 4096:
            large_candidates += 1

    if candidate_fragments <= 0:
        return 0.0, {
            "skills_miss_blend_bonus_ms": 0.0,
            "skills_miss_blend_weighted_tokens": 0.0,
            "skills_miss_blend_candidate_tokens": 0.0,
            "skills_miss_blend_candidate_fragments": 0.0,
            "skills_miss_blend_cached_hit_tokens": 0.0,
            "skills_miss_blend_cached_hit_fragments": 0.0,
            "skills_miss_blend_shared_candidates": 0.0,
            "skills_miss_blend_trace_candidates": 0.0,
            "skills_miss_blend_large_candidates": 0.0,
            "skills_miss_blend_all_miss_long_guard": 0.0,
            "skills_miss_blend_applied": 0.0,
        }

    should_apply = (
        weighted_tokens >= float(SKILLS_MISS_BLEND_MIN_WEIGHTED_TOKENS)
        and (shared_candidates > 0 or trace_candidates > 0 or large_candidates > 0)
    )
    if not should_apply:
        return 0.0, {
            "skills_miss_blend_bonus_ms": 0.0,
            "skills_miss_blend_weighted_tokens": float(weighted_tokens),
            "skills_miss_blend_candidate_tokens": float(candidate_tokens),
            "skills_miss_blend_candidate_fragments": float(candidate_fragments),
            "skills_miss_blend_cached_hit_tokens": float(cached_hit_tokens),
            "skills_miss_blend_cached_hit_fragments": float(cached_hit_fragments),
            "skills_miss_blend_shared_candidates": float(shared_candidates),
            "skills_miss_blend_trace_candidates": float(trace_candidates),
            "skills_miss_blend_large_candidates": float(large_candidates),
            "skills_miss_blend_all_miss_long_guard": 0.0,
            "skills_miss_blend_applied": 0.0,
        }

    # Large all-miss skill requests are where p90 regresses: we pay the
    # foreground materialization cost now, but the current request sees no reuse.
    # Keep this miss-aware bonus for medium requests, and let background admission
    # handle very large all-miss skill prompts.
    if (
        cached_hit_fragments <= 0
        and candidate_tokens >= 6144
        and shared_candidates <= 0
        and trace_candidates <= 0
    ):
        return 0.0, {
            "skills_miss_blend_bonus_ms": 0.0,
            "skills_miss_blend_weighted_tokens": float(weighted_tokens),
            "skills_miss_blend_candidate_tokens": float(candidate_tokens),
            "skills_miss_blend_candidate_fragments": float(candidate_fragments),
            "skills_miss_blend_cached_hit_tokens": float(cached_hit_tokens),
            "skills_miss_blend_cached_hit_fragments": float(cached_hit_fragments),
            "skills_miss_blend_shared_candidates": float(shared_candidates),
            "skills_miss_blend_trace_candidates": float(trace_candidates),
            "skills_miss_blend_large_candidates": float(large_candidates),
            "skills_miss_blend_all_miss_long_guard": 1.0,
            "skills_miss_blend_applied": 0.0,
        }

    effective_tokens = max(int(round(weighted_tokens)), 0)
    direct_saved_ms = float(cfg.cost_model.recompute_ms(effective_tokens))
    direct_load_ms = float(cfg.cost_model.transfer_ms(tokens_to_bytes(effective_tokens)))
    bonus_ms = max(direct_saved_ms - (1.15 * direct_load_ms) - 6.0, 0.0)
    bonus_ms = min(float(bonus_ms), float(SKILLS_MISS_BLEND_MAX_BONUS_MS))
    return float(bonus_ms), {
        "skills_miss_blend_bonus_ms": float(bonus_ms),
        "skills_miss_blend_weighted_tokens": float(weighted_tokens),
        "skills_miss_blend_candidate_tokens": float(candidate_tokens),
        "skills_miss_blend_candidate_fragments": float(candidate_fragments),
        "skills_miss_blend_cached_hit_tokens": float(cached_hit_tokens),
        "skills_miss_blend_cached_hit_fragments": float(cached_hit_fragments),
        "skills_miss_blend_shared_candidates": float(shared_candidates),
        "skills_miss_blend_trace_candidates": float(trace_candidates),
        "skills_miss_blend_large_candidates": float(large_candidates),
        "skills_miss_blend_all_miss_long_guard": 0.0,
        "skills_miss_blend_applied": 1.0 if bonus_ms > 0.0 else 0.0,
    }


def _semantic_miss_blend_bonus_ms(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    locations: list[str],
    cfg: "UtilityPlannerConfig",
    current_query_idx: int | None,
) -> tuple[float, dict[str, float]]:
    weighted_missing_tokens = 0.0
    miss_tokens = 0
    hit_tokens = 0
    support_candidates = 0
    context_candidates = 0
    reused_missing_fragments = 0
    cold_primary_large_missing = 0

    for fragment_id, location in zip(fragment_ids, locations, strict=True):
        fragment = fragments.get(fragment_id) or {}
        family = _canonical_workload_family(fragment)
        if family == "memgas":
            continue

        tokens = max(int(fragment.get("tokens", 0) or 0), 0)
        if tokens <= 0:
            continue

        hints = dict(fragment.get("hints") or {})
        traits = _semantic_pattern_traits(fragment=fragment, hints=hints)
        loc = str(location or "miss").lower()
        if loc != "miss":
            hit_tokens += tokens
            continue

        miss_tokens += tokens
        if bool(traits["cold_large_primary"]):
            cold_primary_large_missing += 1
            continue

        remaining_use_count, next_use_distance = _future_reuse_features(
            hints,
            current_query_idx=current_query_idx,
        )
        seen_count = max(int(hints.get("online_seen_count", 0) or 0), 0)
        group_seen_count = max(int(hints.get("online_group_seen_count", 0) or 0), 0)
        bootstrap = remaining_use_count > 0 or seen_count > 0 or group_seen_count >= 2
        if not bootstrap:
            continue

        support_artifact = bool(traits["support_artifact"])
        coherent_context = bool(traits["coherent_context"])
        if not support_artifact and not coherent_context:
            continue

        if support_artifact:
            support_candidates += 1
            role_strength = 1.0 if str(traits["role"]) == "core_support" else 0.78
        else:
            context_candidates += 1
            role_strength = 0.82 if bool(traits["summary_backed"]) else 0.68

        if next_use_distance is None and group_seen_count >= 2:
            proximity = 0.66
        elif next_use_distance is None:
            proximity = 0.48
        elif next_use_distance <= 1:
            proximity = 1.00
        elif next_use_distance <= 4:
            proximity = 0.82
        elif next_use_distance <= 8:
            proximity = 0.62
        else:
            proximity = 0.42

        reuse_strength = _clamp(
            float(max(remaining_use_count, seen_count, max(group_seen_count - 1, 0)))
            / 4.0,
            0.30,
            1.00,
        )
        semantic_strength = _clamp(
            0.18
            + (0.26 * float(traits["retention"]))
            + (0.24 * float(traits["stability"]))
            + (0.18 * float(traits["sharedness"]))
            + (0.14 * float(traits["importance"])),
            0.35,
            1.00,
        )
        if coherent_context and float(traits["retention"]) >= 0.58:
            semantic_strength += 0.08
        fragment_weighted_tokens = (
            float(tokens) * role_strength * proximity * reuse_strength * semantic_strength
        )
        # Coherent-context workloads like MemoryOS are composed of many
        # individually small but collectively valuable fragments; requiring a
        # large per-fragment floor suppresses the aggregate signal and pushes
        # all-miss queries to recompute.
        fragment_weight_floor = 0.0 if coherent_context else 128.0
        if fragment_weighted_tokens < fragment_weight_floor:
            continue

        weighted_missing_tokens += float(fragment_weighted_tokens)
        reused_missing_fragments += 1

    should_apply = (
        reused_missing_fragments >= 2
        and weighted_missing_tokens >= 768.0
        and miss_tokens >= max(1536, int(1.10 * float(hit_tokens)))
        and (support_candidates > 0 or context_candidates > 0)
    )
    if not should_apply:
        return 0.0, {
            "semantic_miss_blend_bonus_ms": 0.0,
            "semantic_miss_blend_weighted_tokens": float(weighted_missing_tokens),
            "semantic_miss_blend_missing_tokens": float(miss_tokens),
            "semantic_miss_blend_hit_tokens": float(hit_tokens),
            "semantic_miss_blend_support_candidates": float(support_candidates),
            "semantic_miss_blend_context_candidates": float(context_candidates),
            "semantic_miss_blend_reused_missing_fragments": float(reused_missing_fragments),
            "semantic_miss_blend_cold_large_primary_missing": float(cold_primary_large_missing),
            "semantic_miss_blend_applied": 0.0,
        }

    effective_tokens = max(int(round(weighted_missing_tokens)), 0)
    direct_saved_ms = float(cfg.cost_model.recompute_ms(effective_tokens))
    direct_load_ms = float(cfg.cost_model.transfer_ms(tokens_to_bytes(effective_tokens)))
    bonus_ms = max(direct_saved_ms - (1.15 * direct_load_ms) - 8.0, 0.0)
    bonus_ms = min(float(bonus_ms), 320.0)
    return float(bonus_ms), {
        "semantic_miss_blend_bonus_ms": float(bonus_ms),
        "semantic_miss_blend_weighted_tokens": float(weighted_missing_tokens),
        "semantic_miss_blend_missing_tokens": float(miss_tokens),
        "semantic_miss_blend_hit_tokens": float(hit_tokens),
        "semantic_miss_blend_support_candidates": float(support_candidates),
        "semantic_miss_blend_context_candidates": float(context_candidates),
        "semantic_miss_blend_reused_missing_fragments": float(reused_missing_fragments),
        "semantic_miss_blend_cold_large_primary_missing": float(cold_primary_large_missing),
        "semantic_miss_blend_applied": 1.0 if bonus_ms > 0.0 else 0.0,
    }


def _memgas_miss_blend_bonus_ms(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    locations: list[str],
    cfg: "UtilityPlannerConfig",
    current_query_idx: int | None,
) -> tuple[float, dict[str, float]]:
    total_fragments = 0
    miss_fragments = 0
    miss_tokens = 0
    hit_tokens = 0
    weighted_missing_tokens = 0.0
    reused_missing_fragments = 0
    summary_backed_missing = 0

    for fragment_id, location in zip(fragment_ids, locations, strict=True):
        fragment = fragments.get(fragment_id) or {}
        if _canonical_workload_family(fragment) != "memgas":
            continue

        total_fragments += 1
        tokens = max(int(fragment.get("tokens", 0) or 0), 0)
        if tokens <= 0:
            continue

        hints = dict(fragment.get("hints") or {})
        loc = str(location or "miss").lower()
        if loc != "miss":
            hit_tokens += tokens
            continue

        remaining_use_count, next_use_distance = _future_reuse_features(
            hints,
            current_query_idx=current_query_idx,
        )
        group_seen_count = max(int(hints.get("online_group_seen_count", 0) or 0), 0)
        bootstrap_from_group = False
        if remaining_use_count <= 0:
            bootstrap_from_group = group_seen_count >= 3
        if remaining_use_count <= 0 and not bootstrap_from_group:
            continue

        miss_fragments += 1
        miss_tokens += tokens

        importance = _clamp(float(hints.get("importance_score", 0.5) or 0.5), 0.0, 1.0)
        stability = _clamp(float(hints.get("stability_score", 0.5) or 0.5), 0.0, 1.0)
        sharedness = _clamp(float(hints.get("sharedness_score", 0.0) or 0.0), 0.0, 1.0)
        retention = _clamp(float(hints.get("retention_prior", 0.5) or 0.5), 0.0, 1.0)
        summary_signal = 1.0 if str(hints.get("summary") or "").strip() else 0.0
        if summary_signal > 0.0:
            summary_backed_missing += 1
        if bootstrap_from_group and (
            summary_signal <= 0.0 or retention < 0.78 or stability < 0.88
        ):
            continue

        if bootstrap_from_group:
            reuse_strength = _clamp(float(group_seen_count) / 6.0, 0.40, 0.85)
        else:
            reuse_strength = _clamp(float(remaining_use_count) / 3.0, 0.35, 1.0)
        if next_use_distance is None and bootstrap_from_group:
            proximity = 0.62
        elif next_use_distance is None:
            proximity = 0.70
        elif next_use_distance <= 1:
            proximity = 1.00
        elif next_use_distance <= 4:
            proximity = 0.88
        elif next_use_distance <= 8:
            proximity = 0.68
        else:
            proximity = 0.45

        semantic_strength = _clamp(
            0.20
            + (0.28 * retention)
            + (0.22 * stability)
            + (0.16 * importance)
            + (0.08 * sharedness)
            + (0.06 * summary_signal),
            0.35,
            1.0,
        )
        fragment_weighted_tokens = float(tokens) * semantic_strength * reuse_strength * proximity
        if fragment_weighted_tokens < 256.0:
            continue

        weighted_missing_tokens += float(fragment_weighted_tokens)
        reused_missing_fragments += 1

    if miss_fragments <= 0:
        return 0.0, {
            "memgas_miss_blend_bonus_ms": 0.0,
            "memgas_miss_blend_weighted_tokens": 0.0,
            "memgas_miss_blend_missing_tokens": 0.0,
            "memgas_miss_blend_hit_tokens": float(hit_tokens),
            "memgas_miss_blend_missing_fragments": 0.0,
            "memgas_miss_blend_total_fragments": float(total_fragments),
            "memgas_miss_blend_reused_missing_fragments": 0.0,
            "memgas_miss_blend_summary_backed_missing": 0.0,
            "memgas_miss_blend_applied": 0.0,
        }

    miss_dominates = float(miss_tokens) >= max(1536.0, 1.25 * float(hit_tokens))
    should_apply = (
        reused_missing_fragments >= 2
        and weighted_missing_tokens >= float(MEMGAS_MISS_BLEND_MIN_WEIGHTED_TOKENS)
        and miss_dominates
        and summary_backed_missing >= 1
    )
    if not should_apply:
        return 0.0, {
            "memgas_miss_blend_bonus_ms": 0.0,
            "memgas_miss_blend_weighted_tokens": float(weighted_missing_tokens),
            "memgas_miss_blend_missing_tokens": float(miss_tokens),
            "memgas_miss_blend_hit_tokens": float(hit_tokens),
            "memgas_miss_blend_missing_fragments": float(miss_fragments),
            "memgas_miss_blend_total_fragments": float(total_fragments),
            "memgas_miss_blend_reused_missing_fragments": float(reused_missing_fragments),
            "memgas_miss_blend_summary_backed_missing": float(summary_backed_missing),
            "memgas_miss_blend_applied": 0.0,
        }

    effective_tokens = max(int(round(weighted_missing_tokens)), 0)
    direct_saved_ms = float(cfg.cost_model.recompute_ms(effective_tokens))
    direct_load_ms = float(cfg.cost_model.transfer_ms(tokens_to_bytes(effective_tokens)))
    bonus_ms = max(direct_saved_ms - (1.20 * direct_load_ms) - 10.0, 0.0)
    bonus_ms = min(float(bonus_ms), float(MEMGAS_MISS_BLEND_MAX_BONUS_MS))
    return float(bonus_ms), {
        "memgas_miss_blend_bonus_ms": float(bonus_ms),
        "memgas_miss_blend_weighted_tokens": float(weighted_missing_tokens),
        "memgas_miss_blend_missing_tokens": float(miss_tokens),
        "memgas_miss_blend_hit_tokens": float(hit_tokens),
        "memgas_miss_blend_missing_fragments": float(miss_fragments),
        "memgas_miss_blend_total_fragments": float(total_fragments),
        "memgas_miss_blend_reused_missing_fragments": float(reused_missing_fragments),
        "memgas_miss_blend_summary_backed_missing": float(summary_backed_missing),
        "memgas_miss_blend_applied": 1.0 if bonus_ms > 0.0 else 0.0,
    }


def _fragmented_miss_fastpath_bonus_ms(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    locations: list[str],
    cfg: UtilityPlannerConfig,
) -> tuple[float, dict[str, float]]:
    total_fragments = max(len(fragment_ids), 1)
    total_tokens = 0.0
    miss_tokens = 0.0
    miss_fragments = 0
    coherent_context_missing = 0
    support_artifact_missing = 0
    summary_backed_missing = 0
    stable_missing = 0

    for fragment_id, location in zip(fragment_ids, locations, strict=True):
        fragment = fragments.get(fragment_id) or {}
        tokens = max(int(fragment.get("tokens", 0) or 0), 0)
        total_tokens += float(tokens)
        if str(location or "miss").lower() != "miss":
            continue

        miss_tokens += float(tokens)
        miss_fragments += 1
        hints = dict(fragment.get("hints") or {})
        traits = _semantic_pattern_traits(fragment=fragment, hints=hints)
        if bool(traits["coherent_context"]):
            coherent_context_missing += 1
        if bool(traits["support_artifact"]):
            support_artifact_missing += 1
        if bool(traits["summary_backed"]):
            summary_backed_missing += 1
        if (
            float(traits["retention"]) >= 0.58
            or float(traits["stability"]) >= 0.65
            or float(traits["importance"]) >= 0.62
        ):
            stable_missing += 1

    miss_ratio = (float(miss_tokens) / float(total_tokens)) if total_tokens > 0.0 else 0.0
    avg_miss_tokens_per_fragment = (
        float(miss_tokens) / float(miss_fragments) if miss_fragments > 0 else 0.0
    )
    native_runtime = str(getattr(cfg, "native_runtime", "recompute") or "recompute")
    semantic_evidence = (
        summary_backed_missing >= 1
        or coherent_context_missing >= 1
        or support_artifact_missing >= 1
        or stable_missing >= 2
    )

    if native_runtime.lower() != "recompute":
        bonus_ms = 0.0
    elif total_fragments < int(FRAGMENTED_MISS_FASTPATH_MIN_FRAGMENTS):
        bonus_ms = 0.0
    elif miss_fragments < int(FRAGMENTED_MISS_FASTPATH_MIN_FRAGMENTS):
        bonus_ms = 0.0
    elif miss_fragments > int(FRAGMENTED_MISS_FASTPATH_MAX_FRAGMENTS):
        bonus_ms = 0.0
    elif miss_tokens < float(FRAGMENTED_MISS_FASTPATH_MIN_MISS_TOKENS):
        bonus_ms = 0.0
    elif miss_ratio < float(FRAGMENTED_MISS_FASTPATH_MIN_MISS_RATIO):
        bonus_ms = 0.0
    elif avg_miss_tokens_per_fragment < float(FRAGMENTED_MISS_FASTPATH_MIN_AVG_TOKENS):
        bonus_ms = 0.0
    elif avg_miss_tokens_per_fragment > float(FRAGMENTED_MISS_FASTPATH_MAX_AVG_TOKENS):
        bonus_ms = 0.0
    else:
        fragment_strength = _clamp(float(miss_fragments) / 8.0, 0.45, 1.0)
        size_bonus_ms = 0.12 * float(cfg.cost_model.recompute_ms(int(miss_tokens)))
        fragment_bonus_ms = 12.0 * float(fragment_strength)
        semantic_bonus_ms = 10.0 if semantic_evidence else 0.0
        bonus_ms = min(
            float(size_bonus_ms + fragment_bonus_ms + semantic_bonus_ms),
            float(FRAGMENTED_MISS_FASTPATH_MAX_BONUS_MS),
        )

    return float(bonus_ms), {
        "fragmented_miss_fastpath_bonus_ms": float(bonus_ms),
        "fragmented_miss_fastpath_total_tokens": float(total_tokens),
        "fragmented_miss_fastpath_miss_tokens": float(miss_tokens),
        "fragmented_miss_fastpath_miss_ratio": float(miss_ratio),
        "fragmented_miss_fastpath_total_fragments": float(total_fragments),
        "fragmented_miss_fastpath_miss_fragments": float(miss_fragments),
        "fragmented_miss_fastpath_avg_miss_tokens_per_fragment": float(
            avg_miss_tokens_per_fragment
        ),
        "fragmented_miss_fastpath_summary_backed_missing": float(
            summary_backed_missing
        ),
        "fragmented_miss_fastpath_coherent_context_missing": float(
            coherent_context_missing
        ),
        "fragmented_miss_fastpath_support_artifact_missing": float(
            support_artifact_missing
        ),
        "fragmented_miss_fastpath_stable_missing": float(stable_missing),
        "fragmented_miss_fastpath_semantic_evidence": 1.0 if semantic_evidence else 0.0,
        "fragmented_miss_fastpath_applied": 1.0 if bonus_ms > 0.0 else 0.0,
    }


@dataclass(frozen=True)
class UtilityPlannerConfig:
    cost_model: CostModel
    tail_lambda: float = 0.0
    enable_cpu_recompute: bool = False
    gpu_lookup_penalty_ms: float = 0.10
    cpu_lookup_penalty_ms: float = 2.20
    native_runtime: str = "recompute"
    hybrid_gate_base_margin_ms: float = HYBRID_GATE_BASE_MARGIN_MS
    hybrid_gate_cpu_load_margin_ms: float = HYBRID_GATE_CPU_LOAD_MARGIN_MS
    hybrid_gate_miss_margin_ms: float = HYBRID_GATE_MISS_MARGIN_MS
    hybrid_gate_fragment_ref_tokens: float = HYBRID_GATE_FRAGMENT_REF_TOKENS
    hybrid_gate_fragment_scale_min: float = HYBRID_GATE_FRAGMENT_SCALE_MIN
    hybrid_gate_fragment_scale_max: float = HYBRID_GATE_FRAGMENT_SCALE_MAX
    hybrid_gate_small_cached_token_threshold: float = (
        HYBRID_GATE_SMALL_CACHED_TOKEN_THRESHOLD
    )
    hybrid_gate_small_cached_margin_ms: float = HYBRID_GATE_SMALL_CACHED_MARGIN_MS
    hybrid_gate_all_cached_override: bool = True

    @staticmethod
    def from_specs(
        *,
        cost_model_spec: str = "",
        tail_lambda: float = 0.0,
        enable_cpu_recompute: bool = False,
        gpu_lookup_penalty_ms: float = 0.10,
        cpu_lookup_penalty_ms: float = 2.20,
        native_runtime: str = "recompute",
        hybrid_gate_base_margin_ms: float = HYBRID_GATE_BASE_MARGIN_MS,
        hybrid_gate_cpu_load_margin_ms: float = HYBRID_GATE_CPU_LOAD_MARGIN_MS,
        hybrid_gate_miss_margin_ms: float = HYBRID_GATE_MISS_MARGIN_MS,
        hybrid_gate_fragment_ref_tokens: float = HYBRID_GATE_FRAGMENT_REF_TOKENS,
        hybrid_gate_fragment_scale_min: float = HYBRID_GATE_FRAGMENT_SCALE_MIN,
        hybrid_gate_fragment_scale_max: float = HYBRID_GATE_FRAGMENT_SCALE_MAX,
        hybrid_gate_small_cached_token_threshold: float = HYBRID_GATE_SMALL_CACHED_TOKEN_THRESHOLD,
        hybrid_gate_small_cached_margin_ms: float = HYBRID_GATE_SMALL_CACHED_MARGIN_MS,
        hybrid_gate_all_cached_override: bool = True,
    ) -> "UtilityPlannerConfig":
        return UtilityPlannerConfig(
            cost_model=CostModel.from_spec(cost_model_spec),
            tail_lambda=float(tail_lambda or 0.0),
            enable_cpu_recompute=bool(enable_cpu_recompute),
            gpu_lookup_penalty_ms=float(gpu_lookup_penalty_ms or 0.0),
            cpu_lookup_penalty_ms=float(cpu_lookup_penalty_ms or 0.0),
            native_runtime=str(native_runtime or "recompute"),
            hybrid_gate_base_margin_ms=float(hybrid_gate_base_margin_ms or 0.0),
            hybrid_gate_cpu_load_margin_ms=float(
                hybrid_gate_cpu_load_margin_ms or 0.0
            ),
            hybrid_gate_miss_margin_ms=float(hybrid_gate_miss_margin_ms or 0.0),
            hybrid_gate_fragment_ref_tokens=float(
                hybrid_gate_fragment_ref_tokens or 0.0
            ),
            hybrid_gate_fragment_scale_min=float(
                hybrid_gate_fragment_scale_min or 0.0
            ),
            hybrid_gate_fragment_scale_max=float(
                hybrid_gate_fragment_scale_max or 0.0
            ),
            hybrid_gate_small_cached_token_threshold=float(
                hybrid_gate_small_cached_token_threshold or 0.0
            ),
            hybrid_gate_small_cached_margin_ms=float(
                hybrid_gate_small_cached_margin_ms or 0.0
            ),
            hybrid_gate_all_cached_override=bool(hybrid_gate_all_cached_override),
        )


def choose_fetch_action(
    *,
    location: str,
    tokens: int,
    cfg: UtilityPlannerConfig,
) -> FetchAction:
    loc = str(location or "miss").lower()
    if loc == "gpu":
        return "reuse_gpu"
    if loc == "miss":
        return "recompute"
    if loc != "cpu":
        return "recompute"

    if not cfg.enable_cpu_recompute:
        return "load_cpu"

    token_count = max(int(tokens or 0), 0)
    if token_count <= 0:
        return "load_cpu"

    recompute_ms = float(cfg.cost_model.recompute_ms(token_count))
    load_ms = float(cfg.cost_model.transfer_ms(tokens_to_bytes(token_count)))
    load_ms *= 1.0 + 0.25 * _clamp(cfg.tail_lambda, 0.0, 1.0)
    return "recompute" if recompute_ms <= load_ms else "load_cpu"


def estimate_cacheblend_utility_ms(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    locations: list[str],
    cfg: UtilityPlannerConfig,
) -> tuple[float, dict[str, float], list[FetchAction]]:
    saved_gpu_ms = 0.0
    saved_cpu_ms = 0.0
    transfer_ms_total = 0.0
    recompute_ms_total = 0.0
    gpu_lookup_penalty_ms = 0.0
    cpu_lookup_penalty_ms = 0.0
    actions: list[FetchAction] = []

    for fragment_id, location in zip(fragment_ids, locations, strict=True):
        fragment = fragments.get(fragment_id) or {}
        tokens = max(int(fragment.get("tokens", 0) or 0), 0)
        recompute_ms = float(cfg.cost_model.recompute_ms(tokens))
        action = choose_fetch_action(location=location, tokens=tokens, cfg=cfg)
        actions.append(action)
        recompute_ms_total += recompute_ms

        if action == "reuse_gpu":
            saved_gpu_ms += recompute_ms
            gpu_lookup_penalty_ms += float(cfg.gpu_lookup_penalty_ms)
            continue

        if action == "load_cpu":
            load_ms = float(cfg.cost_model.transfer_ms(tokens_to_bytes(tokens)))
            transfer_ms_total += load_ms
            saved_cpu_ms += max(recompute_ms - load_ms, 0.0)
            cpu_lookup_penalty_ms += float(cfg.cpu_lookup_penalty_ms) * (
                1.0 + 0.25 * _clamp(cfg.tail_lambda, 0.0, 1.0)
            )

    compose_ms = 0.0
    if hasattr(cfg.cost_model, "compose_ms"):
        compose_ms = float(cfg.cost_model.compose_ms(len(fragment_ids)))
    transfer_tail_penalty_ms = float(transfer_ms_total) * (
        0.10 + 0.25 * _clamp(cfg.tail_lambda, 0.0, 1.0)
    )
    utility_ms = float(
        saved_gpu_ms
        + saved_cpu_ms
        - compose_ms
        - transfer_tail_penalty_ms
        - gpu_lookup_penalty_ms
        - cpu_lookup_penalty_ms
    )
    debug = {
        "saved_gpu_ms": float(saved_gpu_ms),
        "saved_cpu_ms": float(saved_cpu_ms),
        "transfer_ms_total": float(transfer_ms_total),
        "recompute_ms_total": float(recompute_ms_total),
        "compose_ms": float(compose_ms),
        "transfer_tail_penalty_ms": float(transfer_tail_penalty_ms),
        "gpu_lookup_penalty_ms": float(gpu_lookup_penalty_ms),
        "cpu_lookup_penalty_ms": float(cpu_lookup_penalty_ms),
        "utility_ms": float(utility_ms),
    }
    return float(utility_ms), debug, actions


def estimate_native_prefix_utility_ms(
    *,
    prefix_reuse_tokens: int,
    cfg: UtilityPlannerConfig,
) -> tuple[float, dict[str, float]]:
    reuse_tokens = max(int(prefix_reuse_tokens or 0), 0)
    native_runtime = str(getattr(cfg, "native_runtime", "recompute") or "recompute")
    if native_runtime.lower() != "prefix_cache":
        return 0.0, {
            "prefix_reuse_tokens": float(reuse_tokens),
            "utility_ms": 0.0,
            "native_runtime_recompute": 1.0,
        }
    reused_ms = float(cfg.cost_model.recompute_ms(reuse_tokens))
    return float(reused_ms), {
        "prefix_reuse_tokens": float(reuse_tokens),
        "utility_ms": float(reused_ms),
        "native_runtime_recompute": 0.0,
    }


def _predict_native_ttft_ms(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    prefix_reuse_tokens: int,
    cfg: UtilityPlannerConfig,
) -> tuple[float, dict[str, float]]:
    total_tokens = 0
    for fragment_id in fragment_ids:
        fragment = fragments.get(fragment_id) or {}
        total_tokens += max(int(fragment.get("tokens", 0) or 0), 0)

    recompute_total_ms = float(cfg.cost_model.recompute_ms(total_tokens))
    native_saved_ms, native_debug = estimate_native_prefix_utility_ms(
        prefix_reuse_tokens=int(prefix_reuse_tokens or 0),
        cfg=cfg,
    )
    predicted_native_ttft_ms = max(float(recompute_total_ms) - float(native_saved_ms), 0.0)
    debug = {
        "native_total_tokens": float(total_tokens),
        "native_recompute_total_ms": float(recompute_total_ms),
        "native_saved_ms": float(native_saved_ms),
        "native_predicted_ttft_ms": float(predicted_native_ttft_ms),
    }
    debug.update({f"native_prefix_debug_{k}": float(v) for k, v in native_debug.items()})
    return float(predicted_native_ttft_ms), debug


def _predict_blend_ttft_ms(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    locations: list[str],
    cfg: UtilityPlannerConfig,
    fetch_actions: list[FetchAction],
    cacheblend_debug: dict[str, float],
    skills_bootstrap_bonus_ms: float,
    skills_miss_blend_bonus_ms: float,
    semantic_miss_blend_bonus_ms: float,
    memgas_miss_blend_bonus_ms: float,
    fragmented_miss_fastpath_bonus_ms: float,
) -> tuple[float, dict[str, float]]:
    miss_recompute_ms = 0.0
    miss_tokens = 0.0
    reused_gpu_tokens = 0.0
    cpu_load_tokens = 0.0
    miss_fragments = 0
    cpu_load_fragments = 0

    for fragment_id, location, action in zip(
        fragment_ids, locations, fetch_actions, strict=True
    ):
        del location
        fragment = fragments.get(fragment_id) or {}
        tokens = max(int(fragment.get("tokens", 0) or 0), 0)
        if action == "recompute":
            miss_recompute_ms += float(cfg.cost_model.recompute_ms(tokens))
            miss_tokens += float(tokens)
            miss_fragments += 1
        elif action == "reuse_gpu":
            reused_gpu_tokens += float(tokens)
        elif action == "load_cpu":
            cpu_load_tokens += float(tokens)
            cpu_load_fragments += 1

    lookup_ms = float(cacheblend_debug.get("gpu_lookup_penalty_ms", 0.0) or 0.0) + float(
        cacheblend_debug.get("cpu_lookup_penalty_ms", 0.0) or 0.0
    )
    cpu_load_ms = float(cacheblend_debug.get("transfer_ms_total", 0.0) or 0.0)
    compose_ms = float(cacheblend_debug.get("compose_ms", 0.0) or 0.0)
    transfer_tail_penalty_ms = float(
        cacheblend_debug.get("transfer_tail_penalty_ms", 0.0) or 0.0
    )
    semantic_bonus_ms = (
        float(skills_bootstrap_bonus_ms)
        + float(skills_miss_blend_bonus_ms)
        + float(semantic_miss_blend_bonus_ms)
        + float(memgas_miss_blend_bonus_ms)
        + float(fragmented_miss_fastpath_bonus_ms)
    )
    fragment_overhead_ms = (
        float(BLEND_QUERY_BASE_OVERHEAD_MS)
        + float(BLEND_FRAGMENT_OVERHEAD_MS) * float(len(fragment_ids))
        + float(BLEND_MISS_FRAGMENT_OVERHEAD_MS) * float(miss_fragments)
        + float(BLEND_CPU_LOAD_FRAGMENT_OVERHEAD_MS) * float(cpu_load_fragments)
    )

    predicted_blend_ttft_ms = (
        float(miss_recompute_ms)
        + float(cpu_load_ms)
        + float(compose_ms)
        + float(transfer_tail_penalty_ms)
        + float(lookup_ms)
        + float(fragment_overhead_ms)
        - float(semantic_bonus_ms)
    )
    predicted_blend_ttft_ms = max(float(predicted_blend_ttft_ms), 0.0)
    return float(predicted_blend_ttft_ms), {
        "blend_miss_recompute_ms": float(miss_recompute_ms),
        "blend_miss_tokens": float(miss_tokens),
        "blend_reused_gpu_tokens": float(reused_gpu_tokens),
        "blend_cpu_load_tokens": float(cpu_load_tokens),
        "blend_miss_fragments": float(miss_fragments),
        "blend_cpu_load_fragments": float(cpu_load_fragments),
        "blend_lookup_ms": float(lookup_ms),
        "blend_cpu_load_ms": float(cpu_load_ms),
        "blend_compose_ms": float(compose_ms),
        "blend_transfer_tail_penalty_ms": float(transfer_tail_penalty_ms),
        "blend_fragment_overhead_ms": float(fragment_overhead_ms),
        "blend_semantic_bonus_ms": float(semantic_bonus_ms),
        "blend_predicted_ttft_ms": float(predicted_blend_ttft_ms),
    }


def _hybrid_gate_penalty_ms(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    fetch_actions: list[FetchAction],
    cfg: UtilityPlannerConfig,
) -> tuple[float, dict[str, float]]:
    total_fragments = max(len(fragment_ids), 1)
    total_tokens = 0.0
    cached_tokens = 0.0

    for fragment_id in fragment_ids:
        fragment = fragments.get(fragment_id) or {}
        total_tokens += max(int(fragment.get("tokens", 0) or 0), 0)
    for fragment_id, action in zip(fragment_ids, fetch_actions, strict=True):
        if action == "recompute":
            continue
        fragment = fragments.get(fragment_id) or {}
        cached_tokens += max(int(fragment.get("tokens", 0) or 0), 0)

    avg_tokens_per_fragment = float(total_tokens) / float(total_fragments)
    fragment_scale = _clamp(
        float(cfg.hybrid_gate_fragment_ref_tokens)
        / max(float(avg_tokens_per_fragment), 1.0),
        float(cfg.hybrid_gate_fragment_scale_min),
        float(cfg.hybrid_gate_fragment_scale_max),
    )
    predicted_cpu_loads = sum(1 for action in fetch_actions if action == "load_cpu")
    predicted_gpu_reuses = sum(1 for action in fetch_actions if action == "reuse_gpu")
    predicted_recomputes = sum(1 for action in fetch_actions if action == "recompute")
    predicted_cached_fragments = predicted_cpu_loads + predicted_gpu_reuses
    all_cached = bool(fetch_actions) and predicted_recomputes == 0
    small_cached_penalty_ms = 0.0
    if (
        cached_tokens > 0.0
        and cached_tokens < float(cfg.hybrid_gate_small_cached_token_threshold)
    ):
        small_cached_penalty_ms = float(cfg.hybrid_gate_small_cached_margin_ms)

    coherent_context_miss_evidence = 0
    support_artifact_miss_evidence = 0
    if small_cached_penalty_ms > 0.0 and predicted_recomputes > 0:
        for fragment_id, action in zip(fragment_ids, fetch_actions, strict=True):
            if action != "recompute":
                continue
            fragment = fragments.get(fragment_id) or {}
            hints = dict(fragment.get("hints") or {})
            traits = _semantic_pattern_traits(fragment=fragment, hints=hints)
            remaining_use_count, _next_use_distance = _future_reuse_features(
                hints,
                current_query_idx=None,
            )
            seen_count = max(int(hints.get("online_seen_count", 0) or 0), 0)
            group_seen_count = max(int(hints.get("online_group_seen_count", 0) or 0), 0)
            has_online_evidence = (
                remaining_use_count > 0 or seen_count > 0 or group_seen_count >= 2
            )
            if not has_online_evidence:
                continue
            if bool(traits["coherent_context"]):
                coherent_context_miss_evidence += 1
            elif bool(traits["support_artifact"]):
                support_artifact_miss_evidence += 1

        if coherent_context_miss_evidence >= 1:
            small_cached_penalty_ms = 0.0
        elif support_artifact_miss_evidence >= 2:
            small_cached_penalty_ms *= 0.35

    if all_cached and bool(cfg.hybrid_gate_all_cached_override):
        return 0.0, {
            "hybrid_gate_base_margin_ms": float(cfg.hybrid_gate_base_margin_ms),
            "hybrid_gate_cpu_load_margin_ms": float(
                cfg.hybrid_gate_cpu_load_margin_ms
            ),
            "hybrid_gate_miss_margin_ms": float(cfg.hybrid_gate_miss_margin_ms),
            "hybrid_gate_fragment_ref_tokens": float(
                cfg.hybrid_gate_fragment_ref_tokens
            ),
            "hybrid_gate_fragment_scale_min": float(
                cfg.hybrid_gate_fragment_scale_min
            ),
            "hybrid_gate_fragment_scale_max": float(
                cfg.hybrid_gate_fragment_scale_max
            ),
            "hybrid_gate_fragment_scale": float(fragment_scale),
            "hybrid_gate_avg_tokens_per_fragment": float(avg_tokens_per_fragment),
            "hybrid_gate_cached_tokens": float(cached_tokens),
            "hybrid_gate_cached_fragments": float(predicted_cached_fragments),
            "hybrid_gate_predicted_cpu_loads": float(predicted_cpu_loads),
            "hybrid_gate_predicted_gpu_reuses": float(predicted_gpu_reuses),
            "hybrid_gate_predicted_recomputes": float(predicted_recomputes),
            "hybrid_gate_small_cached_token_threshold": float(
                cfg.hybrid_gate_small_cached_token_threshold
            ),
            "hybrid_gate_small_cached_margin_ms": float(
                cfg.hybrid_gate_small_cached_margin_ms
            ),
            "hybrid_gate_small_cached_penalty_ms": float(small_cached_penalty_ms),
            "hybrid_gate_coherent_context_miss_evidence": float(
                coherent_context_miss_evidence
            ),
            "hybrid_gate_support_artifact_miss_evidence": float(
                support_artifact_miss_evidence
            ),
            "hybrid_gate_all_cached": 1.0,
            "hybrid_gate_all_cached_override": 1.0,
            "hybrid_gate_penalty_ms": 0.0,
        }

    penalty_ms = float(cfg.hybrid_gate_base_margin_ms) + float(fragment_scale) * float(
        float(cfg.hybrid_gate_cpu_load_margin_ms) * predicted_cpu_loads
        + float(cfg.hybrid_gate_miss_margin_ms) * predicted_recomputes
    ) + float(small_cached_penalty_ms)
    return float(penalty_ms), {
        "hybrid_gate_base_margin_ms": float(cfg.hybrid_gate_base_margin_ms),
        "hybrid_gate_cpu_load_margin_ms": float(cfg.hybrid_gate_cpu_load_margin_ms),
        "hybrid_gate_miss_margin_ms": float(cfg.hybrid_gate_miss_margin_ms),
        "hybrid_gate_fragment_ref_tokens": float(
            cfg.hybrid_gate_fragment_ref_tokens
        ),
        "hybrid_gate_fragment_scale_min": float(cfg.hybrid_gate_fragment_scale_min),
        "hybrid_gate_fragment_scale_max": float(cfg.hybrid_gate_fragment_scale_max),
        "hybrid_gate_fragment_scale": float(fragment_scale),
        "hybrid_gate_avg_tokens_per_fragment": float(avg_tokens_per_fragment),
        "hybrid_gate_cached_tokens": float(cached_tokens),
        "hybrid_gate_cached_fragments": float(predicted_cached_fragments),
        "hybrid_gate_predicted_cpu_loads": float(predicted_cpu_loads),
        "hybrid_gate_predicted_gpu_reuses": float(predicted_gpu_reuses),
        "hybrid_gate_predicted_recomputes": float(predicted_recomputes),
        "hybrid_gate_small_cached_token_threshold": float(
            cfg.hybrid_gate_small_cached_token_threshold
        ),
        "hybrid_gate_small_cached_margin_ms": float(
            cfg.hybrid_gate_small_cached_margin_ms
        ),
        "hybrid_gate_small_cached_penalty_ms": float(small_cached_penalty_ms),
        "hybrid_gate_coherent_context_miss_evidence": float(
            coherent_context_miss_evidence
        ),
        "hybrid_gate_support_artifact_miss_evidence": float(
            support_artifact_miss_evidence
        ),
        "hybrid_gate_all_cached": 1.0 if all_cached else 0.0,
        "hybrid_gate_all_cached_override": 0.0,
        "hybrid_gate_penalty_ms": float(penalty_ms),
    }


def choose_execution_mode(
    *,
    fragment_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    locations: list[str],
    prefix_reuse_tokens: int,
    fallback_margin_ms: float,
    cfg: UtilityPlannerConfig,
    current_query_idx: int | None = None,
) -> tuple[ExecutionMode, dict[str, Any]]:
    native_ms, native_debug = estimate_native_prefix_utility_ms(
        prefix_reuse_tokens=int(prefix_reuse_tokens or 0),
        cfg=cfg,
    )
    cacheblend_ms, cacheblend_debug, fetch_actions = estimate_cacheblend_utility_ms(
        fragment_ids=list(fragment_ids),
        fragments=fragments,
        locations=list(locations),
        cfg=cfg,
    )
    margin_ms = float(fallback_margin_ms or 0.0)
    gate_penalty_ms, gate_debug = _hybrid_gate_penalty_ms(
        fragment_ids=list(fragment_ids),
        fragments=fragments,
        fetch_actions=list(fetch_actions),
        cfg=cfg,
    )
    skills_bootstrap_bonus_ms, skills_bootstrap_debug = _skills_bootstrap_bonus_ms(
        fragment_ids=list(fragment_ids),
        fragments=fragments,
        locations=list(locations),
        current_query_idx=current_query_idx,
    )
    skills_miss_blend_bonus_ms, skills_miss_blend_debug = _skills_miss_blend_bonus_ms(
        fragment_ids=list(fragment_ids),
        fragments=fragments,
        locations=list(locations),
        cfg=cfg,
        current_query_idx=current_query_idx,
    )
    semantic_miss_blend_bonus_ms, semantic_miss_blend_debug = _semantic_miss_blend_bonus_ms(
        fragment_ids=list(fragment_ids),
        fragments=fragments,
        locations=list(locations),
        cfg=cfg,
        current_query_idx=current_query_idx,
    )
    memgas_miss_blend_bonus_ms, memgas_miss_blend_debug = _memgas_miss_blend_bonus_ms(
        fragment_ids=list(fragment_ids),
        fragments=fragments,
        locations=list(locations),
        cfg=cfg,
        current_query_idx=current_query_idx,
    )
    fragmented_miss_fastpath_bonus_ms, fragmented_miss_fastpath_debug = (
        _fragmented_miss_fastpath_bonus_ms(
            fragment_ids=list(fragment_ids),
            fragments=fragments,
            locations=list(locations),
            cfg=cfg,
        )
    )
    cacheblend_effective_ms = (
        float(cacheblend_ms)
        + float(skills_bootstrap_bonus_ms)
        + float(skills_miss_blend_bonus_ms)
        + float(semantic_miss_blend_bonus_ms)
        + float(memgas_miss_blend_bonus_ms)
        + float(fragmented_miss_fastpath_bonus_ms)
    )
    predicted_native_ttft_ms, predicted_native_debug = _predict_native_ttft_ms(
        fragment_ids=list(fragment_ids),
        fragments=fragments,
        prefix_reuse_tokens=int(prefix_reuse_tokens or 0),
        cfg=cfg,
    )
    predicted_blend_ttft_ms, predicted_blend_debug = _predict_blend_ttft_ms(
        fragment_ids=list(fragment_ids),
        fragments=fragments,
        locations=list(locations),
        cfg=cfg,
        fetch_actions=list(fetch_actions),
        cacheblend_debug=cacheblend_debug,
        skills_bootstrap_bonus_ms=float(skills_bootstrap_bonus_ms),
        skills_miss_blend_bonus_ms=float(skills_miss_blend_bonus_ms),
        semantic_miss_blend_bonus_ms=float(semantic_miss_blend_bonus_ms),
        memgas_miss_blend_bonus_ms=float(memgas_miss_blend_bonus_ms),
        fragmented_miss_fastpath_bonus_ms=float(fragmented_miss_fastpath_bonus_ms),
    )
    effective_margin_ms = float(margin_ms) + float(gate_penalty_ms)
    force_cacheblend = bool(
        gate_debug.get("hybrid_gate_all_cached_override", 0.0)
    )
    choose_cacheblend = force_cacheblend or (
        float(predicted_blend_ttft_ms) + float(effective_margin_ms)
        < float(predicted_native_ttft_ms)
    )
    mode: ExecutionMode = "cacheblend" if choose_cacheblend else "native_prefix"
    debug: dict[str, Any] = {
        "native_prefix_utility_ms": float(native_ms),
        "cacheblend_utility_ms": float(cacheblend_ms),
        "cacheblend_effective_utility_ms": float(cacheblend_effective_ms),
        "predicted_native_ttft_ms": float(predicted_native_ttft_ms),
        "predicted_blend_ttft_ms": float(predicted_blend_ttft_ms),
        "skills_bootstrap_bonus_ms": float(skills_bootstrap_bonus_ms),
        "skills_miss_blend_bonus_ms": float(skills_miss_blend_bonus_ms),
        "semantic_miss_blend_bonus_ms": float(semantic_miss_blend_bonus_ms),
        "memgas_miss_blend_bonus_ms": float(memgas_miss_blend_bonus_ms),
        "fragmented_miss_fastpath_bonus_ms": float(
            fragmented_miss_fastpath_bonus_ms
        ),
        "fallback_margin_ms": float(margin_ms),
        "effective_fallback_margin_ms": float(effective_margin_ms),
        "hybrid_gate_penalty_ms": float(gate_penalty_ms),
        "force_cacheblend_all_cached": bool(force_cacheblend),
        "decision_gap_ms": float(predicted_native_ttft_ms - predicted_blend_ttft_ms),
        "chosen_mode": str(mode),
        "predicted_regret_ms": float(
            max(predicted_native_ttft_ms, predicted_blend_ttft_ms)
            - (predicted_blend_ttft_ms if choose_cacheblend else predicted_native_ttft_ms)
        ),
        "prefix_reuse_tokens": int(prefix_reuse_tokens or 0),
        "cacheblend_fetch_actions": list(fetch_actions),
        "native_prefix_debug": native_debug,
        "cacheblend_debug": cacheblend_debug,
        "predicted_native_debug": predicted_native_debug,
        "predicted_blend_debug": predicted_blend_debug,
        "hybrid_gate_debug": gate_debug,
        "skills_bootstrap_debug": skills_bootstrap_debug,
        "skills_miss_blend_debug": skills_miss_blend_debug,
        "semantic_miss_blend_debug": semantic_miss_blend_debug,
        "memgas_miss_blend_debug": memgas_miss_blend_debug,
        "fragmented_miss_fastpath_debug": fragmented_miss_fastpath_debug,
    }
    return mode, debug
