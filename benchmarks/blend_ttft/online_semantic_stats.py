from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class AccessStats:
    query_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    last_query_idx: int | None = None
    last_gap: int | None = None
    ema_gap: float | None = None

    def observe(self, *, query_idx: int, hit: bool, alpha: float) -> None:
        if self.last_query_idx is not None:
            gap = max(int(query_idx) - int(self.last_query_idx), 0)
            self.last_gap = gap
            if self.ema_gap is None:
                self.ema_gap = float(gap)
            else:
                self.ema_gap = (float(alpha) * float(gap)) + (
                    (1.0 - float(alpha)) * float(self.ema_gap)
                )
        self.last_query_idx = int(query_idx)
        self.query_count += 1
        if hit:
            self.hit_count += 1
        else:
            self.miss_count += 1

    def queries_since_seen(self, current_query_idx: int) -> int | None:
        if self.last_query_idx is None:
            return None
        return max(int(current_query_idx) - int(self.last_query_idx), 0)


class OnlineSemanticStats:
    def __init__(self, *, gap_alpha: float = 0.5) -> None:
        self.gap_alpha = float(_clamp(gap_alpha, 0.05, 1.0))
        self.fragment_stats: dict[str, AccessStats] = {}
        self.group_stats: dict[str, AccessStats] = {}
        self.supported_families = frozenset(
            {"memos", "skillsbench", "memgas", "memoryos", "dspy_locomo"}
        )

    def inject_chunk_hints(
        self,
        *,
        unique_chunks: dict[str, dict[str, Any]],
        current_query_idx: int,
    ) -> None:
        for chunk_id, chunk in unique_chunks.items():
            hints = dict(chunk.get("hints") or {})
            static_prior = max(
                int(
                    hints.get(
                        "static_prior_use_count",
                        hints.get("prior_use_count", 0),
                    )
                    or 0
                ),
                0,
            )
            fragment_stats = self.fragment_stats.get(str(chunk_id))
            groups = self._groups_for_chunk(chunk)
            group_states = [
                self.group_stats[group]
                for group in groups
                if group in self.group_stats
            ]

            seen_count = int(fragment_stats.query_count) if fragment_stats else 0
            miss_pressure = (
                max(int(fragment_stats.miss_count) - int(fragment_stats.hit_count), 0)
                if fragment_stats
                else 0
            )
            group_seen_count = max(
                (int(state.query_count) for state in group_states),
                default=0,
            )
            group_miss_pressure = max(
                (
                    max(int(state.miss_count) - int(state.hit_count), 0)
                    for state in group_states
                ),
                default=0,
            )

            estimated_gap = self._estimate_gap(
                family=self._family_for_chunk(chunk),
                fragment_stats=fragment_stats,
                group_states=group_states,
                current_query_idx=int(current_query_idx),
            )
            estimated_reuses = self._estimate_future_uses(
                chunk=chunk,
                hints=hints,
                fragment_stats=fragment_stats,
                group_states=group_states,
                seen_count=seen_count,
                miss_pressure=miss_pressure,
                group_seen_count=group_seen_count,
                group_miss_pressure=group_miss_pressure,
            )

            hints["static_prior_use_count"] = static_prior
            hints["online_seen_count"] = seen_count
            hints["online_group_seen_count"] = group_seen_count
            family = self._family_for_chunk(chunk)
            if family not in self.supported_families:
                hints["prior_use_count"] = static_prior
                hints.pop("estimated_future_use_count", None)
                hints.pop("estimated_next_use_distance_queries", None)
                chunk["hints"] = hints
                continue

            hints["prior_use_count"] = max(static_prior, seen_count)
            if estimated_reuses is None:
                hints.pop("estimated_future_use_count", None)
            else:
                hints["estimated_future_use_count"] = int(max(estimated_reuses, 0))
            if estimated_gap is None or estimated_reuses is None:
                hints.pop("estimated_next_use_distance_queries", None)
            else:
                hints["estimated_next_use_distance_queries"] = int(
                    max(int(round(estimated_gap)), 0)
                )
            chunk["hints"] = hints

    def observe_query(
        self,
        *,
        unique_chunks: dict[str, dict[str, Any]],
        chunk_ids: list[str],
        query_idx: int,
        locations: dict[str, str] | None = None,
    ) -> None:
        seen_chunk_ids = list(dict.fromkeys(str(chunk_id) for chunk_id in chunk_ids))
        normalized_locations = {
            str(chunk_id): str(location or "miss").lower()
            for chunk_id, location in (locations or {}).items()
        }
        touched_groups: set[str] = set()

        for chunk_id in seen_chunk_ids:
            hit = normalized_locations.get(chunk_id, "miss") != "miss"
            fragment_stats = self.fragment_stats.setdefault(chunk_id, AccessStats())
            fragment_stats.observe(
                query_idx=int(query_idx),
                hit=bool(hit),
                alpha=self.gap_alpha,
            )
            chunk = unique_chunks.get(chunk_id) or {}
            for group in self._groups_for_chunk(chunk):
                if group in touched_groups:
                    continue
                touched_groups.add(group)
                group_stats = self.group_stats.setdefault(group, AccessStats())
                group_stats.observe(
                    query_idx=int(query_idx),
                    hit=bool(hit),
                    alpha=self.gap_alpha,
                )

    def _estimate_gap(
        self,
        *,
        family: str,
        fragment_stats: AccessStats | None,
        group_states: list[AccessStats],
        current_query_idx: int,
    ) -> float | None:
        candidates: list[float] = []
        if fragment_stats is not None:
            if fragment_stats.query_count >= 2 and fragment_stats.ema_gap is not None:
                candidates.append(float(fragment_stats.ema_gap))
            elif fragment_stats.query_count >= 2 and fragment_stats.last_gap is not None:
                candidates.append(float(fragment_stats.last_gap))

        if family != "memgas":
            for state in group_states:
                if state.query_count >= 2 and state.ema_gap is not None:
                    candidates.append(float(state.ema_gap))
                elif state.query_count >= 2 and state.last_gap is not None:
                    candidates.append(float(state.last_gap))

        if not candidates:
            return None
        return float(min(candidates))

    def _estimate_future_uses(
        self,
        *,
        chunk: dict[str, Any],
        hints: dict[str, Any],
        fragment_stats: AccessStats | None,
        group_states: list[AccessStats],
        seen_count: int,
        miss_pressure: int,
        group_seen_count: int,
        group_miss_pressure: int,
    ) -> int | None:
        family = self._family_for_chunk(chunk)
        role = str(hints.get("attachment_role", "") or "").strip().lower()
        sharedness = float(hints.get("sharedness_score", 0.0) or 0.0)
        retention = float(hints.get("retention_prior", 0.5) or 0.5)
        stability = float(hints.get("stability_score", 0.5) or 0.5)
        importance = float(hints.get("importance_score", 0.5) or 0.5)
        kind = str(chunk.get("type") or "").strip().lower()
        tokens = max(int(chunk.get("tokens", 0) or 0), 0)
        groups = {
            str(item).strip().lower()
            for item in (hints.get("semantic_groups") or [])
            if str(item).strip()
        }
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
            and seen_count <= 0
            and group_seen_count <= 1
            and sharedness < 0.45
            and retention < 0.58
        )

        if family == "memgas":
            if seen_count <= 0:
                return None
            predicted = min(seen_count, 3) + min(miss_pressure, 1)
            return int(max(predicted, 0))

        if cold_large_primary:
            return None

        exact_signal = min(seen_count, 4)
        shared_signal = min(group_seen_count, 4)
        pressure_bonus = min(miss_pressure + group_miss_pressure, 2)
        has_shared_bootstrap = shared_signal >= 2 and (
            support_artifact
            or (coherent_context and retention >= 0.56 and stability >= 0.54)
            or importance >= 0.72
        )

        if seen_count <= 0 and not has_shared_bootstrap:
            return None
        if (
            seen_count == 1
            and not has_shared_bootstrap
            and pressure_bonus <= 0
            and support_artifact is False
            and coherent_context is False
        ):
            return None

        predicted = exact_signal

        if seen_count <= 0:
            if support_artifact:
                predicted = 1 + min(shared_signal // 2, 2)
            elif coherent_context:
                predicted = 1 + min(shared_signal // 3, 1)
            else:
                predicted = 1
        else:
            predicted += 1
            if support_artifact:
                predicted += 1
            elif coherent_context and retention >= 0.60:
                predicted += 1

        if shared_signal >= 3 and role != "primary":
            predicted += 1
        if kind in {"skill_md", "reference"} and shared_signal >= 2:
            predicted += 1
        if coherent_context and stability >= 0.58 and shared_signal >= 2:
            predicted += 1
        if importance >= 0.78 and seen_count >= 1:
            predicted += 1

        predicted += pressure_bonus

        if role == "primary" and seen_count <= 0 and shared_signal <= 1 and sharedness < 0.50:
            return None
        return int(max(predicted, 0))

    def _groups_for_chunk(self, chunk: dict[str, Any]) -> list[str]:
        hints = dict(chunk.get("hints") or {})
        groups = [
            str(item).strip()
            for item in (hints.get("semantic_groups") or [])
            if str(item).strip()
        ]
        kind = str(chunk.get("type") or "").strip()
        role = str(hints.get("attachment_role") or "").strip()

        if not groups:
            if kind:
                groups.append(f"kind::{kind.lower()}")
            if role:
                groups.append(f"role::{role.lower()}")
        return list(dict.fromkeys(groups))

    def _family_for_chunk(self, chunk: dict[str, Any]) -> str:
        hints = dict(chunk.get("hints") or {})
        return str(
            hints.get("canonical_workload_family")
            or hints.get("canonical_workload_kind")
            or ""
        ).strip().lower()
