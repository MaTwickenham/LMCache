from __future__ import annotations

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
HYBRID_GATE_SMALL_CACHED_MARGIN_MS = 50.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


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
    effective_margin_ms = float(margin_ms) + float(gate_penalty_ms)
    force_cacheblend = bool(
        gate_debug.get("hybrid_gate_all_cached_override", 0.0)
    )
    choose_cacheblend = force_cacheblend or (
        float(cacheblend_ms) > float(native_ms) + float(effective_margin_ms)
    )
    mode: ExecutionMode = "cacheblend" if choose_cacheblend else "native_prefix"
    debug: dict[str, Any] = {
        "native_prefix_utility_ms": float(native_ms),
        "cacheblend_utility_ms": float(cacheblend_ms),
        "fallback_margin_ms": float(margin_ms),
        "effective_fallback_margin_ms": float(effective_margin_ms),
        "hybrid_gate_penalty_ms": float(gate_penalty_ms),
        "force_cacheblend_all_cached": bool(force_cacheblend),
        "decision_gap_ms": float(cacheblend_ms - native_ms),
        "chosen_mode": str(mode),
        "predicted_regret_ms": float(
            max(native_ms, cacheblend_ms)
            - (cacheblend_ms if choose_cacheblend else native_ms)
        ),
        "prefix_reuse_tokens": int(prefix_reuse_tokens or 0),
        "cacheblend_fetch_actions": list(fetch_actions),
        "native_prefix_debug": native_debug,
        "cacheblend_debug": cacheblend_debug,
        "hybrid_gate_debug": gate_debug,
    }
    return mode, debug
