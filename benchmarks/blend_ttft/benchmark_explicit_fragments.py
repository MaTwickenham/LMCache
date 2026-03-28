#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

from canonical_semantics import strip_past_only_runtime_hints
from compare_prefix_vs_blend_gpu_workload_server import gib_to_token_budget
from compare_prefix_vs_blend_memoryos_server import (
    LatencySummary,
    PhaseSummary,
    WorkloadStats,
    load_phase_records,
    summarize_measurements,
    summarize_online_phase_breakdown,
)
from explicit_fragment_control_plane import (
    ExplicitFragmentPool,
    LMCacheFragmentRuntime,
    MaintenanceStats,
)
from fragment_utility_planner import (
    UtilityPlannerConfig as OnlineUtilityPlannerConfig,
    choose_execution_mode,
)
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.integration.vllm.vllm_v1_adapter import (
    EXECUTION_MODE_BLEND,
    EXECUTION_MODE_NATIVE_VLLM,
)
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from online_semantic_stats import OnlineSemanticStats
from runtime_blend_probe import (
    build_phase_index,
    build_workload_for_probe,
    normalize_request_id,
)
from compare_prefix_vs_blend_gpu_workload_server import summarize_prefill_plan
from utility_guided_prefill import (
    UtilityPlannerConfig as MaintenanceUtilityPlannerConfig,
)


DEFAULT_MODEL = "/AI/HF_MODELS/Mistral-7B-Instruct-v0.2"
DEFAULT_DATA_ROOT = str(
    Path(__file__).resolve().parents[2] / "data" / "processed"
)
METHOD_CHOICES = ("no_prefix", "native_prefix", "explicit_blend")


def blend_cpu_bench_module() -> Any:
    import compare_prefix_vs_blend_cpu as mod

    return mod


def safe_release_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class RequestMeasurement:
    request_id: str | None
    ttft_s: float | None
    wall_s: float
    prompt_tokens: int
    cached_tokens: int | None
    generated_text: str


@dataclass
class MaintenanceSummary:
    request_count: int
    total_wall_s: float
    mean_wall_s: float
    total_materialize_s: float
    total_move_s: float
    total_evict_s: float
    total_admitted_chunks: int
    total_promoted_chunks: int
    total_demoted_chunks: int
    total_evicted_chunks: int
    total_admitted_tokens: int
    total_promoted_tokens: int
    total_demoted_tokens: int
    total_evicted_tokens: int


@dataclass
class ExplicitBlendSummary:
    online: LatencySummary
    online_phase_breakdown: dict[str, PhaseSummary]
    maintenance: MaintenanceSummary
    initial_materialization: MaintenanceSummary
    cpu_budget_tokens: int
    gpu_budget_tokens: int
    cpu_budget_gib: float
    gpu_budget_gib: float
    initial_fill_mode: str
    admit_misses_to: str
    gpu_lookahead: int
    fragment_management_policy: str
    online_execution_policy: str
    online_execution_mode: str
    blend_engine_prefix_caching: bool
    utility_enable_semantic_hints: bool
    online_use_historical_stats: bool
    utility_mode_recompute_requests: int
    utility_mode_blend_requests: int
    mean_predicted_recompute_utility_ms: float
    mean_predicted_cacheblend_utility_ms: float
    mean_predicted_decision_gap_ms: float
    resident_pool_end: dict[str, int]
    phase_file: str
    mean_shadow_hit_tokens: float
    mean_shadow_gpu_hit_tokens: float
    mean_shadow_cpu_hit_tokens: float
    mean_shadow_miss_tokens: float
    mean_shadow_hit_fragments: float
    mean_shadow_missed_fragments: float
    query_rows: list[dict[str, Any]]


@dataclass
class BenchmarkResult:
    model: str
    workload_kind: str
    datasets: list[str]
    chunk_size: int
    workload: WorkloadStats
    canonical_semantics_summary: dict[str, Any]
    no_prefix: LatencySummary | None
    native_prefix: LatencySummary | None
    explicit_blend: ExplicitBlendSummary | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark explicit-fragment LMCache serving semantics on real workloads. "
            "Online TTFT only measures query-time reuse; inter-query fragment "
            "admission / movement / eviction is reported separately."
        )
    )
    parser.add_argument(
        "--workload-kind",
        choices=[
            "memos",
            "memoryos",
            "amem",
            "skillsbench",
            "swe_skillsbench",
            "dspy_locomo",
            "memgas",
        ],
        required=True,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Comma-separated dataset names. If omitted, use the default "
            "datasets for the selected workload."
        ),
    )
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-local-gpu-size", type=float, default=0.0)
    parser.add_argument("--max-local-cpu-size", type=float, default=2.0)
    parser.add_argument(
        "--prompt-layout",
        choices=["memory_first", "question_first"],
        default="memory_first",
    )
    parser.add_argument(
        "--fragment-order-policy",
        choices=["auto", "trace", "shuffle", "profile_last", "profile_last_shuffle"],
        default="auto",
    )
    parser.add_argument(
        "--prefill-order-policy",
        choices=["sorted", "first_seen", "shuffle"],
        default="sorted",
    )
    parser.add_argument(
        "--prefill-placement-policy",
        choices=["all_gpu", "all_cpu", "utility"],
        default="all_cpu",
    )
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-qa-per-dataset", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--blend-special-str", type=str, default="# #")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--warmup-query-text", type=str, default="Warm up the engine.")
    parser.add_argument(
        "--prefill-query-text",
        type=str,
        default="Warm up this fragment for later QA use.",
    )
    parser.add_argument("--blend-check-layers", type=str, default="1")
    parser.add_argument("--blend-recompute-ratios", type=str, default="0.15")
    parser.add_argument("--blend-connector-impl", type=str, default="fast")
    parser.add_argument(
        "--blend-internal-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--blend-pipeline-buffers", type=int, default=3)
    parser.add_argument("--blend-buffer-bucket-tokens", type=int, default=256)
    parser.add_argument("--blend-max-cached-buffer-packs", type=int, default=4)
    parser.add_argument(
        "--online-execution-policy",
        choices=["fixed", "utility"],
        default="fixed",
    )
    parser.add_argument(
        "--fragment-management-policy",
        choices=["static", "utility"],
        default="static",
    )
    parser.add_argument(
        "--online-execution-mode",
        choices=[EXECUTION_MODE_BLEND, EXECUTION_MODE_NATIVE_VLLM],
        default=EXECUTION_MODE_BLEND,
    )
    parser.add_argument(
        "--blend-engine-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--startup-timeout-s", type=float, default=240.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--utility-cost-model",
        type=str,
        default='{"recompute_ms_per_token":0.105,"transfer_gib_per_s":12.0}',
    )
    parser.add_argument("--utility-tail-lambda", type=float, default=0.0)
    parser.add_argument("--utility-gpu-penalty-ms", type=float, default=0.0)
    parser.add_argument(
        "--utility-enable-semantic-hints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--online-use-historical-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--utility-enable-cpu-recompute",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--utility-fallback-margin-ms", type=float, default=0.0)
    parser.add_argument("--utility-prefix-history-window", type=int, default=0)
    parser.add_argument("--online-gap-alpha", type=float, default=0.5)
    parser.add_argument(
        "--utility-native-runtime",
        choices=["recompute"],
        default="recompute",
    )
    parser.add_argument(
        "--initial-fill-mode",
        choices=["none", "gpu_then_cpu", "cpu_then_gpu"],
        default="cpu_then_gpu",
    )
    parser.add_argument(
        "--admit-misses-to",
        choices=["auto", "cpu", "gpu", "none"],
        default="auto",
    )
    parser.add_argument("--gpu-lookahead", type=int, default=0)
    parser.add_argument(
        "--methods",
        type=str,
        default="no_prefix,native_prefix,explicit_blend",
        help=(
            "Comma-separated methods to run. Choices: "
            "no_prefix,native_prefix,explicit_blend."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and summarize the workload only, without launching any "
            "model runtime."
        ),
    )
    return parser.parse_args()


def ensure_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = (
            Path(__file__).resolve().parent
            / "analysis_results"
            / f"explicit_fragment_{args.workload_kind}_{stamp}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not methods:
        raise ValueError("methods cannot be empty.")

    invalid = [item for item in methods if item not in METHOD_CHOICES]
    if invalid:
        raise ValueError(
            f"Unsupported methods={invalid!r}; expected choices={METHOD_CHOICES!r}"
        )

    seen: set[str] = set()
    ordered: list[str] = []
    for method in methods:
        if method in seen:
            continue
        seen.add(method)
        ordered.append(method)
    return ordered


def build_utility_cfg(args: argparse.Namespace) -> OnlineUtilityPlannerConfig:
    return OnlineUtilityPlannerConfig.from_specs(
        cost_model_spec=str(args.utility_cost_model or ""),
        tail_lambda=float(args.utility_tail_lambda or 0.0),
        enable_cpu_recompute=bool(args.utility_enable_cpu_recompute),
        native_runtime="recompute",
        hybrid_gate_all_cached_override=False,
    )


def build_fragment_management_cfg(
    args: argparse.Namespace,
) -> MaintenanceUtilityPlannerConfig:
    return MaintenanceUtilityPlannerConfig.from_args(
        utility_cost_model=str(args.utility_cost_model or ""),
        tail_lambda=float(args.utility_tail_lambda or 0.0),
        enable_semantic_hints=bool(args.utility_enable_semantic_hints),
        gpu_then_cpu_penalty_ms=float(args.utility_gpu_penalty_ms or 0.0),
    )


def decide_online_execution_mode(
    *,
    args: argparse.Namespace,
    record: Any,
    unique_chunks: dict[str, dict[str, Any]],
    snapshot: Any,
    utility_cfg: OnlineUtilityPlannerConfig | None,
    current_query_idx: int | None = None,
) -> tuple[str, dict[str, Any]]:
    if str(args.online_execution_policy) != "utility" or utility_cfg is None:
        return str(args.online_execution_mode), {
            "planner_choice": (
                "recompute"
                if str(args.online_execution_mode) == EXECUTION_MODE_NATIVE_VLLM
                else "blend"
            ),
            "policy": "fixed",
            "runtime_execution_mode": str(args.online_execution_mode),
            "recompute_utility_ms": 0.0,
            "cacheblend_utility_ms": 0.0,
            "decision_gap_ms": 0.0,
        }

    fragment_ids = list(record.chunk_ids)
    locations = [str(snapshot.locations.get(chunk_id, "miss")) for chunk_id in fragment_ids]
    planner_mode, debug = choose_execution_mode(
        fragment_ids=fragment_ids,
        fragments=unique_chunks,
        locations=locations,
        prefix_reuse_tokens=0,
        fallback_margin_ms=float(args.utility_fallback_margin_ms or 0.0),
        cfg=utility_cfg,
        current_query_idx=current_query_idx,
    )
    mode = (
        EXECUTION_MODE_NATIVE_VLLM
        if str(planner_mode) == "native_prefix"
        else EXECUTION_MODE_BLEND
    )
    return mode, {
        "planner_choice": "recompute" if mode == EXECUTION_MODE_NATIVE_VLLM else "blend",
        "policy": "utility",
        "runtime_execution_mode": str(mode),
        "recompute_utility_ms": float(
            debug.get("native_prefix_utility_ms", 0.0) or 0.0
        ),
        "cacheblend_utility_ms": float(
            debug.get("cacheblend_utility_ms", 0.0) or 0.0
        ),
        "cacheblend_effective_utility_ms": float(
            debug.get("cacheblend_effective_utility_ms", 0.0) or 0.0
        ),
        "predicted_native_ttft_ms": float(
            debug.get("predicted_native_ttft_ms", 0.0) or 0.0
        ),
        "predicted_blend_ttft_ms": float(
            debug.get("predicted_blend_ttft_ms", 0.0) or 0.0
        ),
        "decision_gap_ms": float(debug.get("decision_gap_ms", 0.0) or 0.0),
        "effective_fallback_margin_ms": float(
            debug.get("effective_fallback_margin_ms", 0.0) or 0.0
        ),
        "skills_bootstrap_bonus_ms": float(
            debug.get("skills_bootstrap_bonus_ms", 0.0) or 0.0
        ),
        "skills_miss_blend_bonus_ms": float(
            debug.get("skills_miss_blend_bonus_ms", 0.0) or 0.0
        ),
        "fragmented_miss_fastpath_bonus_ms": float(
            debug.get("fragmented_miss_fastpath_bonus_ms", 0.0) or 0.0
        ),
        "predicted_regret_ms": float(debug.get("predicted_regret_ms", 0.0) or 0.0),
        "cacheblend_fetch_actions": list(debug.get("cacheblend_fetch_actions", [])),
        "prefix_reuse_disabled": True,
    }


@contextmanager
def build_lmcache_llm(
    *,
    model: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    enable_prefix_caching: bool,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
) -> Iterator[LLM]:
    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        enforce_eager=True,
        enable_prefix_caching=enable_prefix_caching,
        kv_transfer_config=kv_transfer_config,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
        del llm


@contextmanager
def build_plain_llm_limited(
    *,
    model: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    enable_prefix_caching: bool,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
) -> Iterator[LLM]:
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        enforce_eager=True,
        enable_prefix_caching=enable_prefix_caching,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    try:
        yield llm
    finally:
        del llm


def measure_request(
    llm: Any,
    *,
    prompt_ids: list[int],
    sampling_params: SamplingParams,
) -> RequestMeasurement:
    start = time.perf_counter()
    outputs = llm.generate(
        prompts={"prompt_token_ids": list(prompt_ids)},
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    wall_s = time.perf_counter() - start

    output = outputs[0]
    metrics = output.metrics
    ttft_s = None
    if metrics is not None and getattr(metrics, "first_token_time", None) is not None:
        ttft_s = float(metrics.first_token_time - metrics.arrival_time)
    elif int(getattr(sampling_params, "max_tokens", 0) or 0) == 1:
        ttft_s = float(wall_s)

    request_id = getattr(output, "request_id", None)
    if request_id is None and metrics is not None:
        request_id = getattr(metrics, "request_id", None)

    cached_tokens = getattr(output, "num_cached_tokens", None)
    if cached_tokens is None:
        cached_tokens = getattr(output, "cached_tokens", None)

    generated_text = ""
    if getattr(output, "outputs", None):
        generated_text = str(output.outputs[0].text)

    return RequestMeasurement(
        request_id=str(request_id) if request_id is not None else None,
        ttft_s=ttft_s,
        wall_s=float(wall_s),
        prompt_tokens=len(prompt_ids),
        cached_tokens=(int(cached_tokens) if cached_tokens is not None else None),
        generated_text=generated_text,
    )


def run_plain_workload(
    *,
    args: argparse.Namespace,
    warmup_prompt_ids: list[int],
    query_records: list[Any],
    sampling_params: SamplingParams,
    enable_prefix_caching: bool,
) -> tuple[list[RequestMeasurement], LatencySummary]:
    measurements: list[RequestMeasurement] = []
    with build_plain_llm_limited(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_prefix_caching=enable_prefix_caching,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    ) as llm:
        measure_request(
            llm,
            prompt_ids=warmup_prompt_ids,
            sampling_params=sampling_params,
        )
        for record in query_records:
            measurements.append(
                measure_request(
                    llm,
                    prompt_ids=list(record.prompt_ids),
                    sampling_params=sampling_params,
                )
            )

    summary = summarize_measurements(
        measurements=measurements,
        prompt_tokens=[record.prompt_tokens for record in query_records],
    )
    return measurements, summary


def build_blend_env(*, args: argparse.Namespace, phase_file: Path) -> dict[str, str]:
    env = {
        "LMCACHE_CHUNK_SIZE": str(args.chunk_size),
        "LMCACHE_ENABLE_BLENDING": "True",
        "LMCACHE_BLEND_SPECIAL_STR": str(args.blend_special_str),
        "LMCACHE_SAVE_UNFULL_CHUNK": "True",
        "LMCACHE_USE_LAYERWISE": "True",
        "LMCACHE_BLEND_CHECK_LAYERS": str(args.blend_check_layers),
        "LMCACHE_BLEND_RECOMPUTE_RATIOS": str(args.blend_recompute_ratios),
        "LMCACHE_PHASE_TIMING_PATH": str(phase_file),
        "LMCACHE_LOCAL_CPU": "True" if float(args.max_local_cpu_size) > 0 else "False",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": str(args.max_local_cpu_size),
        "LMCACHE_LOCAL_GPU": "True" if float(args.max_local_gpu_size) > 0 else "False",
        "LMCACHE_MAX_LOCAL_GPU_SIZE": str(args.max_local_gpu_size),
    }

    extra_config = {
        "blend_internal_timing": bool(args.blend_internal_timing),
        "blend_pipeline_buffers": int(args.blend_pipeline_buffers),
        "blend_buffer_bucket_tokens": int(args.blend_buffer_bucket_tokens),
        "blend_max_cached_buffer_packs": int(args.blend_max_cached_buffer_packs),
    }
    if args.blend_connector_impl:
        extra_config["blend_connector_impl"] = str(args.blend_connector_impl)
    env["LMCACHE_EXTRA_CONFIG"] = json.dumps(extra_config)
    return env


def summarize_maintenance(stats_list: list[MaintenanceStats]) -> MaintenanceSummary:
    if not stats_list:
        return MaintenanceSummary(
            request_count=0,
            total_wall_s=0.0,
            mean_wall_s=0.0,
            total_materialize_s=0.0,
            total_move_s=0.0,
            total_evict_s=0.0,
            total_admitted_chunks=0,
            total_promoted_chunks=0,
            total_demoted_chunks=0,
            total_evicted_chunks=0,
            total_admitted_tokens=0,
            total_promoted_tokens=0,
            total_demoted_tokens=0,
            total_evicted_tokens=0,
        )

    total_wall_s = float(sum(item.wall_s for item in stats_list))
    return MaintenanceSummary(
        request_count=len(stats_list),
        total_wall_s=total_wall_s,
        mean_wall_s=total_wall_s / len(stats_list),
        total_materialize_s=float(sum(item.materialize_s for item in stats_list)),
        total_move_s=float(sum(item.move_s for item in stats_list)),
        total_evict_s=float(sum(item.evict_s for item in stats_list)),
        total_admitted_chunks=int(sum(item.admitted_chunks for item in stats_list)),
        total_promoted_chunks=int(sum(item.promoted_chunks for item in stats_list)),
        total_demoted_chunks=int(sum(item.demoted_chunks for item in stats_list)),
        total_evicted_chunks=int(sum(item.evicted_chunks for item in stats_list)),
        total_admitted_tokens=int(sum(item.admitted_tokens for item in stats_list)),
        total_promoted_tokens=int(sum(item.promoted_tokens for item in stats_list)),
        total_demoted_tokens=int(sum(item.demoted_tokens for item in stats_list)),
        total_evicted_tokens=int(sum(item.evicted_tokens for item in stats_list)),
    )


def seed_pool_from_prefill_plan(
    *,
    pool: ExplicitFragmentPool,
    prefill_order: list[str],
    prefill_plan: dict[str, dict[str, Any]],
) -> MaintenanceStats:
    stats = MaintenanceStats()
    order_index = {chunk_id: idx for idx, chunk_id in enumerate(prefill_order)}

    def _seed_priority(chunk_id: str) -> tuple[int, int, float, int]:
        plan_item = dict(prefill_plan.get(chunk_id) or {})
        if not bool(plan_item.get("enabled", False)):
            return (0, 0, 0.0, -order_index.get(chunk_id, 0))
        target_location = str(plan_item.get("target_location") or "")
        gpu_rank = 1 if target_location == "LocalGPUBackend" else 0
        try:
            utility = float(plan_item.get("utility") or 0.0)
        except Exception:
            utility = 0.0
        return (1, gpu_rank, utility, -order_index.get(chunk_id, 0))

    ordered_chunk_ids = sorted(
        list(prefill_order),
        key=_seed_priority,
        reverse=True,
    )

    for chunk_id in ordered_chunk_ids:
        plan_item = dict(prefill_plan.get(chunk_id) or {})
        if not bool(plan_item.get("enabled", False)):
            continue
        target_location = str(plan_item.get("target_location") or "")
        if target_location == "LocalGPUBackend":
            target_tier = "gpu"
        elif target_location == "LocalCPUBackend":
            target_tier = "cpu"
        else:
            continue
        stats.absorb(pool.materialize_or_move(chunk_id, target_tier=target_tier))
    return stats


def phase_duration(phases: dict[str, dict[str, Any]], phase_name: str) -> float:
    record = phases.get(phase_name)
    if record is None:
        return 0.0
    try:
        return float(record.get("duration_s", 0.0))
    except (TypeError, ValueError):
        return 0.0


def build_query_rows(
    *,
    query_records: list[Any],
    measurements: list[RequestMeasurement],
    shadow_rows: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
    maintenance_rows: list[MaintenanceStats],
    phase_index: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (record, measurement, shadow, execution, maintenance) in enumerate(
        zip(
            query_records,
            measurements,
            shadow_rows,
            execution_rows,
            maintenance_rows,
            strict=True,
        )
    ):
        phases = phase_index.get(str(shadow.get("normalized_request_id", "")), {})
        rows.append(
            {
                "query_index": idx,
                "dataset": record.dataset,
                "qa_index": record.qa_index,
                "sample_id": record.sample_id,
                "request_id": measurement.request_id,
                "prompt_tokens": record.prompt_tokens,
                "num_fragments": len(record.chunk_ids),
                "runtime_cached_tokens": measurement.cached_tokens,
                "shadow_hit_tokens": shadow["shadow_hit_tokens"],
                "shadow_gpu_hit_tokens": shadow["shadow_gpu_hit_tokens"],
                "shadow_cpu_hit_tokens": shadow["shadow_cpu_hit_tokens"],
                "shadow_miss_tokens": shadow["shadow_miss_tokens"],
                "shadow_hit_fragments": shadow["shadow_hit_fragments"],
                "shadow_missed_fragments": shadow["shadow_missed_fragments"],
                "execution_mode": execution["execution_mode"],
                "planner_choice": execution.get("planner_choice"),
                "predicted_recompute_utility_ms": execution.get(
                    "predicted_recompute_utility_ms", 0.0
                ),
                "predicted_cacheblend_utility_ms": execution.get(
                    "predicted_cacheblend_utility_ms", 0.0
                ),
                "predicted_native_ttft_ms": execution.get(
                    "predicted_native_ttft_ms", 0.0
                ),
                "predicted_blend_ttft_ms": execution.get(
                    "predicted_blend_ttft_ms", 0.0
                ),
                "predicted_decision_gap_ms": execution.get(
                    "predicted_decision_gap_ms", 0.0
                ),
                "effective_fallback_margin_ms": execution.get(
                    "effective_fallback_margin_ms", 0.0
                ),
                "hybrid_gate_penalty_ms": execution.get(
                    "hybrid_gate_penalty_ms", 0.0
                ),
                "skills_miss_blend_bonus_ms": execution.get(
                    "skills_miss_blend_bonus_ms", 0.0
                ),
                "fragmented_miss_fastpath_bonus_ms": execution.get(
                    "fragmented_miss_fastpath_bonus_ms", 0.0
                ),
                "ttft_s": measurement.ttft_s,
                "wall_s": measurement.wall_s,
                "maintenance_wall_s": maintenance.wall_s,
                "maintenance_materialize_s": maintenance.materialize_s,
                "maintenance_move_s": maintenance.move_s,
                "maintenance_evict_s": maintenance.evict_s,
                "maintenance_admitted_chunks": maintenance.admitted_chunks,
                "maintenance_promoted_chunks": maintenance.promoted_chunks,
                "maintenance_demoted_chunks": maintenance.demoted_chunks,
                "maintenance_evicted_chunks": maintenance.evicted_chunks,
                "lookup_total_s": phase_duration(phases, "lookup_total_s"),
                "lookup_mode": (
                    phases.get("lookup_total_s", {}).get("lookup_mode")
                    if phases.get("lookup_total_s") is not None
                    else None
                ),
                "retrieve_total_s": phase_duration(phases, "retrieve_total_s"),
                "blend_total_s": phase_duration(phases, "blend_total_s"),
                "to_gpu_total_s": phase_duration(phases, "to_gpu_total_s"),
                "retrieve_prepare_s": phase_duration(phases, "retrieve_prepare_s"),
                "retrieve_storage_wait_s": phase_duration(
                    phases, "retrieve_storage_wait_s"
                ),
                "retrieve_gpu_send_s": phase_duration(phases, "retrieve_gpu_send_s"),
            }
        )
    return rows


def mean_or_zero(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def run_explicit_blend(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    workload: dict[str, Any],
    sampling_params: SamplingParams,
) -> ExplicitBlendSummary:
    mod = blend_cpu_bench_module()
    mod.patch_blend_model_registration()

    query_records = list(workload["query_records"])
    prefill_records = {
        record.chunk_id: record
        for record in workload.get("all_prefill_records", workload["prefill_records"])
    }
    fragment_tokens = {
        chunk_id: int(meta["tokens"])
        for chunk_id, meta in workload["unique_chunks"].items()
    }

    cpu_budget_tokens = gib_to_token_budget(float(args.max_local_cpu_size))
    gpu_budget_tokens = gib_to_token_budget(float(args.max_local_gpu_size))
    phase_file = out_dir / "explicit_blend_phase.jsonl"
    if phase_file.exists():
        phase_file.unlink()

    env = build_blend_env(args=args, phase_file=phase_file)
    initial_fill_mode = str(args.initial_fill_mode)
    online_utility_cfg = (
        build_utility_cfg(args)
        if str(args.online_execution_policy) == "utility"
        else None
    )
    fragment_management_cfg = (
        build_fragment_management_cfg(args)
        if str(args.fragment_management_policy) == "utility"
        else None
    )
    engine_prefix_caching = bool(args.blend_engine_enable_prefix_caching)

    with mod.temporary_environ(env):
        with build_lmcache_llm(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            enable_prefix_caching=engine_prefix_caching,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
        ) as llm:
            runtime = LMCacheFragmentRuntime(llm, sampling_params)
            pool = ExplicitFragmentPool(
                runtime=runtime,
                prefill_records=prefill_records,
                fragment_tokens=fragment_tokens,
                fragments=dict(workload["unique_chunks"]),
                cpu_budget_tokens=cpu_budget_tokens,
                gpu_budget_tokens=gpu_budget_tokens,
                cpu_enabled=float(args.max_local_cpu_size) > 0,
                gpu_enabled=float(args.max_local_gpu_size) > 0,
                admit_misses_to=args.admit_misses_to,
                gpu_lookahead=int(args.gpu_lookahead),
                fragment_management_policy=str(args.fragment_management_policy),
                maintenance_cfg=fragment_management_cfg,
            )

            initial_stats = MaintenanceStats()
            if str(args.prefill_placement_policy) == "utility":
                initial_stats = seed_pool_from_prefill_plan(
                    pool=pool,
                    prefill_order=list(workload["prefill_order"]),
                    prefill_plan=dict(workload["prefill_plan"]),
                )
            elif initial_fill_mode == "gpu_then_cpu":
                initial_stats = pool.seed_initial_resident_pool(
                    chunk_order=list(workload["prefill_order"]),
                    tier_order=["gpu", "cpu"],
                )
            elif initial_fill_mode == "cpu_then_gpu":
                initial_stats = pool.seed_initial_resident_pool(
                    chunk_order=list(workload["prefill_order"]),
                    tier_order=["cpu", "gpu"],
                )

            warmup_execution_mode = (
                str(args.online_execution_mode)
                if str(args.online_execution_policy) == "fixed"
                else EXECUTION_MODE_NATIVE_VLLM
            )
            warmup_sampling_params = runtime.make_sampling_params(
                request_kind="online_blended_query",
                skip_save=True,
                execution_mode=warmup_execution_mode,
            )
            measure_request(
                llm,
                prompt_ids=list(workload["warmup_prompt_ids"]),
                sampling_params=warmup_sampling_params,
            )

            measurements: list[RequestMeasurement] = []
            shadow_rows: list[dict[str, Any]] = []
            execution_rows: list[dict[str, Any]] = []
            maintenance_rows: list[MaintenanceStats] = []
            online_stats = (
                OnlineSemanticStats(gap_alpha=float(args.online_gap_alpha or 0.5))
                if bool(args.online_use_historical_stats)
                else None
            )

            for index, record in enumerate(query_records):
                if online_stats is not None:
                    online_stats.inject_chunk_hints(
                        unique_chunks=workload["unique_chunks"],
                        current_query_idx=index,
                    )
                pool.reconcile_shadow_locations(list(record.chunk_ids))
                snapshot = pool.capture_query_reuse(record.chunk_ids)
                execution_mode, execution_debug = decide_online_execution_mode(
                    args=args,
                    record=record,
                    unique_chunks=workload["unique_chunks"],
                    snapshot=snapshot,
                    utility_cfg=online_utility_cfg,
                    current_query_idx=index,
                )
                query_sampling_params = runtime.make_sampling_params(
                    request_kind="online_blended_query",
                    skip_save=True,
                    execution_mode=str(execution_mode),
                )
                measurement = measure_request(
                    llm,
                    prompt_ids=list(record.prompt_ids),
                    sampling_params=query_sampling_params,
                )
                measurements.append(measurement)
                execution_rows.append(
                    {
                        "execution_mode": str(execution_mode),
                        "planner_choice": str(
                            execution_debug.get("planner_choice", "blend")
                        ),
                        "predicted_recompute_utility_ms": float(
                            execution_debug.get("recompute_utility_ms", 0.0) or 0.0
                        ),
                        "predicted_cacheblend_utility_ms": float(
                            execution_debug.get("cacheblend_utility_ms", 0.0) or 0.0
                        ),
                        "predicted_native_ttft_ms": float(
                            execution_debug.get("predicted_native_ttft_ms", 0.0)
                            or 0.0
                        ),
                        "predicted_blend_ttft_ms": float(
                            execution_debug.get("predicted_blend_ttft_ms", 0.0)
                            or 0.0
                        ),
                        "predicted_decision_gap_ms": float(
                            execution_debug.get("decision_gap_ms", 0.0) or 0.0
                        ),
                        "effective_fallback_margin_ms": float(
                            execution_debug.get("effective_fallback_margin_ms", 0.0)
                            or 0.0
                        ),
                        "hybrid_gate_penalty_ms": float(
                            execution_debug.get("hybrid_gate_penalty_ms", 0.0)
                            or 0.0
                        ),
                        "skills_miss_blend_bonus_ms": float(
                            execution_debug.get("skills_miss_blend_bonus_ms", 0.0)
                            or 0.0
                        ),
                        "fragmented_miss_fastpath_bonus_ms": float(
                            execution_debug.get(
                                "fragmented_miss_fastpath_bonus_ms", 0.0
                            )
                            or 0.0
                        ),
                    }
                )

                if online_stats is not None:
                    online_stats.observe_query(
                        unique_chunks=workload["unique_chunks"],
                        chunk_ids=list(record.chunk_ids),
                        query_idx=index,
                        locations=snapshot.locations,
                    )
                    # Maintenance should consume the current query's access evidence,
                    # otherwise just-used fragments are judged with stale history.
                    online_stats.inject_chunk_hints(
                        unique_chunks=workload["unique_chunks"],
                        current_query_idx=index,
                    )

                next_chunk_ids = None
                if index + 1 < len(query_records) and int(args.gpu_lookahead) > 0:
                    next_chunk_ids = list(query_records[index + 1].chunk_ids)
                maintenance = pool.run_inter_query_maintenance(
                    current_chunk_ids=list(record.chunk_ids),
                    next_chunk_ids=next_chunk_ids,
                )
                maintenance_rows.append(maintenance)

                normalized_request_id = ""
                if measurement.request_id is not None:
                    normalized_request_id = normalize_request_id(
                        str(measurement.request_id)
                    )
                shadow_rows.append(
                    {
                        "normalized_request_id": normalized_request_id,
                        "shadow_hit_tokens": snapshot.hit_tokens,
                        "shadow_gpu_hit_tokens": snapshot.gpu_hit_tokens,
                        "shadow_cpu_hit_tokens": snapshot.cpu_hit_tokens,
                        "shadow_miss_tokens": snapshot.miss_tokens,
                        "shadow_hit_fragments": snapshot.hit_fragments,
                        "shadow_missed_fragments": len(snapshot.missed_fragments),
                    }
                )

    online = summarize_measurements(
        measurements=measurements,
        prompt_tokens=[record.prompt_tokens for record in query_records],
    )
    online_phase_breakdown = summarize_online_phase_breakdown(
        phase_file=phase_file,
        measurements=measurements,
    )
    phase_index = (
        build_phase_index(load_phase_records(phase_file)) if phase_file.exists() else {}
    )
    query_rows = build_query_rows(
        query_records=query_records,
        measurements=measurements,
        shadow_rows=shadow_rows,
        execution_rows=execution_rows,
        maintenance_rows=maintenance_rows,
        phase_index=phase_index,
    )
    recompute_requests = sum(
        1
        for row in execution_rows
        if row["execution_mode"] == EXECUTION_MODE_NATIVE_VLLM
    )
    blend_requests = sum(
        1 for row in execution_rows if row["execution_mode"] == EXECUTION_MODE_BLEND
    )

    return ExplicitBlendSummary(
        online=online,
        online_phase_breakdown=online_phase_breakdown,
        maintenance=summarize_maintenance(maintenance_rows),
        initial_materialization=summarize_maintenance([initial_stats]),
        cpu_budget_tokens=cpu_budget_tokens,
        gpu_budget_tokens=gpu_budget_tokens,
        cpu_budget_gib=float(args.max_local_cpu_size),
        gpu_budget_gib=float(args.max_local_gpu_size),
        initial_fill_mode=initial_fill_mode,
        admit_misses_to=str(args.admit_misses_to),
        gpu_lookahead=int(args.gpu_lookahead),
        fragment_management_policy=str(args.fragment_management_policy),
        online_execution_policy=str(args.online_execution_policy),
        online_execution_mode=str(args.online_execution_mode),
        blend_engine_prefix_caching=bool(engine_prefix_caching),
        utility_enable_semantic_hints=bool(args.utility_enable_semantic_hints),
        online_use_historical_stats=bool(args.online_use_historical_stats),
        utility_mode_recompute_requests=int(recompute_requests),
        utility_mode_blend_requests=int(blend_requests),
        mean_predicted_recompute_utility_ms=mean_or_zero(
            [row["predicted_recompute_utility_ms"] for row in execution_rows]
        ),
        mean_predicted_cacheblend_utility_ms=mean_or_zero(
            [row["predicted_cacheblend_utility_ms"] for row in execution_rows]
        ),
        mean_predicted_decision_gap_ms=mean_or_zero(
            [row["predicted_decision_gap_ms"] for row in execution_rows]
        ),
        resident_pool_end=pool.resident_counts(),
        phase_file=str(phase_file),
        mean_shadow_hit_tokens=mean_or_zero(
            [row["shadow_hit_tokens"] for row in shadow_rows]
        ),
        mean_shadow_gpu_hit_tokens=mean_or_zero(
            [row["shadow_gpu_hit_tokens"] for row in shadow_rows]
        ),
        mean_shadow_cpu_hit_tokens=mean_or_zero(
            [row["shadow_cpu_hit_tokens"] for row in shadow_rows]
        ),
        mean_shadow_miss_tokens=mean_or_zero(
            [row["shadow_miss_tokens"] for row in shadow_rows]
        ),
        mean_shadow_hit_fragments=mean_or_zero(
            [row["shadow_hit_fragments"] for row in shadow_rows]
        ),
        mean_shadow_missed_fragments=mean_or_zero(
            [row["shadow_missed_fragments"] for row in shadow_rows]
        ),
        query_rows=query_rows,
    )


def print_latency_summary(name: str, summary: LatencySummary) -> None:
    print(
        f"{name}: mean_ttft={summary.mean_ttft_s:.4f}s "
        f"p90_ttft={summary.p90_ttft_s:.4f}s "
        f"mean_cached_tokens={summary.mean_cached_tokens if summary.mean_cached_tokens is not None else 'n/a'} "
        f"hit_rate={summary.cache_hit_rate:.3f}"
    )


def print_maintenance_summary(name: str, summary: MaintenanceSummary) -> None:
    print(
        f"{name}: total_wall={summary.total_wall_s:.4f}s "
        f"mean_wall={summary.mean_wall_s:.4f}s "
        f"materialize={summary.total_materialize_s:.4f}s "
        f"move={summary.total_move_s:.4f}s "
        f"evict={summary.total_evict_s:.4f}s"
    )


def _round_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _sorted_count_dict(counts: dict[str, int]) -> dict[str, int]:
    return {key: int(counts[key]) for key in sorted(counts)}


def _canonical_prefill_bucket(meta: dict[str, Any]) -> str:
    target_location = str(meta.get("target_location") or "")
    if target_location == "LocalGPUBackend":
        return "gpu"
    if target_location == "LocalCPUBackend":
        return "cpu"
    return "drop"


def _summarize_canonical_bucket(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    kind_counts: dict[str, int] = {}
    layer_counts: dict[str, int] = {}
    semantic_group_counts: dict[str, int] = {}
    importance_scores: list[float] = []
    stability_scores: list[float] = []
    sharedness_scores: list[float] = []
    relation_degrees: list[float] = []
    retention_priors: list[float] = []
    affinity_scores: list[float] = []
    future_use_reference_counts: list[float] = []
    next_use_window_queries: list[float] = []

    for chunk in chunks:
        semantics = chunk.get("canonical_semantics") or {}
        kind = str(semantics.get("kind") or chunk.get("type") or "unknown")
        layer = str(semantics.get("layer") or chunk.get("layer") or "unknown")
        kind_counts[kind] = int(kind_counts.get(kind, 0) + 1)
        layer_counts[layer] = int(layer_counts.get(layer, 0) + 1)

        for group in semantics.get("semantic_groups") or []:
            group_name = str(group).strip()
            if group_name:
                semantic_group_counts[group_name] = int(
                    semantic_group_counts.get(group_name, 0) + 1
                )

        importance_scores.append(float(semantics.get("importance_score", 0.0)))
        stability_scores.append(float(semantics.get("stability_score", 0.0)))
        sharedness_scores.append(float(semantics.get("sharedness_score", 0.0)))
        relation_degrees.append(float(semantics.get("relation_degree", 0.0)))
        retention_priors.append(float(semantics.get("retention_prior", 0.0)))
        affinity_scores.append(float(semantics.get("affinity_score", 0.0)))
        future_use_reference_counts.append(
            float(semantics.get("future_use_reference_count", 0.0))
        )
        next_use_window_queries.append(
            float(semantics.get("next_use_window_queries", 0.0))
        )

    summary = {
        "fragment_count": len(chunks),
        "kind_counts": _sorted_count_dict(kind_counts),
        "layer_counts": _sorted_count_dict(layer_counts),
        "mean_importance_score": _round_float_or_none(_mean_or_none(importance_scores)),
        "mean_stability_score": _round_float_or_none(_mean_or_none(stability_scores)),
        "mean_sharedness_score": _round_float_or_none(_mean_or_none(sharedness_scores)),
        "mean_relation_degree": _round_float_or_none(_mean_or_none(relation_degrees)),
        "mean_retention_prior": _round_float_or_none(_mean_or_none(retention_priors)),
        "mean_affinity_score": _round_float_or_none(_mean_or_none(affinity_scores)),
        "mean_future_use_reference_count": _round_float_or_none(
            _mean_or_none(future_use_reference_counts)
        ),
        "mean_next_use_window_queries": _round_float_or_none(
            _mean_or_none(next_use_window_queries)
        ),
    }
    if semantic_group_counts:
        summary["semantic_group_counts"] = _sorted_count_dict(semantic_group_counts)
    return summary


def summarize_canonical_semantics(
    *,
    unique_chunks: dict[str, dict[str, Any]],
    prefill_plan: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    version_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    bucket_chunks: dict[str, list[dict[str, Any]]] = {
        "all": [],
        "gpu": [],
        "cpu": [],
        "drop": [],
    }
    missing_count = 0

    for chunk_id, chunk in unique_chunks.items():
        semantics = chunk.get("canonical_semantics")
        if not isinstance(semantics, dict) or not semantics:
            missing_count += 1
            continue

        version_key = str(semantics.get("version") or "unknown")
        family_key = str(semantics.get("workload_family") or "unknown")
        version_counts[version_key] = int(version_counts.get(version_key, 0) + 1)
        family_counts[family_key] = int(family_counts.get(family_key, 0) + 1)

        bucket_chunks["all"].append(chunk)
        bucket = _canonical_prefill_bucket(prefill_plan.get(chunk_id) or {})
        bucket_chunks[bucket].append(chunk)

    return {
        "fragments_with_canonical_semantics": len(bucket_chunks["all"]),
        "fragments_without_canonical_semantics": int(missing_count),
        "canonical_version_counts": _sorted_count_dict(version_counts),
        "canonical_workload_family_counts": _sorted_count_dict(family_counts),
        "all_fragments": _summarize_canonical_bucket(bucket_chunks["all"]),
        "prefill_targets": {
            "gpu": _summarize_canonical_bucket(bucket_chunks["gpu"]),
            "cpu": _summarize_canonical_bucket(bucket_chunks["cpu"]),
            "drop": _summarize_canonical_bucket(bucket_chunks["drop"]),
        },
    }


def print_dry_run_summary(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    methods: list[str],
    workload: dict[str, Any],
) -> None:
    stats = workload["stats"]
    prefill_records = list(workload["prefill_records"])
    unique_chunks = workload["unique_chunks"]
    prefill_plan = workload.get("prefill_plan") or {}

    plan_counts: dict[str, int] = {}
    for record in prefill_records:
        location = str(getattr(record, "store_location", None) or "drop")
        plan_counts[location] = int(plan_counts.get(location, 0) + 1)

    summary = {
        "workload_kind": str(args.workload_kind),
        "datasets": list(workload["datasets"]),
        "methods": list(methods),
        "query_count": int(stats.query_count),
        "unique_fragments": int(stats.unique_fragments),
        "mean_prompt_tokens": float(stats.mean_prompt_tokens),
        "p90_prompt_tokens": float(stats.p90_prompt_tokens),
        "max_prompt_tokens": int(stats.max_prompt_tokens),
        "mean_memory_prefix_tokens": float(stats.mean_memory_prefix_tokens),
        "p90_memory_prefix_tokens": float(stats.p90_memory_prefix_tokens),
        "max_memory_prefix_tokens": int(stats.max_memory_prefix_tokens),
        "prefill_record_count": len(prefill_records),
        "prefill_plan_counts": plan_counts,
        "prefill_plan_summary": summarize_prefill_plan(
            unique_chunks=unique_chunks,
            prefill_plan=prefill_plan,
        ),
        "canonical_semantics_summary": summarize_canonical_semantics(
            unique_chunks=unique_chunks,
            prefill_plan=prefill_plan,
        ),
        "out_dir": str(out_dir),
    }

    output_path = out_dir / "dry_run_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("dry_run: true")
    print(f"workload_kind: {summary['workload_kind']}")
    print(f"datasets: {','.join(summary['datasets'])}")
    print(f"methods: {','.join(summary['methods'])}")
    print(
        "workload: "
        f"queries={summary['query_count']} "
        f"unique_fragments={summary['unique_fragments']} "
        f"mean_prompt_tokens={summary['mean_prompt_tokens']:.1f} "
        f"p90_prompt_tokens={summary['p90_prompt_tokens']:.1f} "
        f"max_prompt_tokens={summary['max_prompt_tokens']}"
    )
    print(
        "memory_prefix: "
        f"mean={summary['mean_memory_prefix_tokens']:.1f} "
        f"p90={summary['p90_memory_prefix_tokens']:.1f} "
        f"max={summary['max_memory_prefix_tokens']}"
    )
    print(
        "prefill: "
        f"records={summary['prefill_record_count']} "
        f"placement_counts={summary['prefill_plan_counts']}"
    )
    canonical_summary = summary["canonical_semantics_summary"]
    all_fragments = canonical_summary["all_fragments"]
    print(
        "canonical: "
        f"fragments={canonical_summary['fragments_with_canonical_semantics']} "
        f"retention={all_fragments['mean_retention_prior']:.4f} "
        f"affinity={all_fragments['mean_affinity_score']:.4f}"
    )
    print(f"dry_run_json: {output_path}")


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args)
    methods = parse_methods(str(args.methods))
    sampling_params = build_sampling_params(int(args.max_tokens))
    runtime_env = {"CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices)}

    with blend_cpu_bench_module().temporary_environ(runtime_env):
        workload = build_workload_for_probe(args)
        if bool(args.online_use_historical_stats):
            strip_past_only_runtime_hints(
                unique_chunks=workload["unique_chunks"],
            )
        if bool(args.dry_run):
            print_dry_run_summary(
                out_dir=out_dir,
                args=args,
                methods=methods,
                workload=workload,
            )
            return
        no_prefix: LatencySummary | None = None
        native_prefix: LatencySummary | None = None
        explicit_blend: ExplicitBlendSummary | None = None

        if "no_prefix" in methods:
            no_prefix_measurements, no_prefix = run_plain_workload(
                args=args,
                warmup_prompt_ids=list(workload["warmup_prompt_ids"]),
                query_records=list(workload["query_records"]),
                sampling_params=sampling_params,
                enable_prefix_caching=False,
            )
            del no_prefix_measurements
            safe_release_cuda_memory()

        if "native_prefix" in methods:
            native_prefix_measurements, native_prefix = run_plain_workload(
                args=args,
                warmup_prompt_ids=list(workload["warmup_prompt_ids"]),
                query_records=list(workload["query_records"]),
                sampling_params=sampling_params,
                enable_prefix_caching=True,
            )
            del native_prefix_measurements
            safe_release_cuda_memory()

        if "explicit_blend" in methods:
            explicit_blend = run_explicit_blend(
                args=args,
                out_dir=out_dir,
                workload=workload,
                sampling_params=sampling_params,
            )
            safe_release_cuda_memory()

    result = BenchmarkResult(
        model=str(args.model),
        workload_kind=str(args.workload_kind),
        datasets=list(workload["datasets"]),
        chunk_size=int(args.chunk_size),
        workload=workload["stats"],
        canonical_semantics_summary=summarize_canonical_semantics(
            unique_chunks=workload["unique_chunks"],
            prefill_plan=workload["prefill_plan"],
        ),
        no_prefix=no_prefix,
        native_prefix=native_prefix,
        explicit_blend=explicit_blend,
    )

    output_path = out_dir / "result.json"
    output_path.write_text(
        json.dumps(asdict(result), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"out_dir: {out_dir}")
    if no_prefix is not None:
        print_latency_summary("no_prefix", no_prefix)
    if native_prefix is not None:
        print_latency_summary("native_prefix", native_prefix)
    if explicit_blend is not None:
        print_latency_summary("explicit_blend_online", explicit_blend.online)
        print(
            "explicit_blend_mode: "
            f"fragment_management_policy={explicit_blend.fragment_management_policy} "
            f"online_execution_policy={explicit_blend.online_execution_policy} "
            f"online_execution_mode={explicit_blend.online_execution_mode} "
            f"blend_engine_prefix_caching={explicit_blend.blend_engine_prefix_caching}"
        )
        print(
            "explicit_blend_utility: "
            f"semantic_hints={explicit_blend.utility_enable_semantic_hints} "
            f"historical_stats={explicit_blend.online_use_historical_stats} "
            f"recompute_requests={explicit_blend.utility_mode_recompute_requests} "
            f"blend_requests={explicit_blend.utility_mode_blend_requests} "
            f"mean_pred_recompute_ms={explicit_blend.mean_predicted_recompute_utility_ms:.3f} "
            f"mean_pred_blend_ms={explicit_blend.mean_predicted_cacheblend_utility_ms:.3f} "
            f"mean_gap_ms={explicit_blend.mean_predicted_decision_gap_ms:.3f}"
        )
        print_maintenance_summary(
            "explicit_blend_initial_materialization",
            explicit_blend.initial_materialization,
        )
        print_maintenance_summary(
            "explicit_blend_inter_query_maintenance",
            explicit_blend.maintenance,
        )
        print(
            "explicit_blend_shadow: "
            f"mean_hit_tokens={explicit_blend.mean_shadow_hit_tokens:.1f} "
            f"mean_gpu_hit_tokens={explicit_blend.mean_shadow_gpu_hit_tokens:.1f} "
            f"mean_cpu_hit_tokens={explicit_blend.mean_shadow_cpu_hit_tokens:.1f} "
            f"mean_miss_tokens={explicit_blend.mean_shadow_miss_tokens:.1f}"
        )
    print(f"result_json: {output_path}")


if __name__ == "__main__":
    main()
