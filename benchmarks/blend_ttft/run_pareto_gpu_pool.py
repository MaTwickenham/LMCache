#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from compare_prefix_vs_blend_gpu_workload_server import (
    DEFAULT_DATASETS_BY_WORKLOAD,
    build_workload,
    gib_to_bytes,
    gib_to_token_budget,
    print_workload_stats,
    resolve_datasets,
    resolve_fragment_order_policy,
    run_blend_gpu_workload,
)


WORKLOAD_KIND_CHOICES = tuple(DEFAULT_DATASETS_BY_WORKLOAD.keys())
PROMPT_LAYOUT_CHOICES = ("memory_first", "question_first")
FRAGMENT_ORDER_POLICY_CHOICES = (
    "auto",
    "trace",
    "shuffle",
    "profile_last",
    "profile_last_shuffle",
)
PREFILL_ORDER_POLICY_CHOICES = ("sorted", "first_seen", "shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep reusable GPU fragment-cache budgets for LMCache CacheBlend "
            "and export plot_fig2_pareto.py-compatible stats JSON files."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/AI/HF_MODELS/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "processed"),
    )
    parser.add_argument(
        "--workload-kind",
        type=str,
        choices=list(WORKLOAD_KIND_CHOICES),
        default="memoryos",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help=(
            "Comma-separated dataset names. If omitted, use all default "
            "datasets for the selected workload."
        ),
    )
    parser.add_argument("--max-qa-per-dataset", type=int, default=0)
    parser.add_argument(
        "--prompt-layout",
        type=str,
        default="memory_first",
        choices=list(PROMPT_LAYOUT_CHOICES),
    )
    parser.add_argument(
        "--fragment-order-policy",
        type=str,
        default="auto",
        choices=list(FRAGMENT_ORDER_POLICY_CHOICES),
    )
    parser.add_argument(
        "--prefill-order-policy",
        type=str,
        default="sorted",
        choices=list(PREFILL_ORDER_POLICY_CHOICES),
    )
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--warmup-query-text", type=str, default="Warm up the engine.")
    parser.add_argument(
        "--prefill-query-text",
        type=str,
        default="Warm up this fragment for later QA use.",
    )
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--blend-special-str", type=str, default="# #")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument(
        "--blend-vllm-kv-cache-gb",
        type=float,
        default=2.0,
        help="Serving-time vLLM paged-KV pool size for each Blend run in GiB.",
    )
    parser.add_argument(
        "--max-local-cpu-size",
        type=float,
        default=0.0,
        help="Fixed LMCache LocalCPUBackend budget in GiB.",
    )
    parser.add_argument(
        "--gpu-budgets",
        type=str,
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated reusable GPU fragment-cache budgets in GiB.",
    )
    parser.add_argument(
        "--prefill-placement-policy",
        type=str,
        choices=["all_gpu", "all_cpu", "utility"],
        default="all_gpu",
    )
    parser.add_argument(
        "--utility-cost-model",
        type=str,
        default='{"recompute_ms_per_token":0.02,"transfer_gib_per_s":12.0}',
    )
    parser.add_argument("--utility-tail-lambda", type=float, default=0.0)
    parser.add_argument("--utility-gpu-penalty-ms", type=float, default=0.0)
    parser.add_argument("--blend-check-layers", type=str, default="1")
    parser.add_argument("--blend-recompute-ratios", type=str, default="0.15")
    parser.add_argument(
        "--connector-impls",
        type=str,
        default="legacy,fast",
        help="Comma-separated LMCache connector implementations to compare.",
    )
    parser.add_argument(
        "--profile-names",
        type=str,
        default="baseline,full",
        help=(
            "Comma-separated output profile names aligned with --connector-impls. "
            "These are written into run_index.tsv for plot_fig2_pareto.py."
        ),
    )
    parser.add_argument(
        "--blend-internal-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable detailed CUDA-event timing inside the connector. "
            "Disabled by default for TTFT-oriented Pareto sweeps."
        ),
    )
    parser.add_argument("--blend-pipeline-buffers", type=int, default=3)
    parser.add_argument("--blend-buffer-bucket-tokens", type=int, default=256)
    parser.add_argument("--blend-max-cached-buffer-packs", type=int, default=4)
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--startup-timeout-s", type=float, default=240.0)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        default="cacheblend_gpu_pool",
        help="Strategy tag stored into stats JSON files for plot filtering.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output run directory. Defaults under benchmarks/blend_ttft/analysis_results.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a budget/profile point when its stats JSON already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the resolved run plan without executing server benchmarks.",
    )
    return parser.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_gpu_budgets(raw: str) -> list[float]:
    budgets = [float(item) for item in parse_csv(raw)]
    if not budgets:
        raise ValueError("gpu_budgets cannot be empty.")
    return budgets


def resolve_profile_names(connector_impls: list[str], raw: str) -> list[str]:
    requested = parse_csv(raw)
    if requested:
        if len(requested) != len(connector_impls):
            raise ValueError(
                "--profile-names must have the same length as --connector-impls."
            )
        return requested
    return [impl for impl in connector_impls]


def default_out_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}__pareto_gpu_pool_{args.workload_kind}"
    return (
        Path(__file__).resolve().parent
        / "analysis_results"
        / "pareto_gpu_pool"
        / name
    )


def dataset_tag(datasets: list[str]) -> str:
    if len(datasets) == 1:
        return datasets[0]
    return "merged"


def format_budget_gb(value: float) -> str:
    return f"{float(value):.1f}"


def make_stats_basename(
    *,
    strategy: str,
    dataset_name: str,
    gpu_budget_gb: float,
    cpu_budget_gb: float,
) -> str:
    return (
        f"{strategy}_{dataset_name}_gpu{format_budget_gb(gpu_budget_gb)}gb_"
        f"cpu{format_budget_gb(cpu_budget_gb)}gb"
    )


def build_workload_for_budget(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    datasets: list[str],
    fragment_order_policy: str,
    gpu_budget_gb: float,
) -> dict[str, Any]:
    return build_workload(
        tokenizer=tokenizer,
        data_root=Path(args.data_root),
        workload_kind=str(args.workload_kind),
        datasets=datasets,
        max_qa_per_dataset=int(args.max_qa_per_dataset),
        prompt_layout=str(args.prompt_layout),
        fragment_order_policy=fragment_order_policy,
        prefill_order_policy=str(args.prefill_order_policy),
        shuffle_seed=int(args.shuffle_seed),
        blend_special_str=str(args.blend_special_str),
        system_prompt=str(args.system_prompt),
        max_model_len=int(args.max_model_len),
        warmup_query_text=str(args.warmup_query_text),
        prefill_query_text=str(args.prefill_query_text),
        prefill_placement_policy=str(args.prefill_placement_policy),
        utility_cost_model=str(args.utility_cost_model),
        utility_tail_lambda=float(args.utility_tail_lambda),
        utility_gpu_penalty_ms=float(args.utility_gpu_penalty_ms),
        max_local_gpu_size=float(gpu_budget_gb),
        max_local_cpu_size=float(args.max_local_cpu_size),
    )


def clone_args(
    args: argparse.Namespace,
    *,
    connector_impl: str,
    gpu_budget_gb: float,
) -> argparse.Namespace:
    payload = dict(vars(args))
    payload["blend_connector_impl"] = str(connector_impl)
    payload["max_local_gpu_size"] = float(gpu_budget_gb)
    return argparse.Namespace(**payload)


def build_stats_record(
    *,
    args: argparse.Namespace,
    profile_name: str,
    connector_impl: str,
    datasets: list[str],
    workload: dict[str, Any],
    gpu_budget_gb: float,
    blend_result: Any,
    strategy: str,
    full_record_file: str,
) -> dict[str, Any]:
    stats = {
        "total_qas": int(blend_result.online.request_count),
        "successful_qas": int(blend_result.online.request_count),
        "failed_qas": 0,
        "avg_ttft": float(blend_result.online.mean_ttft_s),
        "avg_end_to_end_ttft": float(blend_result.online.mean_ttft_s),
        "p50_ttft": float(blend_result.online.p50_ttft_s),
        "p90_ttft": float(blend_result.online.p90_ttft_s),
        "max_ttft": float(blend_result.online.max_ttft_s),
        "avg_wall": float(blend_result.online.mean_wall_s),
        "avg_end_to_end_cycle_time": float(blend_result.online.mean_wall_s),
        "p50_wall": float(blend_result.online.p50_wall_s),
        "p90_wall": float(blend_result.online.p90_wall_s),
        "max_wall": float(blend_result.online.max_wall_s),
        "mean_prompt_tokens": float(blend_result.online.mean_prompt_tokens),
        "mean_cached_tokens": (
            None
            if blend_result.online.mean_cached_tokens is None
            else float(blend_result.online.mean_cached_tokens)
        ),
        "p50_cached_tokens": (
            None
            if blend_result.online.p50_cached_tokens is None
            else float(blend_result.online.p50_cached_tokens)
        ),
        "p90_cached_tokens": (
            None
            if blend_result.online.p90_cached_tokens is None
            else float(blend_result.online.p90_cached_tokens)
        ),
        "cache_hit_requests": int(blend_result.online.cache_hit_requests),
        "cache_hit_rate": float(blend_result.online.cache_hit_rate),
        "prefill_request_count": int(blend_result.prefill.request_count),
        "prefill_total_prompt_tokens": int(blend_result.prefill.total_prompt_tokens),
        "prefill_wall_s": float(blend_result.prefill.wall_s),
        "prefill_mean_wall_s": float(blend_result.prefill.mean_wall_s),
        "prefill_p50_wall_s": float(blend_result.prefill.p50_wall_s),
        "prefill_p90_wall_s": float(blend_result.prefill.p90_wall_s),
        "prefill_mean_ttft_s": float(blend_result.prefill.mean_ttft_s),
        "prefill_p50_ttft_s": float(blend_result.prefill.p50_ttft_s),
        "prefill_p90_ttft_s": float(blend_result.prefill.p90_ttft_s),
        "first_query_total_including_prefill_s": float(
            blend_result.first_query_total_including_prefill_s
        ),
        "end_to_end_total_wall_s": float(blend_result.end_to_end_total_wall_s),
        "end_to_end_total_ttft_s": float(blend_result.end_to_end_total_ttft_s),
        "reusable_gpu_pool_gb": float(gpu_budget_gb),
        "reusable_cpu_pool_gb": float(args.max_local_cpu_size),
        "workload_query_count": int(workload["stats"].query_count),
        "workload_unique_fragments": int(workload["stats"].unique_fragments),
        "workload_total_unique_fragment_tokens": int(
            workload["stats"].total_unique_fragment_tokens
        ),
    }
    for phase_name, summary in sorted(blend_result.online_phase_breakdown.items()):
        stats[f"{phase_name}_sample_count"] = int(summary.sample_count)
        stats[f"{phase_name}_total_s"] = float(summary.total_s)
        stats[f"{phase_name}_mean_s"] = float(summary.mean_s)
        stats[f"{phase_name}_p50_s"] = float(summary.p50_s)
        stats[f"{phase_name}_p90_s"] = float(summary.p90_s)
        stats[f"{phase_name}_max_s"] = float(summary.max_s)

    config = {
        "model": str(args.model),
        "workload_kind": str(args.workload_kind),
        "datasets": list(datasets),
        "dataset_tag": dataset_tag(datasets),
        "chunk_size": int(args.chunk_size),
        "prompt_layout": str(workload["stats"].prompt_layout),
        "fragment_order_policy": str(workload["stats"].fragment_order_policy),
        "prefill_order_policy": str(args.prefill_order_policy),
        "prefill_placement_policy": str(args.prefill_placement_policy),
        "shuffle_seed": int(args.shuffle_seed),
        "profile_name": str(profile_name),
        "connector_impl": str(connector_impl),
        "blend_internal_timing": bool(args.blend_internal_timing),
        "blend_pipeline_buffers": int(args.blend_pipeline_buffers),
        "blend_buffer_bucket_tokens": int(args.blend_buffer_bucket_tokens),
        "blend_max_cached_buffer_packs": int(args.blend_max_cached_buffer_packs),
        "blend_vllm_kv_cache_gb": float(args.blend_vllm_kv_cache_gb),
        "gpu_budget_tokens": int(gib_to_token_budget(float(gpu_budget_gb))),
        "cpu_budget_tokens": int(gib_to_token_budget(float(args.max_local_cpu_size))),
        "gpu_pool_budget_gb": float(gpu_budget_gb),
        "cpu_pool_budget_gb": float(args.max_local_cpu_size),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "dtype": str(args.dtype),
    }

    return {
        "file_kind": "stats",
        "strategy": str(strategy),
        "dataset": dataset_tag(datasets),
        "config": config,
        "statistics": stats,
        "full_record_file": str(full_record_file),
    }


def build_full_record(
    *,
    args: argparse.Namespace,
    profile_name: str,
    connector_impl: str,
    datasets: list[str],
    workload: dict[str, Any],
    gpu_budget_gb: float,
    blend_result: Any,
    strategy: str,
) -> dict[str, Any]:
    return {
        "file_kind": "full",
        "strategy": str(strategy),
        "dataset": dataset_tag(datasets),
        "config": {
            "model": str(args.model),
            "workload_kind": str(args.workload_kind),
            "datasets": list(datasets),
            "profile_name": str(profile_name),
            "connector_impl": str(connector_impl),
            "chunk_size": int(args.chunk_size),
            "gpu_pool_budget_gb": float(gpu_budget_gb),
            "cpu_pool_budget_gb": float(args.max_local_cpu_size),
            "blend_vllm_kv_cache_gb": float(args.blend_vllm_kv_cache_gb),
            "blend_internal_timing": bool(args.blend_internal_timing),
            "prompt_layout": str(workload["stats"].prompt_layout),
            "fragment_order_policy": str(workload["stats"].fragment_order_policy),
            "prefill_placement_policy": str(args.prefill_placement_policy),
        },
        "workload": asdict(workload["stats"]),
        "prefill_plan_summary": dict(workload["prefill_plan_summary"]),
        "blend_result": asdict(blend_result),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_run_index(
    *,
    run_dir: Path,
    workload_kind: str,
    dataset_name: str,
    profile_dirs: dict[str, Path],
) -> None:
    path = run_dir / "run_index.tsv"
    lines = ["workload\tdataset\tprofile\trun_dir"]
    for profile_name, profile_dir in profile_dirs.items():
        lines.append(
            f"{workload_kind}\t{dataset_name}\t{profile_name}\t{profile_dir}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    connector_impls = parse_csv(args.connector_impls)
    if not connector_impls:
        raise ValueError("connector_impls cannot be empty.")
    profile_names = resolve_profile_names(connector_impls, args.profile_names)
    budgets = parse_gpu_budgets(args.gpu_budgets)
    datasets = resolve_datasets(str(args.workload_kind), str(args.datasets))
    fragment_order_policy = resolve_fragment_order_policy(
        workload_kind=str(args.workload_kind),
        raw_policy=str(args.fragment_order_policy),
    )

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_config = {
        "model": str(args.model),
        "data_root": str(args.data_root),
        "workload_kind": str(args.workload_kind),
        "datasets": list(datasets),
        "chunk_size": int(args.chunk_size),
        "gpu_budgets": [float(item) for item in budgets],
        "cpu_budget_gb": float(args.max_local_cpu_size),
        "connector_impls": list(connector_impls),
        "profile_names": list(profile_names),
        "strategy": str(args.strategy),
        "blend_vllm_kv_cache_gb": float(args.blend_vllm_kv_cache_gb),
        "prefill_placement_policy": str(args.prefill_placement_policy),
        "blend_internal_timing": bool(args.blend_internal_timing),
        "cuda_visible_devices": args.cuda_visible_devices,
    }
    write_json(out_dir / "run_config.json", root_config)

    print(f"run_dir: {out_dir}")
    print(f"workload_kind: {args.workload_kind}")
    print(f"datasets: {', '.join(datasets)}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"gpu_budgets_gb: {', '.join(format_budget_gb(v) for v in budgets)}")
    print(f"cpu_budget_gb: {format_budget_gb(args.max_local_cpu_size)}")
    print(
        "profiles: "
        + ", ".join(
            f"{profile}={connector}"
            for profile, connector in zip(profile_names, connector_impls, strict=True)
        )
    )

    if args.dry_run:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vllm_kv_cache_memory_bytes = gib_to_bytes(float(args.blend_vllm_kv_cache_gb))
    profile_dirs = {
        profile_name: out_dir / profile_name
        for profile_name in profile_names
    }
    for profile_name, profile_dir in profile_dirs.items():
        profile_dir.mkdir(parents=True, exist_ok=True)
        write_json(profile_dir / "run_config.json", {"profile_name": profile_name})

    printed_workload = False
    for gpu_budget_gb in budgets:
        workload = build_workload_for_budget(
            args=args,
            tokenizer=tokenizer,
            datasets=datasets,
            fragment_order_policy=fragment_order_policy,
            gpu_budget_gb=float(gpu_budget_gb),
        )
        if (not printed_workload) or str(args.prefill_placement_policy) == "utility":
            print(f"[workload] gpu_budget_gb={gpu_budget_gb:.1f}")
            print_workload_stats(
                workload_kind=str(args.workload_kind),
                stats=workload["stats"],
                prefill_order_policy=str(args.prefill_order_policy),
                prefill_plan_summary=workload["prefill_plan_summary"],
            )
            printed_workload = True

        for profile_name, connector_impl in zip(
            profile_names, connector_impls, strict=True
        ):
            basename = make_stats_basename(
                strategy=str(args.strategy),
                dataset_name=dataset_tag(datasets),
                gpu_budget_gb=float(gpu_budget_gb),
                cpu_budget_gb=float(args.max_local_cpu_size),
            )
            profile_dir = profile_dirs[profile_name]
            stats_path = profile_dir / f"{basename}.json"
            full_path = profile_dir / f"{basename}__full.json"
            if args.skip_existing and stats_path.exists() and full_path.exists():
                print(
                    f"[skip] profile={profile_name} connector={connector_impl} "
                    f"gpu_budget_gb={gpu_budget_gb:.1f}"
                )
                continue

            print(
                f"[run] profile={profile_name} connector={connector_impl} "
                f"gpu_budget_gb={gpu_budget_gb:.1f} cpu_budget_gb={args.max_local_cpu_size:.1f}"
            )
            run_args = clone_args(
                args,
                connector_impl=str(connector_impl),
                gpu_budget_gb=float(gpu_budget_gb),
            )
            blend_result = run_blend_gpu_workload(
                args=run_args,
                query_records=workload["query_records"],
                prefill_records=workload["prefill_records"],
                warmup_prompt_ids=workload["warmup_prompt_ids"],
                chunk_size=int(args.chunk_size),
                vllm_kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
            )
            full_record = build_full_record(
                args=args,
                profile_name=profile_name,
                connector_impl=str(connector_impl),
                datasets=datasets,
                workload=workload,
                gpu_budget_gb=float(gpu_budget_gb),
                blend_result=blend_result,
                strategy=str(args.strategy),
            )
            stats_record = build_stats_record(
                args=args,
                profile_name=profile_name,
                connector_impl=str(connector_impl),
                datasets=datasets,
                workload=workload,
                gpu_budget_gb=float(gpu_budget_gb),
                blend_result=blend_result,
                strategy=str(args.strategy),
                full_record_file=full_path.name,
            )
            write_json(full_path, full_record)
            write_json(stats_path, stats_record)

    write_run_index(
        run_dir=out_dir,
        workload_kind=str(args.workload_kind),
        dataset_name=dataset_tag(datasets),
        profile_dirs=profile_dirs,
    )
    print(f"run_index: {out_dir / 'run_index.tsv'}")


if __name__ == "__main__":
    main()
