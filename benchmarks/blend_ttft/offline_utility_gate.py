#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmark_explicit_fragments import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    build_utility_cfg,
    decide_online_execution_mode,
    mean_or_zero,
    seed_pool_from_prefill_plan,
)
from explicit_fragment_control_plane import ExplicitFragmentPool
from runtime_blend_probe import build_workload_for_probe


@dataclass
class OfflineUtilitySummary:
    workload_kind: str
    datasets: list[str]
    query_count: int
    chunk_size: int
    cpu_budget_gib: float
    gpu_budget_gib: float
    initial_fill_mode: str
    admit_misses_to: str
    gpu_lookahead: int
    utility_mode_recompute_requests: int
    utility_mode_blend_requests: int
    mean_shadow_hit_tokens: float
    mean_shadow_gpu_hit_tokens: float
    mean_shadow_cpu_hit_tokens: float
    mean_shadow_miss_tokens: float
    mean_predicted_recompute_utility_ms: float
    mean_predicted_cacheblend_utility_ms: float
    mean_predicted_decision_gap_ms: float
    mean_prompt_tokens: float
    resident_pool_end: dict[str, int]


class FakeRuntime:
    def materialize_fragment(self, prefill_record: Any, *, location: str) -> float:
        return 0.0

    def move_fragment(
        self,
        prefill_record: Any,
        *,
        src: str,
        dst: str,
        remove_src: bool = True,
    ) -> float:
        return 0.0

    def evict_fragment(self, prefill_record: Any, *, location: str) -> float:
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline utility-gate sweep for the single-engine LMCache benchmark. "
            "This does not run the model; it only replays pool state evolution and "
            "utility decisions over real workloads."
        )
    )
    parser.add_argument(
        "--workload-kinds",
        type=str,
        default="memoryos,memos,amem",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-local-gpu-size", type=float, default=2.0)
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
    parser.add_argument("--max-qa-per-dataset", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--blend-special-str", type=str, default="# #")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--warmup-query-text", type=str, default="Warm up the engine.")
    parser.add_argument(
        "--prefill-query-text",
        type=str,
        default="Warm up this fragment for later QA use.",
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
        "--utility-cost-model",
        type=str,
        default='{"recompute_ms_per_token":0.105,"transfer_gib_per_s":12.0}',
    )
    parser.add_argument("--utility-tail-lambda", type=float, default=0.0)
    parser.add_argument("--utility-gpu-penalty-ms", type=float, default=0.0)
    parser.add_argument(
        "--utility-enable-cpu-recompute",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--utility-fallback-margin-ms", type=float, default=0.0)
    parser.add_argument("--utility-prefix-history-window", type=int, default=0)
    parser.add_argument(
        "--utility-native-runtime",
        choices=["recompute"],
        default="recompute",
    )
    parser.add_argument(
        "--online-execution-policy",
        choices=["fixed", "utility"],
        default="utility",
    )
    parser.add_argument(
        "--online-execution-mode",
        choices=["blend", "native_vllm"],
        default="blend",
    )
    parser.add_argument(
        "--blend-engine-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def ensure_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parent / "analysis_results" / "offline_utility_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def summarize_workload(
    args: argparse.Namespace, workload_kind: str
) -> OfflineUtilitySummary:
    local_args = argparse.Namespace(**vars(args))
    local_args.workload_kind = workload_kind

    workload = build_workload_for_probe(local_args)
    fragment_tokens = {
        chunk_id: int(meta["tokens"])
        for chunk_id, meta in workload["unique_chunks"].items()
    }
    prefill_records = {
        record.chunk_id: record
        for record in workload.get("all_prefill_records", workload["prefill_records"])
    }
    cpu_budget_tokens = int((float(args.max_local_cpu_size) * (1024**3)) / 131072)
    gpu_budget_tokens = int((float(args.max_local_gpu_size) * (1024**3)) / 131072)

    pool = ExplicitFragmentPool(
        runtime=FakeRuntime(),
        prefill_records=prefill_records,
        fragment_tokens=fragment_tokens,
        cpu_budget_tokens=cpu_budget_tokens,
        gpu_budget_tokens=gpu_budget_tokens,
        cpu_enabled=float(args.max_local_cpu_size) > 0,
        gpu_enabled=float(args.max_local_gpu_size) > 0,
        admit_misses_to=args.admit_misses_to,
        gpu_lookahead=int(args.gpu_lookahead),
    )

    if str(args.prefill_placement_policy) == "utility":
        seed_pool_from_prefill_plan(
            pool=pool,
            prefill_order=list(workload["prefill_order"]),
            prefill_plan=dict(workload["prefill_plan"]),
        )
    elif str(args.initial_fill_mode) == "gpu_then_cpu":
        pool.seed_initial_resident_pool(
            chunk_order=list(workload["prefill_order"]),
            tier_order=["gpu", "cpu"],
        )
    elif str(args.initial_fill_mode) == "cpu_then_gpu":
        pool.seed_initial_resident_pool(
            chunk_order=list(workload["prefill_order"]),
            tier_order=["cpu", "gpu"],
        )

    utility_cfg = build_utility_cfg(local_args)
    query_records = list(workload["query_records"])
    execution_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []

    for index, record in enumerate(query_records):
        snapshot = pool.capture_query_reuse(record.chunk_ids)
        execution_mode, execution_debug = decide_online_execution_mode(
            args=local_args,
            record=record,
            unique_chunks=workload["unique_chunks"],
            snapshot=snapshot,
            utility_cfg=utility_cfg,
        )
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
                "predicted_decision_gap_ms": float(
                    execution_debug.get("decision_gap_ms", 0.0) or 0.0
                ),
            }
        )
        shadow_rows.append(
            {
                "shadow_hit_tokens": snapshot.hit_tokens,
                "shadow_gpu_hit_tokens": snapshot.gpu_hit_tokens,
                "shadow_cpu_hit_tokens": snapshot.cpu_hit_tokens,
                "shadow_miss_tokens": snapshot.miss_tokens,
            }
        )

        next_chunk_ids = None
        if index + 1 < len(query_records) and int(args.gpu_lookahead) > 0:
            next_chunk_ids = list(query_records[index + 1].chunk_ids)
        pool.run_inter_query_maintenance(
            current_chunk_ids=list(record.chunk_ids),
            next_chunk_ids=next_chunk_ids,
        )

    recompute_requests = sum(
        1 for row in execution_rows if row["execution_mode"] == "native_vllm"
    )
    blend_requests = sum(1 for row in execution_rows if row["execution_mode"] == "blend")

    return OfflineUtilitySummary(
        workload_kind=workload_kind,
        datasets=list(workload["datasets"]),
        query_count=len(query_records),
        chunk_size=int(args.chunk_size),
        cpu_budget_gib=float(args.max_local_cpu_size),
        gpu_budget_gib=float(args.max_local_gpu_size),
        initial_fill_mode=str(args.initial_fill_mode),
        admit_misses_to=str(args.admit_misses_to),
        gpu_lookahead=int(args.gpu_lookahead),
        utility_mode_recompute_requests=int(recompute_requests),
        utility_mode_blend_requests=int(blend_requests),
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
        mean_predicted_recompute_utility_ms=mean_or_zero(
            [row["predicted_recompute_utility_ms"] for row in execution_rows]
        ),
        mean_predicted_cacheblend_utility_ms=mean_or_zero(
            [row["predicted_cacheblend_utility_ms"] for row in execution_rows]
        ),
        mean_predicted_decision_gap_ms=mean_or_zero(
            [row["predicted_decision_gap_ms"] for row in execution_rows]
        ),
        mean_prompt_tokens=mean_or_zero(
            [record.prompt_tokens for record in query_records]
        ),
        resident_pool_end=pool.resident_counts(),
    )


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args)
    workload_kinds = [item.strip() for item in str(args.workload_kinds).split(",") if item.strip()]

    summaries = [summarize_workload(args, workload_kind) for workload_kind in workload_kinds]
    payload = {
        "args": vars(args),
        "summaries": [asdict(summary) for summary in summaries],
    }
    output_path = out_dir / "result.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"result_json: {output_path}")


if __name__ == "__main__":
    main()
