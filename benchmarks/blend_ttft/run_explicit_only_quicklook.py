#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

from benchmark_explicit_fragments import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    blend_cpu_bench_module,
    build_sampling_params,
    run_explicit_blend,
    safe_release_cuda_memory,
)
from runtime_blend_probe import build_workload_for_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick-look runner that only measures explicit_blend TTFT/hit metrics "
            "without also running no_prefix/native_prefix baselines."
        )
    )
    parser.add_argument(
        "--workload-kinds",
        type=str,
        default="memoryos,memos",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-local-gpu-size", type=float, default=2.0)
    parser.add_argument("--max-local-cpu-size", type=float, default=2.0)
    parser.add_argument("--max-qa-per-dataset", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--blend-connector-impl", type=str, default="fast")
    parser.add_argument("--utility-cost-model", type=str, default='{"recompute_ms_per_token":0.105,"transfer_gib_per_s":12.0}')
    parser.add_argument("--prompt-layout", choices=["memory_first", "question_first"], default="memory_first")
    parser.add_argument("--fragment-order-policy", choices=["auto", "trace", "shuffle", "profile_last", "profile_last_shuffle"], default="auto")
    parser.add_argument("--prefill-order-policy", choices=["sorted", "first_seen", "shuffle"], default="sorted")
    parser.add_argument("--prefill-placement-policy", choices=["all_gpu", "all_cpu", "utility"], default="all_cpu")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--blend-special-str", type=str, default="# #")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--warmup-query-text", type=str, default="Warm up the engine.")
    parser.add_argument("--prefill-query-text", type=str, default="Warm up this fragment for later QA use.")
    parser.add_argument("--blend-check-layers", type=str, default="1")
    parser.add_argument("--blend-recompute-ratios", type=str, default="0.15")
    parser.add_argument("--blend-internal-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blend-pipeline-buffers", type=int, default=3)
    parser.add_argument("--blend-buffer-bucket-tokens", type=int, default=256)
    parser.add_argument("--blend-max-cached-buffer-packs", type=int, default=4)
    parser.add_argument("--online-execution-policy", choices=["fixed", "utility"], default="utility")
    parser.add_argument("--fragment-management-policy", choices=["static", "utility"], default="utility")
    parser.add_argument("--online-execution-mode", choices=["blend", "native_vllm"], default="blend")
    parser.add_argument("--blend-engine-enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--utility-tail-lambda", type=float, default=0.0)
    parser.add_argument("--utility-gpu-penalty-ms", type=float, default=0.0)
    parser.add_argument("--utility-enable-cpu-recompute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--utility-fallback-margin-ms", type=float, default=0.0)
    parser.add_argument("--utility-prefix-history-window", type=int, default=0)
    parser.add_argument("--utility-native-runtime", choices=["recompute"], default="recompute")
    parser.add_argument("--initial-fill-mode", choices=["none", "gpu_then_cpu", "cpu_then_gpu"], default="none")
    parser.add_argument("--admit-misses-to", choices=["auto", "cpu", "gpu", "none"], default="auto")
    parser.add_argument("--gpu-lookahead", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    return parser.parse_args()


def ensure_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = (
            Path(__file__).resolve().parent
            / "analysis_results"
            / f"quicklook_explicit_only_cpu{int(args.max_local_cpu_size)}_gpu{int(args.max_local_gpu_size)}_qa{int(args.max_qa_per_dataset)}_{stamp}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_local_args(base: argparse.Namespace, workload_kind: str, use_hints: bool, run_dir: Path) -> Namespace:
    local_args = Namespace(**vars(base))
    local_args.workload_kind = workload_kind
    local_args.utility_enable_semantic_hints = bool(use_hints)
    local_args.out_dir = str(run_dir)
    return local_args


def main() -> None:
    args = parse_args()
    out_root = ensure_out_dir(args)
    workloads = [item.strip() for item in str(args.workload_kinds).split(",") if item.strip()]
    sampling_params = build_sampling_params(int(args.max_tokens))
    rows: list[dict[str, float | str]] = []

    runtime_env = {"CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices)}
    with blend_cpu_bench_module().temporary_environ(runtime_env):
        for workload_kind in workloads:
            for use_hints in (True, False):
                tag = "with_hints" if use_hints else "without_hints"
                run_dir = out_root / workload_kind / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                local_args = build_local_args(args, workload_kind, use_hints, run_dir)
                workload = build_workload_for_probe(local_args)
                explicit = run_explicit_blend(
                    args=local_args,
                    out_dir=run_dir,
                    workload=workload,
                    sampling_params=sampling_params,
                )
                (run_dir / "explicit_only_result.json").write_text(
                    json.dumps(asdict(explicit), indent=2) + "\n",
                    encoding="utf-8",
                )

                hit_tokens = float(explicit.mean_shadow_hit_tokens)
                miss_tokens = float(explicit.mean_shadow_miss_tokens)
                hit_rate = hit_tokens / (hit_tokens + miss_tokens) if (hit_tokens + miss_tokens) > 0.0 else 0.0
                rows.append(
                    {
                        "workload": workload_kind,
                        "tag": tag,
                        "mean_ttft_s": float(explicit.online.mean_ttft_s),
                        "p90_ttft_s": float(explicit.online.p90_ttft_s),
                        "shadow_hit_rate": float(hit_rate),
                        "mean_shadow_hit_tokens": float(explicit.mean_shadow_hit_tokens),
                        "mean_shadow_miss_tokens": float(explicit.mean_shadow_miss_tokens),
                    }
                )
                safe_release_cuda_memory()

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"summary_json: {summary_path}")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
