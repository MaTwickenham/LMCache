#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from compare_prefix_vs_blend_cpu_server import launch_server, measure_streaming_request
from compare_prefix_vs_blend_gpu_workload_server import (
    DEFAULT_DATASETS_BY_WORKLOAD,
    build_workload,
    gib_to_bytes,
    resolve_datasets,
    resolve_fragment_order_policy,
    run_blend_gpu_workload,
    run_native_prefix_workload,
)
from compare_prefix_vs_blend_memoryos_server import summarize_measurements


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
METHOD_CHOICES = ("no_prefix", "native_prefix", "cacheblend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one workload/method point for nightly TTFT sweeps over "
            "no-prefix, native prefix, and LMCache CacheBlend."
        )
    )
    parser.add_argument("--method", type=str, choices=list(METHOD_CHOICES), required=True)
    parser.add_argument("--model", type=str, default="/AI/HF_MODELS/Mistral-7B-Instruct-v0.2")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "processed"),
    )
    parser.add_argument(
        "--workload-kind",
        type=str,
        choices=list(WORKLOAD_KIND_CHOICES),
        required=True,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names. If omitted, use workload defaults.",
    )
    parser.add_argument("--max-qa-per-dataset", type=int, default=0)
    parser.add_argument(
        "--prompt-layout",
        type=str,
        choices=list(PROMPT_LAYOUT_CHOICES),
        default="memory_first",
    )
    parser.add_argument(
        "--fragment-order-policy",
        type=str,
        choices=list(FRAGMENT_ORDER_POLICY_CHOICES),
        default="auto",
    )
    parser.add_argument(
        "--prefill-order-policy",
        type=str,
        choices=list(PREFILL_ORDER_POLICY_CHOICES),
        default="sorted",
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
        "--gpu-budget-gb",
        type=float,
        required=True,
        help=(
            "Configured GPU budget in GiB. For no-prefix/native-prefix it limits "
            "the vLLM paged KV pool; for CacheBlend it is also used as the LMCache "
            "LocalGPUBackend budget."
        ),
    )
    parser.add_argument(
        "--cpu-budget-gb",
        type=float,
        default=0.0,
        help="LMCache LocalCPUBackend budget in GiB.",
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
        "--blend-connector-impl",
        type=str,
        default="fast",
        help="LMCache connector implementation. Only used when --method cacheblend.",
    )
    parser.add_argument(
        "--blend-internal-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def build_workload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    datasets = resolve_datasets(
        workload_kind=str(args.workload_kind),
        raw=str(args.datasets),
    )
    fragment_order_policy = resolve_fragment_order_policy(
        workload_kind=str(args.workload_kind),
        raw_policy=str(args.fragment_order_policy),
    )
    workload = build_workload(
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
        max_local_gpu_size=float(args.gpu_budget_gb),
        max_local_cpu_size=float(args.cpu_budget_gb),
    )
    return workload


def run_no_prefix_workload(
    *,
    args: argparse.Namespace,
    query_records: list[Any],
    warmup_prompt_ids: list[int],
    kv_cache_memory_bytes: int,
):
    session = requests.Session()
    session.trust_env = False
    measurements = []
    with launch_server(
        model=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_prefix_caching=False,
        enable_blend=False,
        startup_timeout_s=args.startup_timeout_s,
        enforce_eager=args.enforce_eager,
        cuda_visible_devices=args.cuda_visible_devices,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        swap_space=0.0,
        cpu_offload_gb=0.0,
    ):
        measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=list(warmup_prompt_ids),
            max_tokens=args.max_tokens,
        )
        for record in query_records:
            measurements.append(
                measure_streaming_request(
                    session=session,
                    port=args.port,
                    model=args.model,
                    prompt_ids=list(record.prompt_ids),
                    max_tokens=args.max_tokens,
                )
            )
    return summarize_measurements(
        measurements=measurements,
        prompt_tokens=[record.prompt_tokens for record in query_records],
    )


def main() -> None:
    args = parse_args()
    # Keep compatibility with the imported benchmark helpers, which expect the
    # original LMCache CLI field names.
    setattr(args, "max_local_gpu_size", float(args.gpu_budget_gb))
    setattr(args, "max_local_cpu_size", float(args.cpu_budget_gb))
    workload = build_workload_from_args(args)
    kv_cache_memory_bytes = gib_to_bytes(float(args.gpu_budget_gb))

    payload: dict[str, Any] = {
        "method": str(args.method),
        "config": {
            "model": str(args.model),
            "data_root": str(args.data_root),
            "workload_kind": str(args.workload_kind),
            "datasets": resolve_datasets(
                workload_kind=str(args.workload_kind),
                raw=str(args.datasets),
            ),
            "chunk_size": int(args.chunk_size),
            "gpu_budget_gb": float(args.gpu_budget_gb),
            "cpu_budget_gb": float(args.cpu_budget_gb),
            "max_model_len": int(args.max_model_len),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "dtype": str(args.dtype),
            "prefill_placement_policy": str(args.prefill_placement_policy),
            "blend_connector_impl": str(args.blend_connector_impl),
        },
        "workload": asdict(workload["stats"]),
        "prefill_plan_summary": dict(workload["prefill_plan_summary"]),
    }

    if str(args.method) == "no_prefix":
        summary = run_no_prefix_workload(
            args=args,
            query_records=workload["query_records"],
            warmup_prompt_ids=workload["warmup_prompt_ids"],
            kv_cache_memory_bytes=kv_cache_memory_bytes,
        )
        payload["result"] = asdict(summary)
        payload["result_kind"] = "latency_summary"
    elif str(args.method) == "native_prefix":
        summary = run_native_prefix_workload(
            args=args,
            query_records=workload["query_records"],
            warmup_prompt_ids=workload["warmup_prompt_ids"],
            kv_cache_memory_bytes=kv_cache_memory_bytes,
        )
        payload["result"] = asdict(summary)
        payload["result_kind"] = "latency_summary"
    elif str(args.method) == "cacheblend":
        result = run_blend_gpu_workload(
            args=args,
            query_records=workload["query_records"],
            prefill_records=workload["prefill_records"],
            warmup_prompt_ids=workload["warmup_prompt_ids"],
            chunk_size=int(args.chunk_size),
            vllm_kv_cache_memory_bytes=kv_cache_memory_bytes,
        )
        payload["method"] = f"cacheblend_{args.blend_connector_impl}"
        payload["result"] = asdict(result)
        payload["result_kind"] = "blend_chunk_result"
    else:
        raise ValueError(f"Unsupported method={args.method!r}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    result_obj = payload["result"]
    if payload["result_kind"] == "latency_summary":
        print(
            json.dumps(
                {
                    "method": payload["method"],
                    "output_json": str(output_path),
                    "mean_ttft_s": result_obj.get("mean_ttft_s"),
                    "p90_ttft_s": result_obj.get("p90_ttft_s"),
                    "mean_cached_tokens": result_obj.get("mean_cached_tokens"),
                    "cache_hit_rate": result_obj.get("cache_hit_rate"),
                },
                indent=2,
            )
        )
    else:
        online = result_obj.get("online") or {}
        print(
            json.dumps(
                {
                    "method": payload["method"],
                    "output_json": str(output_path),
                    "mean_ttft_s": online.get("mean_ttft_s"),
                    "p90_ttft_s": online.get("p90_ttft_s"),
                    "mean_cached_tokens": online.get("mean_cached_tokens"),
                    "cache_hit_rate": online.get("cache_hit_rate"),
                    "first_query_total_including_prefill_s": result_obj.get(
                        "first_query_total_including_prefill_s"
                    ),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
