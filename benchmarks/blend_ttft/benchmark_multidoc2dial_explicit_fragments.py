#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from benchmark_explicit_fragments import (
    BenchmarkResult,
    DEFAULT_MODEL,
    build_sampling_params,
    ensure_out_dir,
    print_latency_summary,
    print_maintenance_summary,
    run_explicit_blend,
    run_plain_workload,
    safe_release_cuda_memory,
)
from compare_prefix_vs_blend_cpu import temporary_environ
from compare_prefix_vs_blend_gpu_workload_server import (
    apply_prefill_order_policy,
    build_prefill_plan,
    trim_fragment_sequence_to_fit,
)
from compare_prefix_vs_blend_memoryos_server import (
    QueryRecord,
    append_token_segment,
    build_prefill_records,
    build_prompt_token_ids,
    build_system_prompt_ids,
    build_workload_stats,
)
from utility_guided_prefill import attach_prior_use_count


DEFAULT_TRACE_PATH = str(
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "multidoc2dial"
    / "multidoc2dial_q500_validation_trace_windowed_with_llm_answer.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Temporary explicit-fragment TTFT benchmark for MultiDoc2Dial. "
            "This script reuses benchmark_explicit_fragments.py runtime and only "
            "adds a dataset-specific workload builder."
        )
    )
    parser.add_argument("--trace-path", type=str, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--dataset-name", type=str, default="multidoc2dial_q500_validation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=str, default="")
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
        choices=["trace", "shuffle"],
        default="trace",
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
        "--online-execution-mode",
        choices=["blend", "native_vllm"],
        default="blend",
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
        "--utility-enable-cpu-recompute",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--utility-fallback-margin-ms", type=float, default=0.0)
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
        "--include-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include query.history_used in the question section of the prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build the workload and emit summary statistics.",
    )
    return parser.parse_args()


def load_trace(trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file does not exist: {trace_path}")
    records: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_no} of {trace_path}"
                ) from exc
    if not records:
        raise ValueError(f"Trace file is empty: {trace_path}")
    return records


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").split())


def build_question_text(
    *,
    raw_query: str,
    history_used: str,
    include_history: bool,
) -> str:
    query = str(raw_query or "").strip()
    history = str(history_used or "").strip()
    if include_history and history:
        return f"Dialogue history:\n{history}\n\nCurrent user question: {query}"
    return query


def reorder_chunk_ids(
    *,
    chunk_ids: list[str],
    dataset_tag: str,
    qa_index: int,
    policy: str,
    shuffle_seed: int,
) -> list[str]:
    if policy == "trace":
        return list(chunk_ids)
    if policy == "shuffle":
        shuffled = list(chunk_ids)
        rng = random.Random(f"{shuffle_seed}::{dataset_tag}::{qa_index}")
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unsupported fragment_order_policy={policy!r}")


def enrich_multidoc2dial_hints(
    *,
    unique_chunks: dict[str, dict[str, Any]],
    query_chunk_ids: list[list[str]],
) -> None:
    doc_counts: dict[str, int] = {}
    for chunk_ids in query_chunk_ids:
        for chunk_id in chunk_ids:
            doc_id = str(unique_chunks[chunk_id].get("doc_id", ""))
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    max_doc_count = max(doc_counts.values()) if doc_counts else 1

    for chunk_id, chunk in unique_chunks.items():
        doc_id = str(chunk.get("doc_id", ""))
        hints = dict(chunk.get("hints") or {})
        hints.setdefault("importance_score", doc_counts.get(doc_id, 0) / max_doc_count)
        hints.setdefault("graph_in_degree", 0)
        hints.setdefault("hint_role", "window")
        hints.setdefault("doc_id", doc_id)
        hints.setdefault("source_chunk_id", str(chunk.get("source_chunk_id", "")))
        chunk["hints"] = hints


def build_multidoc2dial_workload(
    *,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trace_path = Path(args.trace_path)
    traces = load_trace(trace_path)
    if int(args.max_qa_per_dataset) > 0:
        traces = traces[: int(args.max_qa_per_dataset)]

    dataset_name = str(args.dataset_name)
    unique_chunks: dict[str, dict[str, Any]] = {}
    query_records: list[QueryRecord] = []
    query_chunk_ids: list[list[str]] = []
    prefill_first_seen: list[str] = []
    prefill_seen: set[str] = set()

    system_prompt_ids = build_system_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=str(args.system_prompt),
    )
    blend_special_ids = list(
        tokenizer.encode(str(args.blend_special_str), add_special_tokens=False)
    )
    warmup_prompt_ids = list(system_prompt_ids)
    warmup_query_ids = list(
        tokenizer.encode(str(args.warmup_query_text), add_special_tokens=False)
    )
    append_token_segment(
        warmup_prompt_ids,
        warmup_query_ids,
        skip_bos=bool(warmup_prompt_ids),
    )

    for qa_index, qa_trace in enumerate(traces):
        units = list((qa_trace.get("assembly") or {}).get("final_context_units") or [])
        if not units:
            continue

        ordered_chunk_ids_in_trace: list[str] = []
        for unit in units:
            chunk_id = str(unit.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(
                    f"Missing assembly.final_context_units[].chunk_id in qa_index={qa_index}"
                )
            text = str(unit.get("text") or "").strip()
            if not text:
                raise ValueError(
                    f"Missing assembly.final_context_units[].text for chunk_id={chunk_id}"
                )
            token_ids = list(tokenizer.encode(text, add_special_tokens=False))
            token_count = len(token_ids)
            existing = unique_chunks.get(chunk_id)
            if existing is not None and str(existing.get("text", "")) != text:
                raise ValueError(
                    f"Conflicting text found for chunk_id={chunk_id} in {trace_path}"
                )
            unique_chunks[chunk_id] = {
                "text": text,
                "token_ids": token_ids,
                "tokens": token_count,
                "trace_tokens": int(unit.get("token_len", 0) or token_count),
                "type": "window",
                "layer": "body",
                "doc_id": str(unit.get("doc_id") or ""),
                "source_chunk_id": str(unit.get("source_chunk_id") or ""),
                "content_hash": str(unit.get("content_hash") or ""),
                "window_start_chunk_id": str(unit.get("window_start_chunk_id") or ""),
                "window_end_chunk_id": str(unit.get("window_end_chunk_id") or ""),
                "window_span_ids": list(unit.get("window_span_ids") or []),
            }
            ordered_chunk_ids_in_trace.append(chunk_id)

        ordered_chunk_ids = reorder_chunk_ids(
            chunk_ids=ordered_chunk_ids_in_trace,
            dataset_tag=dataset_name,
            qa_index=qa_index,
            policy=str(args.fragment_order_policy),
            shuffle_seed=int(args.shuffle_seed),
        )
        question_text = build_question_text(
            raw_query=str((qa_trace.get("query") or {}).get("raw_query", "")),
            history_used=str((qa_trace.get("query") or {}).get("history_used", "")),
            include_history=bool(args.include_history),
        )
        ordered_chunk_ids, fragment_token_ids, _trimmed = trim_fragment_sequence_to_fit(
            tokenizer=tokenizer,
            system_prompt_ids=system_prompt_ids,
            blend_special_ids=blend_special_ids,
            ordered_chunk_ids=ordered_chunk_ids,
            unique_chunks=unique_chunks,
            question_text=question_text,
            prompt_layout=str(args.prompt_layout),
            max_model_len=int(args.max_model_len),
        )
        query_chunk_ids.append(list(ordered_chunk_ids))

        for chunk_id in ordered_chunk_ids:
            if chunk_id in prefill_seen:
                continue
            prefill_seen.add(chunk_id)
            prefill_first_seen.append(chunk_id)

        prompt_ids = build_prompt_token_ids(
            tokenizer=tokenizer,
            system_prompt_ids=system_prompt_ids,
            blend_special_ids=blend_special_ids,
            fragment_token_ids=fragment_token_ids,
            question_text=question_text,
            prompt_layout=str(args.prompt_layout),
        )
        query_records.append(
            QueryRecord(
                dataset=dataset_name,
                qa_index=qa_index,
                sample_id=str(
                    qa_trace.get("trace_id")
                    or qa_trace.get("dialog_id")
                    or f"{dataset_name}:{qa_index}"
                ),
                question=question_text,
                chunk_ids=list(ordered_chunk_ids),
                prompt_ids=prompt_ids,
                prompt_tokens=len(prompt_ids),
                memory_prefix_tokens=sum(len(ids) for ids in fragment_token_ids),
            )
        )

    if not query_records:
        raise ValueError(f"No usable queries were built from trace: {trace_path}")

    attach_prior_use_count(
        unique_chunks=unique_chunks,
        query_chunk_ids=query_chunk_ids,
    )
    enrich_multidoc2dial_hints(
        unique_chunks=unique_chunks,
        query_chunk_ids=query_chunk_ids,
    )

    prefill_order = apply_prefill_order_policy(
        prefill_first_seen=prefill_first_seen,
        policy=str(args.prefill_order_policy),
        shuffle_seed=int(args.shuffle_seed),
        workload_kind="multidoc2dial",
    )
    prefill_plan = build_prefill_plan(
        unique_chunks=unique_chunks,
        prefill_order=prefill_order,
        prefill_placement_policy=str(args.prefill_placement_policy),
        utility_cost_model=str(args.utility_cost_model),
        utility_tail_lambda=float(args.utility_tail_lambda),
        utility_gpu_penalty_ms=float(args.utility_gpu_penalty_ms),
        max_local_gpu_size=float(args.max_local_gpu_size),
        max_local_cpu_size=float(args.max_local_cpu_size),
    )
    prefill_records = build_prefill_records(
        tokenizer=tokenizer,
        unique_chunks=unique_chunks,
        prefill_order=prefill_order,
        system_prompt_ids=system_prompt_ids,
        blend_special_ids=blend_special_ids,
        prefill_query_text=str(args.prefill_query_text),
        prompt_layout=str(args.prompt_layout),
        max_model_len=int(args.max_model_len),
        prefill_plan=prefill_plan,
    )
    all_prefill_records = build_prefill_records(
        tokenizer=tokenizer,
        unique_chunks=unique_chunks,
        prefill_order=prefill_order,
        system_prompt_ids=system_prompt_ids,
        blend_special_ids=blend_special_ids,
        prefill_query_text=str(args.prefill_query_text),
        prompt_layout=str(args.prompt_layout),
        max_model_len=int(args.max_model_len),
        prefill_plan=None,
    )
    stats = build_workload_stats(
        datasets=[dataset_name],
        query_records=query_records,
        unique_chunks=unique_chunks,
        prompt_layout=str(args.prompt_layout),
        fragment_order_policy=str(args.fragment_order_policy),
        shuffle_seed=int(args.shuffle_seed),
        system_prompt_tokens=len(system_prompt_ids),
        blend_special_str=str(args.blend_special_str),
    )
    return {
        "datasets": [dataset_name],
        "query_records": query_records,
        "query_chunk_ids": query_chunk_ids,
        "unique_chunks": unique_chunks,
        "prefill_order": prefill_order,
        "prefill_records": prefill_records,
        "all_prefill_records": all_prefill_records,
        "prefill_plan": prefill_plan,
        "warmup_prompt_ids": warmup_prompt_ids,
        "stats": stats,
    }


def print_workload_overview(workload: dict[str, Any]) -> None:
    stats = workload["stats"]
    print(
        "workload: "
        f"queries={stats.query_count} "
        f"unique_fragments={stats.unique_fragments} "
        f"mean_fragments={stats.mean_fragments_per_query:.2f} "
        f"mean_prompt_tokens={stats.mean_prompt_tokens:.1f} "
        f"mean_memory_prefix_tokens={stats.mean_memory_prefix_tokens:.1f}"
    )


def main() -> None:
    args = parse_args()
    args.workload_kind = "multidoc2dial"
    out_dir = ensure_out_dir(args)
    sampling_params = build_sampling_params(int(args.max_tokens))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    workload = build_multidoc2dial_workload(tokenizer=tokenizer, args=args)

    print_workload_overview(workload)
    if bool(args.dry_run):
        return

    runtime_env = {"CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices)}
    with temporary_environ(runtime_env):
        no_prefix_measurements, no_prefix = run_plain_workload(
            args=args,
            warmup_prompt_ids=list(workload["warmup_prompt_ids"]),
            query_records=list(workload["query_records"]),
            sampling_params=sampling_params,
            enable_prefix_caching=False,
        )
        del no_prefix_measurements
        safe_release_cuda_memory()

        native_prefix_measurements, native_prefix = run_plain_workload(
            args=args,
            warmup_prompt_ids=list(workload["warmup_prompt_ids"]),
            query_records=list(workload["query_records"]),
            sampling_params=sampling_params,
            enable_prefix_caching=True,
        )
        del native_prefix_measurements
        safe_release_cuda_memory()

        explicit_blend = run_explicit_blend(
            args=args,
            out_dir=out_dir,
            workload=workload,
            sampling_params=sampling_params,
        )
        safe_release_cuda_memory()

    result = BenchmarkResult(
        model=str(args.model),
        workload_kind="multidoc2dial",
        datasets=list(workload["datasets"]),
        chunk_size=int(args.chunk_size),
        workload=workload["stats"],
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
    print_latency_summary("no_prefix", no_prefix)
    print_latency_summary("native_prefix", native_prefix)
    print_latency_summary("explicit_blend_online", explicit_blend.online)
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
