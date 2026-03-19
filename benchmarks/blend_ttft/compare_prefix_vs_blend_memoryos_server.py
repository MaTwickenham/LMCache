# SPDX-License-Identifier: Apache-2.0
"""Benchmark native prefix caching vs LMCache CacheBlend on real MemoryOS traces.

This benchmark reuses the OpenAI-compatible streaming server path from
``compare_prefix_vs_blend_cpu_server.py`` so TTFT is measured as wall-clock
time from request submission to the first streamed token.

Unlike the synthetic benchmark, this script feeds real MemoryOS QA traces:

1. Load MemoryOS traces and extract reusable fragments.
2. Build the same prompt sequence for all modes.
3. Run the online workload sequentially with:
   - plain vLLM without prefix caching,
   - native vLLM prefix caching,
   - LMCache CacheBlend on CPU RAM for one or more chunk sizes.
4. Report both online query TTFT and offline fragment-prefill wall time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from compare_prefix_vs_blend_cpu_server import (
    RequestMeasurement,
    format_optional_float,
    launch_server,
    measure_streaming_request,
)


PROMPT_LAYOUT_CHOICES = ("memory_first", "question_first")
FRAGMENT_ORDER_POLICY_CHOICES = (
    "trace",
    "shuffle",
    "profile_last",
    "profile_last_shuffle",
)
BLEND_BACKEND_CHOICES = ("cpu", "gpu")


@dataclass
class QueryRecord:
    dataset: str
    qa_index: int
    sample_id: str
    question: str
    chunk_ids: list[str]
    prompt_ids: list[int]
    prompt_tokens: int
    memory_prefix_tokens: int


@dataclass
class PrefillRecord:
    chunk_id: str
    chunk_type: str
    prompt_ids: list[int]
    prompt_tokens: int
    store_location: str | None = None
    utility_action: str | None = None
    utility_score: float | None = None


@dataclass
class LatencySummary:
    request_count: int
    total_ttft_s: float
    mean_ttft_s: float
    p50_ttft_s: float
    p90_ttft_s: float
    max_ttft_s: float
    total_wall_s: float
    mean_wall_s: float
    p50_wall_s: float
    p90_wall_s: float
    max_wall_s: float
    mean_prompt_tokens: float
    mean_cached_tokens: float | None
    p50_cached_tokens: float | None
    p90_cached_tokens: float | None
    cache_hit_requests: int
    cache_hit_rate: float


@dataclass
class PrefillSummary:
    request_count: int
    total_prompt_tokens: int
    wall_s: float
    mean_wall_s: float
    p50_wall_s: float
    p90_wall_s: float
    mean_ttft_s: float
    p50_ttft_s: float
    p90_ttft_s: float


@dataclass
class BlendChunkResult:
    backend: str
    chunk_size: int
    prefill: PrefillSummary
    online: LatencySummary
    online_phase_breakdown: dict[str, "PhaseSummary"]
    end_to_end_total_wall_s: float
    end_to_end_total_ttft_s: float
    first_query_total_including_prefill_s: float


@dataclass
class PhaseSummary:
    sample_count: int
    total_s: float
    mean_s: float
    p50_s: float
    p90_s: float
    max_s: float


@dataclass
class WorkloadStats:
    datasets: list[str]
    query_count: int
    unique_fragments: int
    total_unique_fragment_tokens: int
    mean_fragments_per_query: float
    p50_fragments_per_query: float
    p90_fragments_per_query: float
    mean_prompt_tokens: float
    p50_prompt_tokens: float
    p90_prompt_tokens: float
    max_prompt_tokens: int
    mean_memory_prefix_tokens: float
    p50_memory_prefix_tokens: float
    p90_memory_prefix_tokens: float
    max_memory_prefix_tokens: int
    prompt_layout: str
    fragment_order_policy: str
    shuffle_seed: int
    system_prompt_tokens: int
    blend_special_str: str


@dataclass
class BenchmarkResult:
    model: str
    data_dir: str
    workload: WorkloadStats
    no_prefix: LatencySummary
    native_prefix: LatencySummary
    blend_results: list[BlendChunkResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark real MemoryOS prompts under plain vLLM, native prefix "
            "caching, and LMCache CacheBlend."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/AI/HF_MODELS/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2] / "data" / "processed" / "memoryos"
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="conv26",
        help="Comma-separated MemoryOS datasets, for example conv26 or conv26,conv30,conv41",
    )
    parser.add_argument(
        "--max-qa-per-dataset",
        type=int,
        default=16,
        help="Maximum QA traces to load from each dataset. 0 means all.",
    )
    parser.add_argument(
        "--prompt-layout",
        type=str,
        default="memory_first",
        choices=list(PROMPT_LAYOUT_CHOICES),
    )
    parser.add_argument(
        "--fragment-order-policy",
        type=str,
        default="profile_last",
        choices=list(FRAGMENT_ORDER_POLICY_CHOICES),
        help=(
            "How to reorder fragments inside each query. "
            "'profile_last' is the recommended fair default for MemoryOS."
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Base seed for deterministic fragment reordering.",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="512,256,128",
        help="Comma-separated LMCache chunk sizes to test.",
    )
    parser.add_argument(
        "--blend-backends",
        type=str,
        default="cpu",
        help="Comma-separated LMCache backends to test from {cpu,gpu}.",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--warmup-query-text", type=str, default="Warm up the engine.")
    parser.add_argument(
        "--prefill-query-text",
        type=str,
        default="Warm up this fragment for later QA use.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--blend-special-str", type=str, default="# #")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--max-local-cpu-size", type=float, default=8.0)
    parser.add_argument("--max-local-gpu-size", type=float, default=4.0)
    parser.add_argument("--blend-check-layers", type=str, default="1")
    parser.add_argument("--blend-recompute-ratios", type=str, default="0.15")
    parser.add_argument(
        "--blend-connector-impl",
        type=str,
        default="fast",
        help="LMCache blend connector implementation name.",
    )
    parser.add_argument(
        "--blend-internal-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable detailed internal CUDA-event timing inside the blend "
            "connector. Keep this on for profiling benchmarks; turn it off "
            "for lower serving overhead."
        ),
    )
    parser.add_argument(
        "--blend-pipeline-buffers",
        type=int,
        default=3,
        help="Number of in-flight full-window layer buffers used by the fast blend connector.",
    )
    parser.add_argument(
        "--blend-buffer-bucket-tokens",
        type=int,
        default=256,
        help="Token granularity used to bucket reusable fast-connector layer buffers.",
    )
    parser.add_argument(
        "--blend-max-cached-buffer-packs",
        type=int,
        default=4,
        help="Maximum number of cached fast-connector buffer buckets kept warm.",
    )
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--startup-timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Optional vLLM max_num_seqs override for constrained GPUs.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional vLLM max_num_batched_tokens override for constrained GPUs.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load traces and build prompts, then print workload statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_sizes = [int(value) for value in args.chunk_sizes.split(",") if value]
    if not chunk_sizes:
        raise ValueError("chunk_sizes cannot be empty.")
    blend_backends = parse_blend_backends(args.blend_backends)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    workload = build_memoryos_workload(
        tokenizer=tokenizer,
        data_dir=Path(args.data_dir),
        datasets=parse_datasets(args.datasets),
        max_qa_per_dataset=int(args.max_qa_per_dataset),
        prompt_layout=str(args.prompt_layout),
        fragment_order_policy=str(args.fragment_order_policy),
        shuffle_seed=int(args.shuffle_seed),
        blend_special_str=str(args.blend_special_str),
        system_prompt=str(args.system_prompt),
        max_model_len=int(args.max_model_len),
        warmup_query_text=str(args.warmup_query_text),
        prefill_query_text=str(args.prefill_query_text),
    )

    print_workload_stats(workload["stats"])
    if args.dry_run:
        return

    no_prefix = run_online_workload(
        args=args,
        query_records=workload["query_records"],
        warmup_prompt_ids=workload["warmup_prompt_ids"],
        enable_prefix_caching=False,
        enable_blend=False,
    )
    native_prefix = run_online_workload(
        args=args,
        query_records=workload["query_records"],
        warmup_prompt_ids=workload["warmup_prompt_ids"],
        enable_prefix_caching=True,
        enable_blend=False,
    )

    blend_results: list[BlendChunkResult] = []
    for backend in blend_backends:
        for chunk_size in chunk_sizes:
            blend_results.append(
                run_blend_workload(
                    args=args,
                    query_records=workload["query_records"],
                    prefill_records=workload["prefill_records"],
                    warmup_prompt_ids=workload["warmup_prompt_ids"],
                    chunk_size=chunk_size,
                    backend=backend,
                )
            )

    result = BenchmarkResult(
        model=str(args.model),
        data_dir=str(args.data_dir),
        workload=workload["stats"],
        no_prefix=no_prefix,
        native_prefix=native_prefix,
        blend_results=blend_results,
    )

    print_result(result)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(result), file, indent=2)
            file.write("\n")


def parse_datasets(raw: str) -> list[str]:
    datasets = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not datasets:
        raise ValueError("datasets cannot be empty.")
    return datasets


def parse_blend_backends(raw: str) -> list[str]:
    backends = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not backends:
        raise ValueError("blend_backends cannot be empty.")
    invalid = [item for item in backends if item not in BLEND_BACKEND_CHOICES]
    if invalid:
        raise ValueError(
            f"Unsupported blend backend(s): {invalid}. "
            f"Expected a subset of {list(BLEND_BACKEND_CHOICES)}."
        )
    return backends


def build_memoryos_workload(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_dir: Path,
    datasets: list[str],
    max_qa_per_dataset: int,
    prompt_layout: str,
    fragment_order_policy: str,
    shuffle_seed: int,
    blend_special_str: str,
    system_prompt: str,
    max_model_len: int,
    warmup_query_text: str,
    prefill_query_text: str,
) -> dict[str, Any]:
    if not data_dir.exists():
        raise FileNotFoundError(f"MemoryOS data directory does not exist: {data_dir}")

    unique_chunks: dict[str, dict[str, Any]] = {}
    query_records: list[QueryRecord] = []
    prefill_order: list[str] = []

    system_prompt_ids = build_system_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    blend_special_ids = list(tokenizer.encode(blend_special_str, add_special_tokens=False))
    warmup_prompt_ids = list(system_prompt_ids)
    warmup_query_ids = list(
        tokenizer.encode(warmup_query_text, add_special_tokens=False)
    )
    append_token_segment(
        warmup_prompt_ids,
        warmup_query_ids,
        skip_bos=bool(warmup_prompt_ids),
    )

    for dataset in datasets:
        traces = load_memoryos_trace(data_dir=data_dir, dataset=dataset)
        if max_qa_per_dataset > 0:
            traces = traces[:max_qa_per_dataset]

        for qa_index, qa_trace in enumerate(traces):
            chunk_ids = extract_memory_chunks(
                qa_trace=qa_trace,
                tokenizer=tokenizer,
                unique_chunks=unique_chunks,
            )
            ordered_chunk_ids = reorder_chunk_ids(
                chunk_ids=chunk_ids,
                unique_chunks=unique_chunks,
                dataset_tag=dataset,
                qa_index=qa_index,
                policy=fragment_order_policy,
                shuffle_seed=shuffle_seed,
            )

            for chunk_id in ordered_chunk_ids:
                if chunk_id not in prefill_order:
                    prefill_order.append(chunk_id)

            fragment_token_ids = [
                list(unique_chunks[chunk_id]["token_ids"]) for chunk_id in ordered_chunk_ids
            ]
            prompt_ids = build_prompt_token_ids(
                tokenizer=tokenizer,
                system_prompt_ids=system_prompt_ids,
                blend_special_ids=blend_special_ids,
                fragment_token_ids=fragment_token_ids,
                question_text=str(qa_trace.get("question", "")),
                prompt_layout=prompt_layout,
            )
            memory_prefix_tokens = estimate_memory_prefix_tokens(
                tokenizer=tokenizer,
                system_prompt_ids=system_prompt_ids,
                blend_special_ids=blend_special_ids,
                fragment_token_ids=fragment_token_ids,
                question_text=str(qa_trace.get("question", "")),
                prompt_layout=prompt_layout,
            )
            if len(prompt_ids) > max_model_len:
                raise ValueError(
                    "Prompt length exceeds max_model_len. "
                    f"dataset={dataset}, qa_index={qa_index}, "
                    f"prompt_tokens={len(prompt_ids)}, max_model_len={max_model_len}."
                )

            query_records.append(
                QueryRecord(
                    dataset=dataset,
                    qa_index=qa_index,
                    sample_id=str(qa_trace.get("sample_id", dataset)),
                    question=str(qa_trace.get("question", "")),
                    chunk_ids=ordered_chunk_ids,
                    prompt_ids=prompt_ids,
                    prompt_tokens=len(prompt_ids),
                    memory_prefix_tokens=memory_prefix_tokens,
                )
            )

    prefill_records = build_prefill_records(
        tokenizer=tokenizer,
        unique_chunks=unique_chunks,
        prefill_order=prefill_order,
        system_prompt_ids=system_prompt_ids,
        blend_special_ids=blend_special_ids,
        prefill_query_text=prefill_query_text,
        prompt_layout=prompt_layout,
        max_model_len=max_model_len,
    )
    stats = build_workload_stats(
        datasets=datasets,
        query_records=query_records,
        unique_chunks=unique_chunks,
        prompt_layout=prompt_layout,
        fragment_order_policy=fragment_order_policy,
        shuffle_seed=shuffle_seed,
        system_prompt_tokens=len(system_prompt_ids),
        blend_special_str=blend_special_str,
    )
    return {
        "stats": stats,
        "query_records": query_records,
        "prefill_records": prefill_records,
        "warmup_prompt_ids": warmup_prompt_ids,
    }


def load_memoryos_trace(*, data_dir: Path, dataset: str) -> list[dict[str, Any]]:
    trace_path = data_dir / f"qa_{dataset}_traces.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"MemoryOS dataset not found: {trace_path}")
    with trace_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a list in {trace_path}, got {type(loaded)!r}")
    return loaded


def extract_memory_chunks(
    *,
    qa_trace: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    unique_chunks: dict[str, dict[str, Any]],
) -> list[str]:
    chunk_ids: list[str] = []
    conversation_id = str(qa_trace.get("sample_id", "unknown"))
    long_term = qa_trace.get("long_term") or {}
    mid_term = qa_trace.get("mid_term") or {}
    short_term = qa_trace.get("short_term") or {}

    def register_chunk(text: str, *, prefix: str, layer: str, chunk_type: str, suffix: str = "") -> None:
        normalized_text = str(text or "")
        if not normalized_text:
            return
        chunk_id = f"{prefix}_{hash_text(normalized_text + suffix)}"
        if chunk_id not in unique_chunks:
            token_ids = list(tokenizer.encode(normalized_text, add_special_tokens=False))
            unique_chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "text": normalized_text,
                "token_ids": token_ids,
                "tokens": len(token_ids),
                "layer": layer,
                "type": chunk_type,
                "conversation_id": conversation_id,
            }
        chunk_ids.append(chunk_id)

    if long_term.get("user_profile_used"):
        register_chunk(
            str(long_term["user_profile_used"]),
            prefix="long_profile",
            layer="long_term",
            chunk_type="user_profile",
        )

    for knowledge in long_term.get("knowledge_retrieved") or []:
        text = knowledge.get("knowledge", "") if isinstance(knowledge, dict) else str(knowledge)
        register_chunk(text, prefix="long_knowledge", layer="long_term", chunk_type="knowledge")

    for knowledge in long_term.get("assistant_knowledge") or []:
        text = knowledge.get("knowledge", "") if isinstance(knowledge, dict) else str(knowledge)
        register_chunk(
            text,
            prefix="long_asst",
            layer="long_term",
            chunk_type="assistant_knowledge",
        )

    for page in mid_term.get("retrieved_pages") or []:
        user_input = str(page.get("user_input", ""))
        agent_response = str(page.get("agent_response", ""))
        timestamp = str(page.get("timestamp", ""))
        text = f"User: {user_input}\nAssistant: {agent_response}"
        register_chunk(
            text,
            prefix="mid",
            layer="mid_term",
            chunk_type="retrieved_page",
            suffix=timestamp,
        )

    for page in short_term.get("pages") or []:
        user_input = str(page.get("user_input", ""))
        agent_response = str(page.get("agent_response", ""))
        timestamp = str(page.get("timestamp", ""))
        text = f"User: {user_input}\nAssistant: {agent_response}"
        register_chunk(
            text,
            prefix="short",
            layer="short_term",
            chunk_type="page",
            suffix=timestamp,
        )

    return chunk_ids


def reorder_chunk_ids(
    *,
    chunk_ids: list[str],
    unique_chunks: dict[str, dict[str, Any]],
    dataset_tag: str,
    qa_index: int,
    policy: str,
    shuffle_seed: int,
) -> list[str]:
    ordered = list(chunk_ids)
    if len(ordered) <= 1:
        return ordered

    def deterministic_shuffle(values: list[str]) -> list[str]:
        seed_material = (
            f"{int(shuffle_seed)}::{dataset_tag}::{int(qa_index)}::{len(values)}::{policy}"
        )
        rng = random.Random(seed_material)
        shuffled = list(values)
        rng.shuffle(shuffled)
        return shuffled

    if policy == "trace":
        return ordered
    if policy == "shuffle":
        return deterministic_shuffle(ordered)

    profile_ids = [
        chunk_id
        for chunk_id in ordered
        if str(unique_chunks[chunk_id].get("type")) == "user_profile"
    ]
    other_ids = [
        chunk_id
        for chunk_id in ordered
        if str(unique_chunks[chunk_id].get("type")) != "user_profile"
    ]

    if policy == "profile_last":
        return other_ids + profile_ids
    if policy == "profile_last_shuffle":
        return deterministic_shuffle(other_ids) + profile_ids

    raise ValueError(f"Unsupported fragment order policy: {policy}")


def build_prompt_token_ids(
    *,
    tokenizer: PreTrainedTokenizerBase,
    system_prompt_ids: list[int],
    blend_special_ids: list[int],
    fragment_token_ids: list[list[int]],
    question_text: str,
    prompt_layout: str,
) -> list[int]:
    prompt_ids: list[int] = []
    if system_prompt_ids:
        prompt_ids.extend(system_prompt_ids)

    if prompt_layout == "memory_first":
        append_memory_first_segments(
            prompt_ids=prompt_ids,
            fragment_token_ids=fragment_token_ids,
            blend_special_ids=blend_special_ids,
            question_token_ids=list(
                tokenizer.encode(
                    f"Question: {question_text}\nAnswer:",
                    add_special_tokens=False,
                )
            ),
        )
        return prompt_ids

    if prompt_layout == "question_first":
        append_question_first_segments(
            prompt_ids=prompt_ids,
            question_prefix_ids=list(
                tokenizer.encode(f"Question: {question_text}\n", add_special_tokens=False)
            ),
            answer_suffix_ids=list(
                tokenizer.encode("\nAnswer:", add_special_tokens=False)
            ),
            fragment_token_ids=fragment_token_ids,
            blend_special_ids=blend_special_ids,
        )
        return prompt_ids

    raise ValueError(
        f"Unsupported prompt_layout={prompt_layout!r}; choose one of {PROMPT_LAYOUT_CHOICES}"
    )


def estimate_memory_prefix_tokens(
    *,
    tokenizer: PreTrainedTokenizerBase,
    system_prompt_ids: list[int],
    blend_special_ids: list[int],
    fragment_token_ids: list[list[int]],
    question_text: str,
    prompt_layout: str,
) -> int:
    """Estimate the theoretical reusable prefix length for the query prompt."""

    if prompt_layout == "memory_first":
        question_token_ids = list(
            tokenizer.encode(
                f"Question: {question_text}\nAnswer:",
                add_special_tokens=False,
            )
        )
        prompt_ids = build_prompt_token_ids(
            tokenizer=tokenizer,
            system_prompt_ids=system_prompt_ids,
            blend_special_ids=blend_special_ids,
            fragment_token_ids=fragment_token_ids,
            question_text=question_text,
            prompt_layout=prompt_layout,
        )
        return max(len(prompt_ids) - len(question_token_ids), 0)

    if prompt_layout == "question_first":
        prompt_ids = list(system_prompt_ids)
        question_prefix_ids = list(
            tokenizer.encode(f"Question: {question_text}\n", add_special_tokens=False)
        )
        append_token_segment(prompt_ids, question_prefix_ids, skip_bos=bool(prompt_ids))
        return len(prompt_ids)

    raise ValueError(
        f"Unsupported prompt_layout={prompt_layout!r}; choose one of {PROMPT_LAYOUT_CHOICES}"
    )


def append_memory_first_segments(
    *,
    prompt_ids: list[int],
    fragment_token_ids: list[list[int]],
    blend_special_ids: list[int],
    question_token_ids: list[int],
) -> None:
    for fragment_ids in fragment_token_ids:
        prompt_ids.extend(blend_special_ids)
        prompt_ids.extend(fragment_ids)
    if fragment_token_ids:
        prompt_ids.extend(blend_special_ids)
    append_token_segment(prompt_ids, list(question_token_ids), skip_bos=False)


def append_question_first_segments(
    *,
    prompt_ids: list[int],
    question_prefix_ids: list[int],
    answer_suffix_ids: list[int],
    fragment_token_ids: list[list[int]],
    blend_special_ids: list[int],
) -> None:
    append_token_segment(prompt_ids, list(question_prefix_ids), skip_bos=False)
    for fragment_ids in fragment_token_ids:
        prompt_ids.extend(blend_special_ids)
        prompt_ids.extend(fragment_ids)
    if fragment_token_ids:
        prompt_ids.extend(blend_special_ids)
    append_token_segment(prompt_ids, list(answer_suffix_ids), skip_bos=False)


def append_token_segment(dst: list[int], token_ids: list[int], *, skip_bos: bool) -> None:
    if not token_ids:
        return
    if skip_bos:
        dst.extend(token_ids[1:])
    else:
        dst.extend(token_ids)


def build_system_prompt_ids(
    *,
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
) -> list[int]:
    bos_id = tokenizer.bos_token_id
    prompt_ids: list[int] = []
    if bos_id is not None:
        prompt_ids.append(int(bos_id))
    if system_prompt:
        prompt_ids.extend(
            tokenizer.encode(system_prompt, add_special_tokens=False)
        )
    return prompt_ids


def build_prefill_records(
    *,
    tokenizer: PreTrainedTokenizerBase,
    unique_chunks: dict[str, dict[str, Any]],
    prefill_order: list[str],
    system_prompt_ids: list[int],
    blend_special_ids: list[int],
    prefill_query_text: str,
    prompt_layout: str,
    max_model_len: int,
    prefill_plan: dict[str, dict[str, Any]] | None = None,
) -> list[PrefillRecord]:
    prefill_records: list[PrefillRecord] = []
    for chunk_id in prefill_order:
        if prefill_plan is not None:
            plan = prefill_plan.get(chunk_id)
            if plan is None or not plan.get("enabled", False):
                continue
        else:
            plan = None
        chunk = unique_chunks[chunk_id]
        prompt_ids = build_prompt_token_ids(
            tokenizer=tokenizer,
            system_prompt_ids=system_prompt_ids,
            blend_special_ids=blend_special_ids,
            fragment_token_ids=[list(chunk["token_ids"])],
            question_text=prefill_query_text,
            prompt_layout=prompt_layout,
        )
        if len(prompt_ids) > max_model_len:
            raise ValueError(
                "Prefill prompt length exceeds max_model_len. "
                f"chunk_id={chunk_id}, prompt_tokens={len(prompt_ids)}, "
                f"max_model_len={max_model_len}."
            )
        prefill_records.append(
            PrefillRecord(
                chunk_id=chunk_id,
                chunk_type=str(chunk.get("type", "unknown")),
                prompt_ids=prompt_ids,
                prompt_tokens=len(prompt_ids),
                store_location=(
                    str(plan.get("target_location"))
                    if plan is not None and plan.get("target_location") is not None
                    else None
                ),
                utility_action=(
                    str(plan.get("action"))
                    if plan is not None and plan.get("action") is not None
                    else None
                ),
                utility_score=(
                    float(plan.get("utility"))
                    if plan is not None and plan.get("utility") is not None
                    else None
                ),
            )
        )
    return prefill_records


def build_workload_stats(
    *,
    datasets: list[str],
    query_records: list[QueryRecord],
    unique_chunks: dict[str, dict[str, Any]],
    prompt_layout: str,
    fragment_order_policy: str,
    shuffle_seed: int,
    system_prompt_tokens: int,
    blend_special_str: str,
) -> WorkloadStats:
    fragment_counts = [len(record.chunk_ids) for record in query_records]
    prompt_lengths = [record.prompt_tokens for record in query_records]
    memory_prefix_lengths = [record.memory_prefix_tokens for record in query_records]
    total_unique_fragment_tokens = sum(
        int(chunk.get("tokens", 0)) for chunk in unique_chunks.values()
    )
    return WorkloadStats(
        datasets=list(datasets),
        query_count=len(query_records),
        unique_fragments=len(unique_chunks),
        total_unique_fragment_tokens=total_unique_fragment_tokens,
        mean_fragments_per_query=_safe_mean(fragment_counts),
        p50_fragments_per_query=percentile(fragment_counts, 0.50),
        p90_fragments_per_query=percentile(fragment_counts, 0.90),
        mean_prompt_tokens=_safe_mean(prompt_lengths),
        p50_prompt_tokens=percentile(prompt_lengths, 0.50),
        p90_prompt_tokens=percentile(prompt_lengths, 0.90),
        max_prompt_tokens=max(prompt_lengths) if prompt_lengths else 0,
        mean_memory_prefix_tokens=_safe_mean(memory_prefix_lengths),
        p50_memory_prefix_tokens=percentile(memory_prefix_lengths, 0.50),
        p90_memory_prefix_tokens=percentile(memory_prefix_lengths, 0.90),
        max_memory_prefix_tokens=max(memory_prefix_lengths) if memory_prefix_lengths else 0,
        prompt_layout=prompt_layout,
        fragment_order_policy=fragment_order_policy,
        shuffle_seed=int(shuffle_seed),
        system_prompt_tokens=system_prompt_tokens,
        blend_special_str=blend_special_str,
    )


def run_online_workload(
    *,
    args: argparse.Namespace,
    query_records: list[QueryRecord],
    warmup_prompt_ids: list[int],
    enable_prefix_caching: bool,
    enable_blend: bool,
    lmcache_env: dict[str, str] | None = None,
) -> LatencySummary:
    import requests

    session = requests.Session()
    session.trust_env = False

    measurements: list[RequestMeasurement] = []
    with launch_server(
        model=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_prefix_caching=enable_prefix_caching,
        enable_blend=enable_blend,
        startup_timeout_s=args.startup_timeout_s,
        enforce_eager=args.enforce_eager,
        cuda_visible_devices=args.cuda_visible_devices,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        lmcache_env=lmcache_env,
    ) as server:
        measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=warmup_prompt_ids,
            max_tokens=args.max_tokens,
        )

        for query_index, record in enumerate(query_records):
            try:
                measurements.append(
                    measure_streaming_request(
                        session=session,
                        port=args.port,
                        model=args.model,
                        prompt_ids=record.prompt_ids,
                        max_tokens=args.max_tokens,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Online workload request failed.\n"
                    f"dataset={record.dataset}\n"
                    f"qa_index={record.qa_index}\n"
                    f"query_index={query_index}\n"
                    f"sample_id={record.sample_id}\n"
                    f"prompt_tokens={record.prompt_tokens}\n"
                    f"server_tail:\n{server.tail_text()}"
                ) from exc

    return summarize_measurements(
        measurements=measurements,
        prompt_tokens=[record.prompt_tokens for record in query_records],
    )


def build_blend_lmcache_env(
    *,
    args: argparse.Namespace,
    chunk_size: int,
    backend: str,
    phase_file: Path,
) -> dict[str, str]:
    env = {
        "LMCACHE_CHUNK_SIZE": str(chunk_size),
        "LMCACHE_ENABLE_BLENDING": "True",
        "LMCACHE_BLEND_SPECIAL_STR": str(args.blend_special_str),
        "LMCACHE_SAVE_UNFULL_CHUNK": "True",
        "LMCACHE_USE_LAYERWISE": "True",
        "LMCACHE_BLEND_CHECK_LAYERS": str(args.blend_check_layers),
        "LMCACHE_BLEND_RECOMPUTE_RATIOS": str(args.blend_recompute_ratios),
        "LMCACHE_PHASE_TIMING_PATH": str(phase_file),
        "LMCACHE_LOCAL_CPU": "False",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "0",
        "LMCACHE_LOCAL_GPU": "False",
        "LMCACHE_MAX_LOCAL_GPU_SIZE": "0",
    }
    if backend == "cpu":
        env["LMCACHE_LOCAL_CPU"] = "True"
        env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(args.max_local_cpu_size)
    elif backend == "gpu":
        env["LMCACHE_LOCAL_GPU"] = "True"
        env["LMCACHE_MAX_LOCAL_GPU_SIZE"] = str(args.max_local_gpu_size)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    extra_config = {}
    if args.blend_connector_impl:
        extra_config["blend_connector_impl"] = str(args.blend_connector_impl)
    extra_config["blend_internal_timing"] = bool(args.blend_internal_timing)
    extra_config["blend_pipeline_buffers"] = int(args.blend_pipeline_buffers)
    extra_config["blend_buffer_bucket_tokens"] = int(
        args.blend_buffer_bucket_tokens
    )
    extra_config["blend_max_cached_buffer_packs"] = int(
        args.blend_max_cached_buffer_packs
    )
    env["LMCACHE_EXTRA_CONFIG"] = json.dumps(extra_config)

    return env


def run_blend_workload(
    *,
    args: argparse.Namespace,
    query_records: list[QueryRecord],
    prefill_records: list[PrefillRecord],
    warmup_prompt_ids: list[int],
    chunk_size: int,
    backend: str,
) -> BlendChunkResult:
    import requests
    import time

    session = requests.Session()
    session.trust_env = False
    phase_file = Path(
        tempfile.NamedTemporaryFile(
            prefix=f"lmcache_{backend}_phase_chunk{chunk_size}_",
            suffix=".jsonl",
            delete=False,
        ).name
    )

    prefill_measurements: list[RequestMeasurement] = []
    query_measurements: list[RequestMeasurement] = []

    lmcache_env = build_blend_lmcache_env(
        args=args,
        chunk_size=chunk_size,
        backend=backend,
        phase_file=phase_file,
    )

    with launch_server(
        model=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_prefix_caching=False,
        enable_blend=True,
        startup_timeout_s=args.startup_timeout_s,
        enforce_eager=args.enforce_eager,
        cuda_visible_devices=args.cuda_visible_devices,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        lmcache_env=lmcache_env,
    ) as server:
        measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=warmup_prompt_ids,
            max_tokens=args.max_tokens,
        )

        prefill_start = time.perf_counter()
        for prefill_index, record in enumerate(prefill_records):
            try:
                prefill_measurements.append(
                    measure_streaming_request(
                        session=session,
                        port=args.port,
                        model=args.model,
                        prompt_ids=record.prompt_ids,
                        max_tokens=args.max_tokens,
                        kv_transfer_params={
                            "lmcache.request_kind": "fragment_prefill",
                        },
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Blend prefill request failed.\n"
                    f"prefill_index={prefill_index}\n"
                    f"chunk_id={record.chunk_id}\n"
                    f"chunk_type={record.chunk_type}\n"
                    f"prompt_tokens={record.prompt_tokens}\n"
                    f"server_tail:\n{server.tail_text()}"
                ) from exc
        prefill_wall_s = time.perf_counter() - prefill_start

        for query_index, record in enumerate(query_records):
            try:
                query_measurements.append(
                    measure_streaming_request(
                        session=session,
                        port=args.port,
                        model=args.model,
                        prompt_ids=record.prompt_ids,
                        max_tokens=args.max_tokens,
                        kv_transfer_params={
                            "lmcache.request_kind": "online_blended_query",
                        },
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Blend online query failed.\n"
                    f"backend={backend}\n"
                    f"chunk_size={chunk_size}\n"
                    f"dataset={record.dataset}\n"
                    f"qa_index={record.qa_index}\n"
                    f"query_index={query_index}\n"
                    f"sample_id={record.sample_id}\n"
                    f"prompt_tokens={record.prompt_tokens}\n"
                    f"server_tail:\n{server.tail_text()}"
                ) from exc

    online_phase_breakdown = summarize_online_phase_breakdown(
        phase_file=phase_file,
        measurements=query_measurements,
    )

    prefill_summary = summarize_prefill(
        measurements=prefill_measurements,
        prompt_tokens=[record.prompt_tokens for record in prefill_records],
        wall_s=prefill_wall_s,
    )
    online_summary = summarize_measurements(
        measurements=query_measurements,
        prompt_tokens=[record.prompt_tokens for record in query_records],
    )
    first_query_ttft_s = 0.0
    if query_measurements:
        first_query_ttft_s = query_measurements[0].ttft_s or 0.0
    return BlendChunkResult(
        backend=backend,
        chunk_size=chunk_size,
        prefill=prefill_summary,
        online=online_summary,
        online_phase_breakdown=online_phase_breakdown,
        end_to_end_total_wall_s=prefill_summary.wall_s + online_summary.total_wall_s,
        end_to_end_total_ttft_s=prefill_summary.wall_s + online_summary.total_ttft_s,
        first_query_total_including_prefill_s=prefill_summary.wall_s + first_query_ttft_s,
    )


def summarize_prefill(
    *,
    measurements: list[RequestMeasurement],
    prompt_tokens: list[int],
    wall_s: float,
) -> PrefillSummary:
    ttfts = [item.ttft_s for item in measurements if item.ttft_s is not None]
    per_request_walls = [item.wall_s for item in measurements]
    return PrefillSummary(
        request_count=len(measurements),
        total_prompt_tokens=int(sum(prompt_tokens)),
        wall_s=float(wall_s),
        mean_wall_s=_safe_mean(per_request_walls),
        p50_wall_s=percentile(per_request_walls, 0.50),
        p90_wall_s=percentile(per_request_walls, 0.90),
        mean_ttft_s=_safe_mean(ttfts),
        p50_ttft_s=percentile(ttfts, 0.50),
        p90_ttft_s=percentile(ttfts, 0.90),
    )


def summarize_measurements(
    *,
    measurements: list[RequestMeasurement],
    prompt_tokens: list[int],
) -> LatencySummary:
    ttfts = [item.ttft_s for item in measurements if item.ttft_s is not None]
    walls = [item.wall_s for item in measurements]
    cached = [int(item.cached_tokens) for item in measurements if item.cached_tokens is not None]
    cache_hit_requests = sum(
        1
        for item in measurements
        if item.cached_tokens is not None and int(item.cached_tokens) > 0
    )
    return LatencySummary(
        request_count=len(measurements),
        total_ttft_s=float(sum(ttfts)),
        mean_ttft_s=_safe_mean(ttfts),
        p50_ttft_s=percentile(ttfts, 0.50),
        p90_ttft_s=percentile(ttfts, 0.90),
        max_ttft_s=max(ttfts) if ttfts else 0.0,
        total_wall_s=float(sum(walls)),
        mean_wall_s=_safe_mean(walls),
        p50_wall_s=percentile(walls, 0.50),
        p90_wall_s=percentile(walls, 0.90),
        max_wall_s=max(walls) if walls else 0.0,
        mean_prompt_tokens=_safe_mean(prompt_tokens),
        mean_cached_tokens=_safe_mean(cached) if cached else None,
        p50_cached_tokens=percentile(cached, 0.50) if cached else None,
        p90_cached_tokens=percentile(cached, 0.90) if cached else None,
        cache_hit_requests=cache_hit_requests,
        cache_hit_rate=(cache_hit_requests / len(measurements)) if measurements else 0.0,
    )


def summarize_online_phase_breakdown(
    *,
    phase_file: Path,
    measurements: list[RequestMeasurement],
) -> dict[str, PhaseSummary]:
    request_ids = [m.request_id for m in measurements if m.request_id]
    if not request_ids or not phase_file.exists():
        return {}

    phase_records = load_phase_records(phase_file)
    phase_to_request_duration = build_phase_duration_map(
        request_ids=request_ids,
        phase_records=phase_records,
    )
    add_derived_phase_durations(phase_to_request_duration, request_ids)
    return {
        phase: summarize_phase_values(
            [phase_to_request_duration[phase].get(req_id, 0.0) for req_id in request_ids]
        )
        for phase in sorted(phase_to_request_duration)
    }


def load_phase_records(phase_file: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in phase_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def build_phase_duration_map(
    *,
    request_ids: list[str],
    phase_records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    normalized_to_original = {
        normalize_request_id(req_id): req_id for req_id in request_ids
    }
    request_id_set = set(normalized_to_original)
    phase_to_request_duration: dict[str, dict[str, float]] = {}
    for record in phase_records:
        req_id = normalize_request_id(str(record.get("req_id", "")))
        if req_id not in request_id_set:
            continue
        original_req_id = normalized_to_original[req_id]
        phase = str(record.get("phase", ""))
        duration_s = float(record.get("duration_s", 0.0))
        if phase not in phase_to_request_duration:
            phase_to_request_duration[phase] = {}
        phase_to_request_duration[phase][original_req_id] = (
            phase_to_request_duration[phase].get(original_req_id, 0.0) + duration_s
        )
    return phase_to_request_duration


def normalize_request_id(req_id: str) -> str:
    return re.sub(r"-\d+$", "", req_id)


def add_derived_phase_durations(
    phase_to_request_duration: dict[str, dict[str, float]],
    request_ids: list[str],
) -> None:
    def get_duration(phase: str, req_id: str) -> float:
        return phase_to_request_duration.get(phase, {}).get(req_id, 0.0)

    blend_compute_only: dict[str, float] = {}
    retrieve_other: dict[str, float] = {}
    to_gpu_other: dict[str, float] = {}
    has_to_gpu_internal_breakdown = any(
        phase in phase_to_request_duration
        for phase in (
            "to_gpu_buffer_load_copy_s",
            "to_gpu_paged_kv_transfer_s",
            "to_gpu_rope_recover_s",
        )
    )

    for req_id in request_ids:
        blend_compute_only[req_id] = max(
            get_duration("blend_total_s", req_id)
            - get_duration("retrieve_total_s", req_id),
            0.0,
        )
        retrieve_other[req_id] = max(
            get_duration("retrieve_total_s", req_id)
            - get_duration("retrieve_prepare_s", req_id)
            - get_duration("retrieve_storage_wait_s", req_id)
            - get_duration("to_gpu_total_s", req_id),
            0.0,
        )
        if has_to_gpu_internal_breakdown:
            to_gpu_other[req_id] = max(
                get_duration("to_gpu_total_s", req_id)
                - get_duration("to_gpu_buffer_load_copy_s", req_id)
                - get_duration("to_gpu_paged_kv_transfer_s", req_id)
                - get_duration("to_gpu_rope_recover_s", req_id),
                0.0,
            )

    phase_to_request_duration["blend_compute_only_s"] = blend_compute_only
    phase_to_request_duration["retrieve_other_s"] = retrieve_other
    if has_to_gpu_internal_breakdown:
        phase_to_request_duration["to_gpu_other_s"] = to_gpu_other


def summarize_phase_values(values: list[float]) -> PhaseSummary:
    return PhaseSummary(
        sample_count=len(values),
        total_s=float(sum(values)),
        mean_s=_safe_mean(values),
        p50_s=percentile(values, 0.50),
        p90_s=percentile(values, 0.90),
        max_s=max(values) if values else 0.0,
    )


def percentile(values: Iterable[float | int], ratio: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(ratio)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_mean(values: Iterable[float | int]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def print_workload_stats(stats: WorkloadStats) -> None:
    print("=" * 80)
    print("MemoryOS Workload")
    print("=" * 80)
    print(f"datasets: {', '.join(stats.datasets)}")
    print(f"query_count: {stats.query_count}")
    print(f"unique_fragments: {stats.unique_fragments}")
    print(f"total_unique_fragment_tokens: {stats.total_unique_fragment_tokens}")
    print(
        "fragments_per_query: "
        f"mean={stats.mean_fragments_per_query:.2f}, "
        f"p50={stats.p50_fragments_per_query:.2f}, "
        f"p90={stats.p90_fragments_per_query:.2f}"
    )
    print(
        "prompt_tokens: "
        f"mean={stats.mean_prompt_tokens:.1f}, "
        f"p50={stats.p50_prompt_tokens:.1f}, "
        f"p90={stats.p90_prompt_tokens:.1f}, "
        f"max={stats.max_prompt_tokens}"
    )
    print(
        "theoretical_memory_prefix_tokens: "
        f"mean={stats.mean_memory_prefix_tokens:.1f}, "
        f"p50={stats.p50_memory_prefix_tokens:.1f}, "
        f"p90={stats.p90_memory_prefix_tokens:.1f}, "
        f"max={stats.max_memory_prefix_tokens}"
    )
    print(f"prompt_layout: {stats.prompt_layout}")
    print(f"fragment_order_policy: {stats.fragment_order_policy}")
    print(f"shuffle_seed: {stats.shuffle_seed}")
    print(f"system_prompt_tokens: {stats.system_prompt_tokens}")
    print(f"blend_special_str: {stats.blend_special_str!r}")
    print("=" * 80)


def print_result(result: BenchmarkResult) -> None:
    print()
    print("=" * 80)
    print("MemoryOS Prefix vs CacheBlend Streaming TTFT Benchmark")
    print("=" * 80)
    print(f"model: {result.model}")
    print(f"data_dir: {result.data_dir}")
    print_mode_summary("plain vLLM (no prefix)", result.no_prefix)
    print_mode_summary("native vLLM prefix caching", result.native_prefix)

    for item in result.blend_results:
        print()
        print(
            f"[LMCache CacheBlend {item.backend.upper()} chunk={item.chunk_size}]"
        )
        print(
            "prefill: "
            f"requests={item.prefill.request_count}, "
            f"wall_s={item.prefill.wall_s:.4f}, "
            f"mean_wall_s={item.prefill.mean_wall_s:.4f}, "
            f"p50_wall_s={item.prefill.p50_wall_s:.4f}, "
            f"p90_wall_s={item.prefill.p90_wall_s:.4f}"
        )
        print(
            "prefill_ttft: "
            f"mean={item.prefill.mean_ttft_s:.4f}, "
            f"p50={item.prefill.p50_ttft_s:.4f}, "
            f"p90={item.prefill.p90_ttft_s:.4f}"
        )
        print_mode_summary("online query", item.online, print_header=False)
        print_phase_breakdown(item.online_phase_breakdown)
        print(
            "including_prefill: "
            f"first_query_total_s={item.first_query_total_including_prefill_s:.4f}, "
            f"workload_total_ttft_s={item.end_to_end_total_ttft_s:.4f}, "
            f"workload_total_wall_s={item.end_to_end_total_wall_s:.4f}"
        )
    print("=" * 80)


def print_mode_summary(title: str, summary: LatencySummary, *, print_header: bool = True) -> None:
    if print_header:
        print()
        print(f"[{title}]")
    else:
        print(f"[{title}]")
    print(
        "ttft_s: "
        f"mean={summary.mean_ttft_s:.4f}, "
        f"p50={summary.p50_ttft_s:.4f}, "
        f"p90={summary.p90_ttft_s:.4f}, "
        f"max={summary.max_ttft_s:.4f}, "
        f"total={summary.total_ttft_s:.4f}"
    )
    print(
        "wall_s: "
        f"mean={summary.mean_wall_s:.4f}, "
        f"p50={summary.p50_wall_s:.4f}, "
        f"p90={summary.p90_wall_s:.4f}, "
        f"max={summary.max_wall_s:.4f}, "
        f"total={summary.total_wall_s:.4f}"
    )
    mean_cached = (
        format_optional_float(summary.mean_cached_tokens)
        if summary.mean_cached_tokens is not None
        else "n/a"
    )
    p50_cached = (
        format_optional_float(summary.p50_cached_tokens)
        if summary.p50_cached_tokens is not None
        else "n/a"
    )
    p90_cached = (
        format_optional_float(summary.p90_cached_tokens)
        if summary.p90_cached_tokens is not None
        else "n/a"
    )
    print(
        "cached_tokens: "
        f"mean={mean_cached}, "
        f"p50={p50_cached}, "
        f"p90={p90_cached}, "
        f"hit_requests={summary.cache_hit_requests}/{summary.request_count}, "
        f"hit_rate={summary.cache_hit_rate:.2%}"
    )
    print(f"mean_prompt_tokens: {summary.mean_prompt_tokens:.1f}")


def print_phase_breakdown(phase_breakdown: dict[str, PhaseSummary]) -> None:
    if not phase_breakdown:
        return

    major_order = [
        "lookup_total_s",
        "blend_total_s",
        "retrieve_total_s",
        "blend_compute_only_s",
    ]
    detail_order = [
        "retrieve_prepare_s",
        "retrieve_storage_wait_s",
        "retrieve_gpu_send_s",
        "retrieve_other_s",
        "to_gpu_total_s",
        "to_gpu_buffer_load_copy_s",
        "to_gpu_paged_kv_transfer_s",
        "to_gpu_rope_recover_s",
        "to_gpu_other_s",
    ]

    print("[online phase breakdown]")
    for phase in major_order + detail_order:
        summary = phase_breakdown.get(phase)
        if summary is None:
            continue
        print(
            f"{phase}: "
            f"mean={summary.mean_s:.4f}, "
            f"p50={summary.p50_s:.4f}, "
            f"p90={summary.p90_s:.4f}, "
            f"max={summary.max_s:.4f}, "
            f"total={summary.total_s:.4f}"
        )


if __name__ == "__main__":
    main()
