# SPDX-License-Identifier: Apache-2.0
"""Benchmark vLLM prefix caching vs LMCache CacheBlend on real workloads.

This benchmark intentionally compares only GPU-resident reusable cache budgets:

1. Native vLLM prefix caching with an explicit `kv_cache_memory_bytes` cap.
2. LMCache CacheBlend with `LocalGPUBackend` enabled and `LocalCPUBackend`
   disabled.

The script supports three processed workload formats under
`example/benchmark_e2e/data/processed`:
- `memoryos`
- `amem`
- `memos`

For the Blend path, the offline fragment prefill stage stores reusable
fragment KV in the LMCache GPU backend. Online TTFT is measured from request
submission to the first streamed token and therefore includes online fragment
lookup / retrieval / injection overhead.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from compare_prefix_vs_blend_cpu_server import (
    RequestMeasurement,
    format_optional_float,
    launch_server,
    measure_streaming_request,
)
from compare_prefix_vs_blend_memoryos_server import (
    BlendChunkResult,
    LatencySummary,
    PrefillRecord,
    QueryRecord,
    WorkloadStats,
    append_token_segment,
    build_prefill_records,
    build_prompt_token_ids,
    build_system_prompt_ids,
    build_workload_stats,
    estimate_memory_prefix_tokens,
    hash_text,
    print_phase_breakdown,
    print_mode_summary,
    reorder_chunk_ids,
    summarize_measurements,
    summarize_online_phase_breakdown,
    summarize_prefill,
)


WORKLOAD_KIND_CHOICES = ("memoryos", "amem", "memos")
PROMPT_LAYOUT_CHOICES = ("memory_first", "question_first")
FRAGMENT_ORDER_POLICY_CHOICES = (
    "auto",
    "trace",
    "shuffle",
    "profile_last",
    "profile_last_shuffle",
)
PREFILL_ORDER_POLICY_CHOICES = ("sorted", "first_seen", "shuffle")

DEFAULT_DATASETS_BY_WORKLOAD = {
    "memoryos": ["conv26", "conv30", "conv41"],
    "amem": ["conv26", "conv30", "conv41"],
    "memos": ["locomo_conv0", "locomo_conv1", "locomo_conv2"],
}


@dataclass
class BenchmarkResult:
    model: str
    data_root: str
    workload_kind: str
    datasets: list[str]
    prefix_kv_cache_bytes: int
    blend_vllm_kv_cache_bytes: int
    blend_local_gpu_size_gb: float
    chunk_sizes: list[int]
    prefill_order_policy: str
    workload: WorkloadStats
    native_prefix: LatencySummary
    blend_gpu_results: list[BlendChunkResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare vLLM native prefix caching vs LMCache CacheBlend GPU "
            "under explicit GPU cache budgets on real workloads."
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
        default=(
            "/home/mahaoran/research/compoundai/CacheBlend/example/"
            "benchmark_e2e/data/processed"
        ),
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
    parser.add_argument(
        "--max-qa-per-dataset",
        type=int,
        default=0,
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
        default="auto",
        choices=list(FRAGMENT_ORDER_POLICY_CHOICES),
        help=(
            "'auto' resolves to 'profile_last' for memoryos and 'trace' for "
            "amem/memos."
        ),
    )
    parser.add_argument(
        "--prefill-order-policy",
        type=str,
        default="sorted",
        choices=list(PREFILL_ORDER_POLICY_CHOICES),
        help=(
            "Ordering of the offline unique-fragment prefill pass. "
            "'sorted' is the fairest default."
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Base seed for deterministic fragment/prefill reordering.",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="512",
        help="Comma-separated LMCache chunk sizes to test on the Blend GPU path.",
    )
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
        "--prefix-kv-cache-gb",
        type=float,
        default=2.0,
        help="Native vLLM prefix-caching KV pool size in GiB.",
    )
    parser.add_argument(
        "--blend-vllm-kv-cache-gb",
        type=float,
        default=2.0,
        help="Serving-time vLLM paged-KV pool size for the Blend run in GiB.",
    )
    parser.add_argument(
        "--max-local-gpu-size",
        type=float,
        default=2.0,
        help="LMCache LocalGPUBackend budget in GiB.",
    )
    parser.add_argument("--blend-check-layers", type=str, default="1")
    parser.add_argument("--blend-recompute-ratios", type=str, default="0.15")
    parser.add_argument(
        "--blend-connector-impl",
        type=str,
        default="fast",
        help="LMCache blend connector implementation name.",
    )
    parser.add_argument("--port", type=int, default=8015)
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
    chunk_sizes = parse_chunk_sizes(args.chunk_sizes)
    datasets = resolve_datasets(
        workload_kind=str(args.workload_kind),
        raw=args.datasets,
    )
    fragment_order_policy = resolve_fragment_order_policy(
        workload_kind=str(args.workload_kind),
        raw_policy=str(args.fragment_order_policy),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
    )

    print_workload_stats(
        workload_kind=str(args.workload_kind),
        stats=workload["stats"],
        prefill_order_policy=str(args.prefill_order_policy),
    )
    if args.dry_run:
        return

    prefix_kv_cache_bytes = gib_to_bytes(float(args.prefix_kv_cache_gb))
    blend_vllm_kv_cache_bytes = gib_to_bytes(float(args.blend_vllm_kv_cache_gb))

    native_prefix = run_native_prefix_workload(
        args=args,
        query_records=workload["query_records"],
        warmup_prompt_ids=workload["warmup_prompt_ids"],
        kv_cache_memory_bytes=prefix_kv_cache_bytes,
    )

    blend_gpu_results: list[BlendChunkResult] = []
    for chunk_size in chunk_sizes:
        blend_gpu_results.append(
            run_blend_gpu_workload(
                args=args,
                query_records=workload["query_records"],
                prefill_records=workload["prefill_records"],
                warmup_prompt_ids=workload["warmup_prompt_ids"],
                chunk_size=chunk_size,
                vllm_kv_cache_memory_bytes=blend_vllm_kv_cache_bytes,
            )
        )

    result = BenchmarkResult(
        model=str(args.model),
        data_root=str(args.data_root),
        workload_kind=str(args.workload_kind),
        datasets=datasets,
        prefix_kv_cache_bytes=prefix_kv_cache_bytes,
        blend_vllm_kv_cache_bytes=blend_vllm_kv_cache_bytes,
        blend_local_gpu_size_gb=float(args.max_local_gpu_size),
        chunk_sizes=chunk_sizes,
        prefill_order_policy=str(args.prefill_order_policy),
        workload=workload["stats"],
        native_prefix=native_prefix,
        blend_gpu_results=blend_gpu_results,
    )

    print_result(result)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(result), file, indent=2)
            file.write("\n")


def parse_chunk_sizes(raw: str) -> list[int]:
    chunk_sizes = [int(value) for value in str(raw).split(",") if value]
    if not chunk_sizes:
        raise ValueError("chunk_sizes cannot be empty.")
    return chunk_sizes


def parse_datasets(raw: str) -> list[str]:
    datasets = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not datasets:
        raise ValueError("datasets cannot be empty.")
    return datasets


def resolve_datasets(workload_kind: str, raw: str) -> list[str]:
    if raw.strip():
        return parse_datasets(raw)
    return list(DEFAULT_DATASETS_BY_WORKLOAD[workload_kind])


def resolve_fragment_order_policy(workload_kind: str, raw_policy: str) -> str:
    if raw_policy != "auto":
        return raw_policy
    if workload_kind == "memoryos":
        return "profile_last"
    return "trace"


def gib_to_bytes(size_gib: float) -> int:
    return int(float(size_gib) * 1024**3)


def build_workload(
    *,
    tokenizer: PreTrainedTokenizerBase,
    data_root: Path,
    workload_kind: str,
    datasets: list[str],
    max_qa_per_dataset: int,
    prompt_layout: str,
    fragment_order_policy: str,
    prefill_order_policy: str,
    shuffle_seed: int,
    blend_special_str: str,
    system_prompt: str,
    max_model_len: int,
    warmup_query_text: str,
    prefill_query_text: str,
) -> dict[str, Any]:
    if not data_root.exists():
        raise FileNotFoundError(f"Processed data root does not exist: {data_root}")

    unique_chunks: dict[str, dict[str, Any]] = {}
    query_records: list[QueryRecord] = []
    prefill_first_seen: list[str] = []
    prefill_seen: set[str] = set()

    system_prompt_ids = build_system_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    blend_special_ids = list(
        tokenizer.encode(blend_special_str, add_special_tokens=False)
    )
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
        traces = load_trace(
            data_root=data_root,
            workload_kind=workload_kind,
            dataset=dataset,
        )
        if max_qa_per_dataset > 0:
            traces = traces[:max_qa_per_dataset]

        memory_index = None
        if workload_kind in ("amem", "memos"):
            memory_index = load_memory_index(
                data_root=data_root,
                workload_kind=workload_kind,
                dataset=dataset,
            )

        for qa_index, qa_trace in enumerate(traces):
            if workload_kind == "memoryos":
                chunk_ids = extract_memoryos_chunks(
                    qa_trace=qa_trace,
                    tokenizer=tokenizer,
                    unique_chunks=unique_chunks,
                )
            else:
                assert memory_index is not None
                chunk_ids = extract_id_backed_chunks(
                    workload_kind=workload_kind,
                    dataset=dataset,
                    qa_trace=qa_trace,
                    tokenizer=tokenizer,
                    unique_chunks=unique_chunks,
                    memory_index=memory_index,
                )

            ordered_chunk_ids = reorder_chunk_ids(
                chunk_ids=chunk_ids,
                unique_chunks=unique_chunks,
                dataset_tag=f"{workload_kind}:{dataset}",
                qa_index=qa_index,
                policy=fragment_order_policy,
                shuffle_seed=shuffle_seed,
            )

            for chunk_id in ordered_chunk_ids:
                if chunk_id in prefill_seen:
                    continue
                prefill_seen.add(chunk_id)
                prefill_first_seen.append(chunk_id)

            fragment_token_ids = [
                list(unique_chunks[chunk_id]["token_ids"])
                for chunk_id in ordered_chunk_ids
            ]
            question_text = str(qa_trace.get("question", ""))
            prompt_ids = build_prompt_token_ids(
                tokenizer=tokenizer,
                system_prompt_ids=system_prompt_ids,
                blend_special_ids=blend_special_ids,
                fragment_token_ids=fragment_token_ids,
                question_text=question_text,
                prompt_layout=prompt_layout,
            )
            memory_prefix_tokens = estimate_memory_prefix_tokens(
                tokenizer=tokenizer,
                system_prompt_ids=system_prompt_ids,
                blend_special_ids=blend_special_ids,
                fragment_token_ids=fragment_token_ids,
                question_text=question_text,
                prompt_layout=prompt_layout,
            )
            if len(prompt_ids) > max_model_len:
                raise ValueError(
                    "Prompt length exceeds max_model_len. "
                    f"workload_kind={workload_kind}, dataset={dataset}, "
                    f"qa_index={qa_index}, prompt_tokens={len(prompt_ids)}, "
                    f"max_model_len={max_model_len}."
                )

            query_records.append(
                QueryRecord(
                    dataset=dataset,
                    qa_index=qa_index,
                    sample_id=resolve_sample_id(
                        workload_kind=workload_kind,
                        dataset=dataset,
                        qa_index=qa_index,
                        qa_trace=qa_trace,
                    ),
                    question=question_text,
                    chunk_ids=ordered_chunk_ids,
                    prompt_ids=prompt_ids,
                    prompt_tokens=len(prompt_ids),
                    memory_prefix_tokens=memory_prefix_tokens,
                )
            )

    prefill_order = apply_prefill_order_policy(
        prefill_first_seen=prefill_first_seen,
        policy=prefill_order_policy,
        shuffle_seed=shuffle_seed,
        workload_kind=workload_kind,
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


def load_trace(
    *,
    data_root: Path,
    workload_kind: str,
    dataset: str,
) -> list[dict[str, Any]]:
    if workload_kind == "memoryos":
        trace_path = data_root / "memoryos" / f"qa_{dataset}_traces.json"
    elif workload_kind == "amem":
        trace_path = data_root / "amem" / f"amem_{dataset}_qa_trace.json"
    elif workload_kind == "memos":
        trace_path = data_root / "memos" / f"memos_{dataset}_qa_trace.json"
    else:
        raise ValueError(f"Unsupported workload_kind={workload_kind!r}")

    if not trace_path.exists():
        raise FileNotFoundError(f"Dataset trace not found: {trace_path}")
    with trace_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a list in {trace_path}, got {type(loaded)!r}")
    return loaded


def load_memory_index(
    *,
    data_root: Path,
    workload_kind: str,
    dataset: str,
) -> dict[str, dict[str, Any]]:
    if workload_kind == "amem":
        memory_path = data_root / "amem" / f"amem_{dataset}_memorys.json"
    elif workload_kind == "memos":
        memory_path = data_root / "memos" / f"memos_{dataset}_memorys.json"
    else:
        raise ValueError(f"Unsupported workload_kind={workload_kind!r}")

    if not memory_path.exists():
        raise FileNotFoundError(f"Dataset memory file not found: {memory_path}")
    with memory_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a list in {memory_path}, got {type(loaded)!r}")

    index: dict[str, dict[str, Any]] = {}
    for item in loaded:
        if not isinstance(item, dict):
            continue
        memory_id = str(item.get("memory_id", "")).strip()
        if not memory_id:
            continue
        index[memory_id] = item
    return index


def extract_memoryos_chunks(
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

    def register_chunk(
        text: str,
        *,
        prefix: str,
        layer: str,
        chunk_type: str,
        suffix: str = "",
    ) -> None:
        normalized_text = str(text or "")
        if not normalized_text:
            return
        chunk_id = f"{prefix}_{hash_text(normalized_text + suffix)}"
        if chunk_id not in unique_chunks:
            token_ids = list(
                tokenizer.encode(normalized_text, add_special_tokens=False)
            )
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
        text = (
            knowledge.get("knowledge", "")
            if isinstance(knowledge, dict)
            else str(knowledge)
        )
        register_chunk(
            text,
            prefix="long_knowledge",
            layer="long_term",
            chunk_type="knowledge",
        )

    for knowledge in long_term.get("assistant_knowledge") or []:
        text = (
            knowledge.get("knowledge", "")
            if isinstance(knowledge, dict)
            else str(knowledge)
        )
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


def extract_id_backed_chunks(
    *,
    workload_kind: str,
    dataset: str,
    qa_trace: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    unique_chunks: dict[str, dict[str, Any]],
    memory_index: dict[str, dict[str, Any]],
) -> list[str]:
    chunk_ids: list[str] = []
    for memory_id in qa_trace.get("retrieved_memories") or []:
        memory_key = str(memory_id)
        memory_record = memory_index.get(memory_key)
        if memory_record is None:
            raise KeyError(
                f"Missing memory_id={memory_key!r} in {workload_kind} dataset={dataset}"
            )

        text = str(memory_record.get("content", "")).strip()
        if not text:
            continue

        chunk_id = f"{workload_kind}_{memory_key}"
        if chunk_id not in unique_chunks:
            token_ids = list(tokenizer.encode(text, add_special_tokens=False))
            meta = memory_record.get("meta") or {}
            chunk_type = str(
                meta.get("memory_type")
                or meta.get("type")
                or memory_record.get("source_trace", {}).get("speaker")
                or "retrieved_memory"
            )
            unique_chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "text": text,
                "token_ids": token_ids,
                "tokens": len(token_ids),
                "layer": workload_kind,
                "type": chunk_type,
                "conversation_id": dataset,
                "memory_id": memory_key,
            }
        chunk_ids.append(chunk_id)
    return chunk_ids


def resolve_sample_id(
    *,
    workload_kind: str,
    dataset: str,
    qa_index: int,
    qa_trace: dict[str, Any],
) -> str:
    if workload_kind == "memoryos":
        return str(qa_trace.get("sample_id", f"{dataset}:{qa_index}"))
    if workload_kind == "memos":
        return str(qa_trace.get("query_id", f"{dataset}:{qa_index}"))
    return f"{workload_kind}:{dataset}:{qa_index}"


def apply_prefill_order_policy(
    *,
    prefill_first_seen: list[str],
    policy: str,
    shuffle_seed: int,
    workload_kind: str,
) -> list[str]:
    if policy == "first_seen":
        return list(prefill_first_seen)
    if policy == "sorted":
        return sorted(prefill_first_seen)
    if policy == "shuffle":
        shuffled = list(prefill_first_seen)
        rng = random.Random(f"{shuffle_seed}::{workload_kind}::prefill")
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unsupported prefill_order_policy={policy!r}")


def run_native_prefix_workload(
    *,
    args: argparse.Namespace,
    query_records: list[QueryRecord],
    warmup_prompt_ids: list[int],
    kv_cache_memory_bytes: int,
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
        enable_prefix_caching=True,
        enable_blend=False,
        startup_timeout_s=args.startup_timeout_s,
        enforce_eager=args.enforce_eager,
        cuda_visible_devices=args.cuda_visible_devices,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        swap_space=0.0,
        cpu_offload_gb=0.0,
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
                    "Native prefix request failed.\n"
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


def build_blend_gpu_lmcache_env(
    *,
    args: argparse.Namespace,
    chunk_size: int,
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
        "LMCACHE_LOCAL_GPU": "True",
        "LMCACHE_MAX_LOCAL_GPU_SIZE": str(args.max_local_gpu_size),
    }
    extra_config = {}
    if args.blend_connector_impl:
        extra_config["blend_connector_impl"] = str(args.blend_connector_impl)
    if extra_config:
        env["LMCACHE_EXTRA_CONFIG"] = json.dumps(extra_config)
    return env


def run_blend_gpu_workload(
    *,
    args: argparse.Namespace,
    query_records: list[QueryRecord],
    prefill_records: list[PrefillRecord],
    warmup_prompt_ids: list[int],
    chunk_size: int,
    vllm_kv_cache_memory_bytes: int,
) -> BlendChunkResult:
    import requests
    import time

    session = requests.Session()
    session.trust_env = False
    phase_file = Path(
        tempfile.NamedTemporaryFile(
            prefix=f"lmcache_gpu_phase_chunk{chunk_size}_",
            suffix=".jsonl",
            delete=False,
        ).name
    )

    prefill_measurements: list[RequestMeasurement] = []
    query_measurements: list[RequestMeasurement] = []
    lmcache_env = build_blend_gpu_lmcache_env(
        args=args,
        chunk_size=chunk_size,
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
        kv_cache_memory_bytes=vllm_kv_cache_memory_bytes,
        swap_space=0.0,
        cpu_offload_gb=0.0,
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
                    "Blend GPU prefill request failed.\n"
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
                    "Blend GPU online query failed.\n"
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
        backend="gpu",
        chunk_size=chunk_size,
        prefill=prefill_summary,
        online=online_summary,
        online_phase_breakdown=online_phase_breakdown,
        end_to_end_total_wall_s=prefill_summary.wall_s + online_summary.total_wall_s,
        end_to_end_total_ttft_s=prefill_summary.wall_s + online_summary.total_ttft_s,
        first_query_total_including_prefill_s=prefill_summary.wall_s + first_query_ttft_s,
    )


def print_workload_stats(
    *,
    workload_kind: str,
    stats: WorkloadStats,
    prefill_order_policy: str,
) -> None:
    print("=" * 80)
    print("Workload")
    print("=" * 80)
    print(f"workload_kind: {workload_kind}")
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
    print(f"prefill_order_policy: {prefill_order_policy}")
    print(f"shuffle_seed: {stats.shuffle_seed}")
    print(f"system_prompt_tokens: {stats.system_prompt_tokens}")
    print(f"blend_special_str: {stats.blend_special_str!r}")
    print("=" * 80)


def print_result(result: BenchmarkResult) -> None:
    print()
    print("=" * 80)
    print("GPU Cache Budget Benchmark")
    print("=" * 80)
    print(f"model: {result.model}")
    print(f"data_root: {result.data_root}")
    print(f"workload_kind: {result.workload_kind}")
    print(f"datasets: {', '.join(result.datasets)}")
    print(f"prefix_kv_cache_bytes: {result.prefix_kv_cache_bytes}")
    print(f"blend_vllm_kv_cache_bytes: {result.blend_vllm_kv_cache_bytes}")
    print(f"blend_local_gpu_size_gb: {result.blend_local_gpu_size_gb:.2f}")
    print(f"chunk_sizes: {', '.join(str(size) for size in result.chunk_sizes)}")
    print()
    print("[native vLLM prefix caching]")
    print_mode_summary("online query", result.native_prefix, print_header=False)

    for item in result.blend_gpu_results:
        print()
        print(f"[LMCache CacheBlend GPU chunk={item.chunk_size}]")
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

        speedup = 0.0
        if item.online.mean_ttft_s > 0:
            speedup = result.native_prefix.mean_ttft_s / item.online.mean_ttft_s
        print(
            "vs_prefix_online_ttft: "
            f"prefix_mean={format_optional_float(result.native_prefix.mean_ttft_s)}, "
            f"blend_mean={format_optional_float(item.online.mean_ttft_s)}, "
            f"speedup={speedup:.4f}x"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
