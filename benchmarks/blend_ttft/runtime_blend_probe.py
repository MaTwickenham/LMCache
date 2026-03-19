from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from compare_prefix_vs_blend_gpu_workload_server import (
    append_token_segment,
    apply_prefill_order_policy,
    build_prefill_plan,
    build_prefill_records,
    build_prompt_token_ids,
    build_system_prompt_ids,
    build_workload_stats,
    enrich_id_backed_hints,
    enrich_memoryos_hints,
    extract_id_backed_chunks,
    extract_memoryos_chunks,
    load_memory_index,
    load_trace,
    reorder_chunk_ids,
    resolve_datasets,
    resolve_fragment_order_policy,
    resolve_sample_id,
)
from compare_prefix_vs_blend_memoryos_server import QueryRecord


def build_workload_for_probe(args: Any) -> dict[str, Any]:
    data_root = Path(args.data_root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    datasets = resolve_datasets(args.workload_kind, "")
    fragment_order_policy = resolve_fragment_order_policy(
        workload_kind=args.workload_kind,
        raw_policy=args.fragment_order_policy,
    )

    unique_chunks: dict[str, dict[str, Any]] = {}
    query_records: list[QueryRecord] = []
    query_chunk_ids: list[list[str]] = []
    prefill_first_seen: list[str] = []
    prefill_seen: set[str] = set()

    system_prompt_ids = build_system_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
    )
    blend_special_ids = list(
        tokenizer.encode(args.blend_special_str, add_special_tokens=False)
    )
    warmup_prompt_ids = list(system_prompt_ids)
    warmup_query_ids = list(
        tokenizer.encode(args.warmup_query_text, add_special_tokens=False)
    )
    append_token_segment(
        warmup_prompt_ids,
        warmup_query_ids,
        skip_bos=bool(warmup_prompt_ids),
    )

    for dataset in datasets:
        traces = load_trace(
            data_root=data_root,
            workload_kind=args.workload_kind,
            dataset=dataset,
        )
        if args.max_qa_per_dataset > 0:
            traces = traces[: args.max_qa_per_dataset]

        memory_index = None
        if args.workload_kind in ("amem", "memos"):
            memory_index = load_memory_index(
                data_root=data_root,
                workload_kind=args.workload_kind,
                dataset=dataset,
            )

        for qa_index, qa_trace in enumerate(traces):
            if args.workload_kind == "memoryos":
                chunk_ids = extract_memoryos_chunks(
                    qa_trace=qa_trace,
                    tokenizer=tokenizer,
                    unique_chunks=unique_chunks,
                )
            else:
                assert memory_index is not None
                chunk_ids = extract_id_backed_chunks(
                    workload_kind=args.workload_kind,
                    dataset=dataset,
                    qa_trace=qa_trace,
                    tokenizer=tokenizer,
                    unique_chunks=unique_chunks,
                    memory_index=memory_index,
                )

            ordered_chunk_ids = reorder_chunk_ids(
                chunk_ids=chunk_ids,
                unique_chunks=unique_chunks,
                dataset_tag=f"{args.workload_kind}:{dataset}",
                qa_index=qa_index,
                policy=fragment_order_policy,
                shuffle_seed=args.shuffle_seed,
            )
            query_chunk_ids.append(list(ordered_chunk_ids))

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
                prompt_layout=args.prompt_layout,
            )
            memory_prefix_tokens = sum(len(ids) for ids in fragment_token_ids)
            if len(prompt_ids) > args.max_model_len:
                raise ValueError(
                    f"Prompt too long for {args.workload_kind} dataset={dataset} "
                    f"qa_index={qa_index}: {len(prompt_ids)} > {args.max_model_len}"
                )
            query_records.append(
                QueryRecord(
                    dataset=dataset,
                    qa_index=qa_index,
                    sample_id=resolve_sample_id(
                        workload_kind=args.workload_kind,
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

    if args.workload_kind == "memoryos":
        enrich_memoryos_hints(unique_chunks)
    else:
        enrich_id_backed_hints(
            workload_kind=args.workload_kind,
            unique_chunks=unique_chunks,
            data_root=data_root,
            datasets=datasets,
        )

    prefill_order = apply_prefill_order_policy(
        prefill_first_seen=prefill_first_seen,
        policy=args.prefill_order_policy,
        shuffle_seed=args.shuffle_seed,
        workload_kind=args.workload_kind,
    )
    prefill_plan = build_prefill_plan(
        unique_chunks=unique_chunks,
        prefill_order=prefill_order,
        prefill_placement_policy=args.prefill_placement_policy,
        utility_cost_model=args.utility_cost_model,
        utility_tail_lambda=args.utility_tail_lambda,
        utility_gpu_penalty_ms=args.utility_gpu_penalty_ms,
        max_local_gpu_size=args.max_local_gpu_size,
        max_local_cpu_size=args.max_local_cpu_size,
    )
    prefill_records = build_prefill_records(
        tokenizer=tokenizer,
        unique_chunks=unique_chunks,
        prefill_order=prefill_order,
        system_prompt_ids=system_prompt_ids,
        blend_special_ids=blend_special_ids,
        prefill_query_text=args.prefill_query_text,
        prompt_layout=args.prompt_layout,
        max_model_len=args.max_model_len,
        prefill_plan=prefill_plan,
    )
    all_prefill_records = build_prefill_records(
        tokenizer=tokenizer,
        unique_chunks=unique_chunks,
        prefill_order=prefill_order,
        system_prompt_ids=system_prompt_ids,
        blend_special_ids=blend_special_ids,
        prefill_query_text=args.prefill_query_text,
        prompt_layout=args.prompt_layout,
        max_model_len=args.max_model_len,
        prefill_plan=None,
    )
    stats = build_workload_stats(
        datasets=datasets,
        query_records=query_records,
        unique_chunks=unique_chunks,
        prompt_layout=args.prompt_layout,
        fragment_order_policy=fragment_order_policy,
        shuffle_seed=args.shuffle_seed,
        system_prompt_tokens=len(system_prompt_ids),
        blend_special_str=args.blend_special_str,
    )
    return {
        "datasets": datasets,
        "fragment_order_policy": fragment_order_policy,
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


def build_phase_index(
    phase_records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    phase_index: dict[str, dict[str, dict[str, Any]]] = {}
    for record in phase_records:
        req_id = normalize_request_id(str(record.get("req_id", "")))
        phase = str(record.get("phase", ""))
        phase_index.setdefault(req_id, {})[phase] = record
    return phase_index


def normalize_request_id(req_id: str) -> str:
    return re.sub(r"-\d+$", "", req_id)
