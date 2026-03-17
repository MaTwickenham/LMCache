#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the synthetic prefix-vs-blend benchmark with safer vLLM limits.

This wrapper reuses the existing benchmark logic but overrides the ``LLM``
construction so that the Python API benchmark can run on a single 4090 without
the default profiling pass over-allocating memory.
"""

# Standard
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator


def load_benchmark_module():
    benchmark_path = (
        Path(__file__).resolve().parent / "compare_prefix_vs_blend_cpu.py"
    )
    spec = importlib.util.spec_from_file_location(
        "blend_ttft_compare_prefix_vs_blend_cpu",
        benchmark_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load benchmark module from {benchmark_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run compare_prefix_vs_blend_cpu.py with safer vLLM scheduler "
            "limits and optional legacy/fast connector selection."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/AI/HF_MODELS/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument("--num-fragments", type=int, default=4)
    parser.add_argument("--fragment-tokens", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--query-tokens", type=int, default=16)
    parser.add_argument("--warmup-query-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--blend-special-str", type=str, default=" # # ")
    parser.add_argument("--max-local-cpu-size", type=float, default=8.0)
    parser.add_argument("--blend-check-layers", type=str, default="1")
    parser.add_argument("--blend-recompute-ratios", type=str, default="0.15")
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--skip-vllm-reference",
        action="store_true",
        help="Skip the plain vLLM direct/prefix baseline phase.",
    )
    parser.add_argument(
        "--skip-blend",
        action="store_true",
        help="Skip all LMCache blend runs.",
    )
    parser.add_argument(
        "--connector-impls",
        type=str,
        default="legacy,fast",
        help="Comma-separated list from {legacy,fast}.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    mod = load_benchmark_module()
    args = parse_args()

    llm_kwargs = {
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enable_chunked_prefill": True,
        "disable_log_stats": True,
    }

    @contextmanager
    def custom_plain(
        model: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        dtype: str,
        enable_prefix_caching: bool,
    ) -> Iterator[object]:
        llm = mod.LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=True,
            enable_prefix_caching=enable_prefix_caching,
            **llm_kwargs,
        )
        try:
            yield llm
        finally:
            del llm

    @contextmanager
    def custom_blend(
        model: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        dtype: str,
    ) -> Iterator[object]:
        kv_transfer_config = mod.KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
        )
        llm = mod.LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=True,
            enable_prefix_caching=False,
            kv_transfer_config=kv_transfer_config,
            **llm_kwargs,
        )
        try:
            yield llm
        finally:
            mod.LMCacheEngineBuilder.destroy(mod.ENGINE_NAME)
            del llm

    mod.build_plain_llm = custom_plain
    mod.build_blend_llm = custom_blend

    tokenizer = mod.AutoTokenizer.from_pretrained(args.model)
    prompt_bundle = mod.build_prompt_bundle(
        tokenizer=tokenizer,
        num_fragments=args.num_fragments,
        fragment_tokens=args.fragment_tokens,
        query_tokens=args.query_tokens,
        warmup_query_tokens=args.warmup_query_tokens,
        blend_special_str=args.blend_special_str,
    )
    sampling_params = mod.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    result = {
        "model": args.model,
        "num_fragments": args.num_fragments,
        "fragment_tokens": args.fragment_tokens,
        "chunk_size": args.chunk_size,
        "total_prompt_tokens": len(prompt_bundle["blend_prompt_ids"]),
        "fragment_order": prompt_bundle["blend_order"],
    }

    if not args.skip_vllm_reference:
        baseline_direct, prefix_result = mod.run_vllm_reference(
            model=args.model,
            prompt_ids=prompt_bundle["final_prompt_ids"],
            warmup_prompt_ids=prompt_bundle["engine_warmup_prompt_ids"],
            sampling_params=sampling_params,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )
        mod.release_cuda_memory()
        result["baseline_direct"] = asdict(baseline_direct)
        result["vllm_prefix"] = prefix_result

    if not args.skip_blend:
        connector_impls = [
            item.strip() for item in args.connector_impls.split(",") if item.strip()
        ]
        for connector_impl in connector_impls:
            mod.release_cuda_memory()
            with mod.temporary_environ(
                {
                    "LMCACHE_EXTRA_CONFIG": json.dumps(
                        {"blend_connector_impl": connector_impl}
                    )
                }
            ):
                blend_result = mod.run_blend_cpu_benchmark(
                    model=args.model,
                    prefill_prompt_ids=prompt_bundle["prefill_prompt_ids"],
                    final_prompt_ids=prompt_bundle["blend_prompt_ids"],
                    warmup_prompt_ids=prompt_bundle["engine_warmup_prompt_ids"],
                    sampling_params=sampling_params,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    dtype=args.dtype,
                    chunk_size=args.chunk_size,
                    max_local_cpu_size=args.max_local_cpu_size,
                    blend_special_str=args.blend_special_str,
                    blend_check_layers=args.blend_check_layers,
                    blend_recompute_ratios=args.blend_recompute_ratios,
                )
            result[f"blend_cpu_{connector_impl}"] = blend_result
            mod.release_cuda_memory()

    payload = json.dumps(result, indent=2)
    print(payload)

    if args.output_json is not None:
        Path(args.output_json).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
