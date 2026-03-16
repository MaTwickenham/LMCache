# SPDX-License-Identifier: Apache-2.0
"""Compare direct vLLM, exact-prefix caching, and LMCache blending on CPU RAM.

This benchmark focuses on a synthetic prompt composed of several fixed-size
fragments. It reports:

1. Direct vLLM TTFT on the full prompt.
2. vLLM exact-prefix cache hit latency on the same full prompt.
3. LMCache CacheBlend latency when each fragment is prefetched separately into
   the CPU backend and a final prompt reuses those fragments in a new order.

The LMCache result reports both the final request TTFT and the total latency
including the fragment-prefill preparation phase.
"""

# Standard
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

# Third Party
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.compute.models.utils import VLLMModelTracker


@dataclass
class RequestMeasurement:
    """Request-level latency measurements."""

    ttft_s: float | None
    wall_s: float
    prompt_tokens: int
    num_cached_tokens: int | None
    generated_text: str


@dataclass
class BenchmarkResult:
    """Top-level benchmark result payload."""

    model: str
    num_fragments: int
    fragment_tokens: int
    chunk_size: int
    total_prompt_tokens: int
    fragment_order: list[int]
    baseline_direct: dict[str, Any]
    vllm_prefix: dict[str, Any]
    blend_cpu: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare direct vLLM TTFT, exact-prefix cache hits, and LMCache "
            "CacheBlend CPU-RAM reuse on a synthetic multi-fragment prompt."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/AI/HF_MODELS/Mistral-7B-Instruct-v0.2",
        help="Model path or Hugging Face model identifier.",
    )
    parser.add_argument(
        "--num-fragments",
        type=int,
        default=6,
        help="Number of fragments in the final synthetic prompt.",
    )
    parser.add_argument(
        "--fragment-tokens",
        type=int,
        default=512,
        help="Exact token length for each fragment.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="LMCache chunk size in tokens.",
    )
    parser.add_argument(
        "--query-tokens",
        type=int,
        default=32,
        help="Exact token length for the trailing query segment.",
    )
    parser.add_argument(
        "--warmup-query-tokens",
        type=int,
        default=16,
        help="Exact token length for the fragment-prefill query segment.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Number of output tokens to generate for latency measurement.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model length passed to vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.55,
        help="vLLM gpu_memory_utilization setting.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="vLLM dtype argument.",
    )
    parser.add_argument(
        "--blend-special-str",
        type=str,
        default=" # # ",
        help="Separator inserted between fragments for CacheBlend.",
    )
    parser.add_argument(
        "--max-local-cpu-size",
        type=float,
        default=8.0,
        help="LMCache CPU backend budget in GB.",
    )
    parser.add_argument(
        "--blend-check-layers",
        type=str,
        default="1",
        help="LMCache blending check layers configuration.",
    )
    parser.add_argument(
        "--blend-recompute-ratios",
        type=str,
        default="0.15",
        help="LMCache blending recompute ratios configuration.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the result JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and print the results."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_bundle = build_prompt_bundle(
        tokenizer=tokenizer,
        num_fragments=args.num_fragments,
        fragment_tokens=args.fragment_tokens,
        query_tokens=args.query_tokens,
        warmup_query_tokens=args.warmup_query_tokens,
        blend_special_str=args.blend_special_str,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    baseline_direct, prefix_result = run_vllm_reference(
        model=args.model,
        prompt_ids=prompt_bundle["final_prompt_ids"],
        warmup_prompt_ids=prompt_bundle["engine_warmup_prompt_ids"],
        sampling_params=sampling_params,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )
    release_cuda_memory()

    blend_result = run_blend_cpu_benchmark(
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

    result = BenchmarkResult(
        model=args.model,
        num_fragments=args.num_fragments,
        fragment_tokens=args.fragment_tokens,
        chunk_size=args.chunk_size,
        total_prompt_tokens=len(prompt_bundle["blend_prompt_ids"]),
        fragment_order=prompt_bundle["blend_order"],
        baseline_direct=asdict(baseline_direct),
        vllm_prefix=prefix_result,
        blend_cpu=blend_result,
    )

    print_result(result)
    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(asdict(result), file, indent=2)
            file.write("\n")


def run_vllm_reference(
    model: str,
    prompt_ids: list[int],
    warmup_prompt_ids: list[int],
    sampling_params: SamplingParams,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
) -> tuple[RequestMeasurement, dict[str, Any]]:
    """Run direct vLLM and exact-prefix-cache reference measurements."""

    with build_plain_llm(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        enable_prefix_caching=True,
    ) as llm:
        measure_request(llm, warmup_prompt_ids, sampling_params)
        direct = measure_request(llm, prompt_ids, sampling_params)
        prefix_hit = measure_request(llm, prompt_ids, sampling_params)

    prefix_result = {
        "warmup_kind": "exact_same_prompt",
        "query_ttft_s": prefix_hit.ttft_s,
        "query_wall_s": prefix_hit.wall_s,
        "query_num_cached_tokens": prefix_hit.num_cached_tokens,
        "total_including_prepare_s": direct.wall_s + (prefix_hit.ttft_s or 0.0),
    }
    return direct, prefix_result


def run_blend_cpu_benchmark(
    model: str,
    prefill_prompt_ids: list[list[int]],
    final_prompt_ids: list[int],
    warmup_prompt_ids: list[int],
    sampling_params: SamplingParams,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    chunk_size: int,
    max_local_cpu_size: float,
    blend_special_str: str,
    blend_check_layers: str,
    blend_recompute_ratios: str,
) -> dict[str, Any]:
    """Run the LMCache CacheBlend benchmark on the CPU backend."""

    overrides = {
        "LMCACHE_CHUNK_SIZE": str(chunk_size),
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": str(max_local_cpu_size),
        "LMCACHE_ENABLE_BLENDING": "True",
        "LMCACHE_BLEND_SPECIAL_STR": blend_special_str,
        "LMCACHE_USE_LAYERWISE": "True",
        "LMCACHE_BLEND_CHECK_LAYERS": blend_check_layers,
        "LMCACHE_BLEND_RECOMPUTE_RATIOS": blend_recompute_ratios,
    }

    with temporary_environ(overrides):
        patch_blend_model_registration()
        with build_blend_llm(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        ) as llm:
            measure_request(llm, warmup_prompt_ids, sampling_params)

            prefill_measurements: list[RequestMeasurement] = []
            prefill_start = time.perf_counter()
            for prompt_ids in prefill_prompt_ids:
                prefill_measurements.append(
                    measure_request(llm, prompt_ids, sampling_params)
                )
            prefill_wall_s = time.perf_counter() - prefill_start

            final_request = measure_request(llm, final_prompt_ids, sampling_params)

    return {
        "prefill_requests": len(prefill_prompt_ids),
        "prefill_wall_s": prefill_wall_s,
        "prefill_mean_wall_s": _safe_mean(
            [measurement.wall_s for measurement in prefill_measurements]
        ),
        "prefill_mean_ttft_s": _safe_mean(
            [
                measurement.ttft_s
                for measurement in prefill_measurements
                if measurement.ttft_s is not None
            ]
        ),
        "query_ttft_s": final_request.ttft_s,
        "query_wall_s": final_request.wall_s,
        "query_num_cached_tokens": final_request.num_cached_tokens,
        "total_including_prepare_s": prefill_wall_s + (final_request.ttft_s or 0.0),
    }


def build_prompt_bundle(
    tokenizer: PreTrainedTokenizerBase,
    num_fragments: int,
    fragment_tokens: int,
    query_tokens: int,
    warmup_query_tokens: int,
    blend_special_str: str,
) -> dict[str, Any]:
    """Build synthetic prompt token IDs for all benchmark modes."""

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("The tokenizer must expose a bos_token_id.")

    system_ids = [bos_id] + tokenizer.encode(
        "You are a retrieval QA assistant. Use the provided fragments only.",
        add_special_tokens=False,
    )
    engine_warmup_prompt_ids = [bos_id] + build_exact_token_ids(
        tokenizer,
        seed_text="engine warmup prompt",
        target_tokens=64,
    )
    blend_special_ids = tokenizer.encode(
        blend_special_str, add_special_tokens=False
    )
    query_ids = build_exact_token_ids(
        tokenizer,
        seed_text="answer the question using the fragment ids only",
        target_tokens=query_tokens,
    )
    warmup_query_ids = build_exact_token_ids(
        tokenizer,
        seed_text="warmup query",
        target_tokens=warmup_query_tokens,
    )

    fragments = [
        build_exact_token_ids(
            tokenizer,
            seed_text=(
                f"fragment {fragment_index} evidence token group "
                f"{fragment_index} synthetic benchmark "
            ),
            target_tokens=fragment_tokens,
        )
        for fragment_index in range(num_fragments)
    ]

    blend_order = list(range(num_fragments))
    blend_order = blend_order[1::2] + blend_order[0::2]

    blend_prompt_ids = build_fragment_prompt(
        system_ids=system_ids,
        fragment_ids=[fragments[index] for index in blend_order],
        blend_special_ids=blend_special_ids,
        query_ids=query_ids,
    )
    prefill_prompt_ids = [
        build_fragment_prompt(
            system_ids=system_ids,
            fragment_ids=[fragment],
            blend_special_ids=blend_special_ids,
            query_ids=warmup_query_ids,
        )
        for fragment in fragments
    ]

    return {
        "engine_warmup_prompt_ids": engine_warmup_prompt_ids,
        "final_prompt_ids": blend_prompt_ids,
        "blend_prompt_ids": blend_prompt_ids,
        "prefill_prompt_ids": prefill_prompt_ids,
        "blend_order": blend_order,
    }


def build_fragment_prompt(
    system_ids: list[int],
    fragment_ids: list[list[int]],
    blend_special_ids: list[int],
    query_ids: list[int],
) -> list[int]:
    """Build a prompt from a system segment, fragments, and a trailing query."""

    prompt_ids = list(system_ids)
    for fragment in fragment_ids:
        prompt_ids.extend(blend_special_ids)
        prompt_ids.extend(fragment)
    prompt_ids.extend(blend_special_ids)
    prompt_ids.extend(query_ids)
    return prompt_ids


def build_exact_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    seed_text: str,
    target_tokens: int,
) -> list[int]:
    """Build a deterministic token sequence with an exact token count."""

    unit_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not unit_ids:
        raise ValueError(f"Failed to tokenize seed_text={seed_text!r}.")

    repeat_count = math.ceil(target_tokens / len(unit_ids))
    return (unit_ids * repeat_count)[:target_tokens]


def measure_request(
    llm: LLM,
    prompt_ids: list[int],
    sampling_params: SamplingParams,
) -> RequestMeasurement:
    """Generate one request and return TTFT and wall-clock latency."""

    start_time = time.perf_counter()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt_ids},
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    wall_s = time.perf_counter() - start_time
    output = outputs[0]
    metrics = output.metrics

    ttft_s = None
    if metrics is not None and getattr(metrics, "first_token_time", None) is not None:
        ttft_s = metrics.first_token_time - metrics.arrival_time

    return RequestMeasurement(
        ttft_s=ttft_s,
        wall_s=wall_s,
        prompt_tokens=len(prompt_ids),
        num_cached_tokens=output.num_cached_tokens,
        generated_text=output.outputs[0].text,
    )


def patch_blend_model_registration() -> None:
    """Register the loaded vLLM model for CacheBlend in Python API benchmarks.

    LMCache's blend path expects the model object to be retrievable through
    ``VLLMModelTracker``. In the current local environment, the Python API path
    does not register that model automatically, so the benchmark patches
    ``GPUModelRunner.load_model`` to do the registration right after the model
    is loaded.
    """

    # Third Party
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner.load_model, "_lmcache_blend_patched", False):
        return

    original_load_model = GPUModelRunner.load_model

    def patched_load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load_model(self, *args, **kwargs)
        VLLMModelTracker.register_model(ENGINE_NAME, self.get_model())
        return result

    patched_load_model._lmcache_blend_patched = True  # type: ignore[attr-defined]
    GPUModelRunner.load_model = patched_load_model


@contextmanager
def build_plain_llm(
    model: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    enable_prefix_caching: bool,
) -> Iterator[LLM]:
    """Build a plain vLLM engine without LMCache."""

    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        enforce_eager=True,
        enable_prefix_caching=enable_prefix_caching,
    )
    try:
        yield llm
    finally:
        del llm


@contextmanager
def build_blend_llm(
    model: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
) -> Iterator[LLM]:
    """Build a vLLM engine with LMCache CacheBlend enabled."""

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
        enable_prefix_caching=False,
        kv_transfer_config=kv_transfer_config,
    )
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
        del llm


@contextmanager
def temporary_environ(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily override environment variables."""

    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def release_cuda_memory() -> None:
    """Run best-effort Python and CUDA cleanup between benchmark phases."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_result(result: BenchmarkResult) -> None:
    """Print a readable result summary."""

    baseline = result.baseline_direct
    prefix = result.vllm_prefix
    blend = result.blend_cpu

    print("=" * 80)
    print("Synthetic Fragment Benchmark")
    print("=" * 80)
    print(f"model: {result.model}")
    print(
        f"prompt: {result.num_fragments} fragments x {result.fragment_tokens} tokens, "
        f"chunk_size={result.chunk_size}, total_prompt_tokens={result.total_prompt_tokens}"
    )
    print(f"blend_fragment_order: {result.fragment_order}")
    print()
    print("[plain vLLM]")
    print(f"direct_ttft_s: {format_optional_float(baseline['ttft_s'])}")
    print(f"direct_wall_s: {baseline['wall_s']:.4f}")
    print(
        "exact_prefix_hit_ttft_s: "
        f"{format_optional_float(prefix['query_ttft_s'])}"
    )
    print(f"exact_prefix_hit_wall_s: {prefix['query_wall_s']:.4f}")
    print(
        "exact_prefix_total_including_prepare_s: "
        f"{prefix['total_including_prepare_s']:.4f}"
    )
    print()
    print("[LMCache CacheBlend on CPU RAM]")
    print(f"prefill_requests: {blend['prefill_requests']}")
    print(f"prefill_wall_s: {blend['prefill_wall_s']:.4f}")
    print(f"prefill_mean_wall_s: {blend['prefill_mean_wall_s']:.4f}")
    print(
        "prefill_mean_ttft_s: "
        f"{format_optional_float(blend['prefill_mean_ttft_s'])}"
    )
    print(f"query_ttft_s: {format_optional_float(blend['query_ttft_s'])}")
    print(f"query_wall_s: {blend['query_wall_s']:.4f}")
    print(
        "total_including_prepare_s: "
        f"{blend['total_including_prepare_s']:.4f}"
    )
    print("=" * 80)


def format_optional_float(value: float | None) -> str:
    """Format a float that may be missing."""

    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _safe_mean(values: list[float]) -> float:
    """Return the arithmetic mean or 0.0 for an empty list."""

    if not values:
        return 0.0
    return sum(values) / len(values)


patch_blend_model_registration()


if __name__ == "__main__":
    main()
