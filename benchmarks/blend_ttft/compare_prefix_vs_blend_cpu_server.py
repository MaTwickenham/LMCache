# SPDX-License-Identifier: Apache-2.0
"""Compare vLLM prefix reuse and LMCache CacheBlend TTFT over streaming APIs.

This benchmark uses the OpenAI-compatible streaming server path so TTFT is
measured as wall-clock time from request submission to the first non-empty
streamed token chunk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class RequestMeasurement:
    """Request-level latency measurements."""

    request_id: str | None
    ttft_s: float | None
    wall_s: float
    prompt_tokens: int
    cached_tokens: int | None
    generated_text: str


@dataclass
class ChunkBlendResult:
    """CacheBlend CPU benchmark output for one chunk size."""

    chunk_size: int
    prefill_requests: int
    prefill_wall_s: float
    prefill_mean_wall_s: float
    prefill_mean_ttft_s: float
    query_ttft_s: float | None
    query_wall_s: float
    query_cached_tokens: int | None
    total_including_prepare_s: float


@dataclass
class BenchmarkResult:
    """Top-level result payload."""

    model: str
    num_fragments: int
    fragment_tokens: int
    total_prompt_tokens: int
    fragment_order: list[int]
    baseline_no_prefix: dict[str, Any]
    baseline_prefix_miss: dict[str, Any]
    vllm_prefix: dict[str, Any]
    blend_cpu_results: list[dict[str, Any]]


class ServerProcess:
    """Utility wrapper around a background vLLM server process."""

    def __init__(self, process: subprocess.Popen[str], log_tail: deque[str]):
        self.process = process
        self.log_tail = log_tail

    def tail_text(self) -> str:
        return "".join(self.log_tail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct vLLM, exact-prefix reuse, and LMCache CacheBlend "
            "CPU-RAM reuse with true streaming TTFT measurements."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/AI/HF_MODELS/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument("--num-fragments", type=int, default=6)
    parser.add_argument("--fragment-tokens", type=int, default=512)
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="512,256,128",
        help="Comma-separated LMCache chunk sizes.",
    )
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--warmup-query-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--blend-special-str", type=str, default="# #")
    parser.add_argument("--max-local-cpu-size", type=float, default=8.0)
    parser.add_argument("--blend-check-layers", type=str, default="1")
    parser.add_argument("--blend-recompute-ratios", type=str, default="0.15")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--startup-timeout-s", type=float, default=240.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_sizes = [int(value) for value in args.chunk_sizes.split(",") if value]
    if not chunk_sizes:
        raise ValueError("chunk_sizes cannot be empty.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_bundle = build_prompt_bundle(
        tokenizer=tokenizer,
        num_fragments=args.num_fragments,
        fragment_tokens=args.fragment_tokens,
        query_tokens=args.query_tokens,
        warmup_query_tokens=args.warmup_query_tokens,
        blend_special_str=args.blend_special_str,
    )

    session = requests.Session()
    session.trust_env = False

    baseline_no_prefix = run_vllm_no_prefix_reference(
        session=session,
        args=args,
        final_prompt_ids=prompt_bundle["final_prompt_ids"],
        warmup_prompt_ids=prompt_bundle["engine_warmup_prompt_ids"],
    )

    baseline_prefix_miss, prefix_result = run_vllm_prefix_reference(
        session=session,
        args=args,
        final_prompt_ids=prompt_bundle["final_prompt_ids"],
        warmup_prompt_ids=prompt_bundle["engine_warmup_prompt_ids"],
    )

    blend_cpu_results: list[ChunkBlendResult] = []
    for chunk_size in chunk_sizes:
        blend_cpu_results.append(
            run_blend_cpu_benchmark(
                session=session,
                args=args,
                chunk_size=chunk_size,
                prefill_prompt_ids=prompt_bundle["prefill_prompt_ids"],
                final_prompt_ids=prompt_bundle["blend_prompt_ids"],
                warmup_prompt_ids=prompt_bundle["engine_warmup_prompt_ids"],
            )
        )

    result = BenchmarkResult(
        model=args.model,
        num_fragments=args.num_fragments,
        fragment_tokens=args.fragment_tokens,
        total_prompt_tokens=len(prompt_bundle["blend_prompt_ids"]),
        fragment_order=prompt_bundle["blend_order"],
        baseline_no_prefix=asdict(baseline_no_prefix),
        baseline_prefix_miss=asdict(baseline_prefix_miss),
        vllm_prefix=prefix_result,
        blend_cpu_results=[asdict(item) for item in blend_cpu_results],
    )

    print_result(result)
    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(asdict(result), file, indent=2)
            file.write("\n")


def run_vllm_no_prefix_reference(
    session: requests.Session,
    args: argparse.Namespace,
    final_prompt_ids: list[int],
    warmup_prompt_ids: list[int],
) -> RequestMeasurement:
    """Run vLLM without prefix caching for a pure compute baseline."""

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
    ):
        measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=warmup_prompt_ids,
            max_tokens=args.max_tokens,
        )
        return measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=final_prompt_ids,
            max_tokens=args.max_tokens,
        )


def run_vllm_prefix_reference(
    session: requests.Session,
    args: argparse.Namespace,
    final_prompt_ids: list[int],
    warmup_prompt_ids: list[int],
) -> tuple[RequestMeasurement, dict[str, Any]]:
    """Run direct vLLM and exact-prefix cache-hit measurements."""

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
    ):
        measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=warmup_prompt_ids,
            max_tokens=args.max_tokens,
        )
        direct = measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=final_prompt_ids,
            max_tokens=args.max_tokens,
        )
        prefix_hit = measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=final_prompt_ids,
            max_tokens=args.max_tokens,
        )

    prefix_result = {
        "warmup_kind": "exact_same_prompt",
        "query_ttft_s": prefix_hit.ttft_s,
        "query_wall_s": prefix_hit.wall_s,
        "query_cached_tokens": prefix_hit.cached_tokens,
        "total_including_prepare_s": direct.wall_s + (prefix_hit.ttft_s or 0.0),
    }
    return direct, prefix_result


def run_blend_cpu_benchmark(
    session: requests.Session,
    args: argparse.Namespace,
    chunk_size: int,
    prefill_prompt_ids: list[list[int]],
    final_prompt_ids: list[int],
    warmup_prompt_ids: list[int],
) -> ChunkBlendResult:
    """Run the LMCache CacheBlend benchmark on the CPU backend."""

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
        lmcache_env={
            "LMCACHE_CHUNK_SIZE": str(chunk_size),
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(args.max_local_cpu_size),
            "LMCACHE_ENABLE_BLENDING": "True",
            "LMCACHE_BLEND_SPECIAL_STR": args.blend_special_str,
            "LMCACHE_SAVE_UNFULL_CHUNK": "True",
            "LMCACHE_USE_LAYERWISE": "True",
            "LMCACHE_BLEND_CHECK_LAYERS": args.blend_check_layers,
            "LMCACHE_BLEND_RECOMPUTE_RATIOS": args.blend_recompute_ratios,
        },
    ):
        measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=warmup_prompt_ids,
            max_tokens=args.max_tokens,
        )

        prefill_measurements: list[RequestMeasurement] = []
        prefill_start = time.perf_counter()
        for prompt_ids in prefill_prompt_ids:
            prefill_measurements.append(
                measure_streaming_request(
                    session=session,
                    port=args.port,
                    model=args.model,
                    prompt_ids=prompt_ids,
                    max_tokens=args.max_tokens,
                    kv_transfer_params={
                        "lmcache.request_kind": "fragment_prefill",
                    },
                )
            )
        prefill_wall_s = time.perf_counter() - prefill_start

        final_request = measure_streaming_request(
            session=session,
            port=args.port,
            model=args.model,
            prompt_ids=final_prompt_ids,
            max_tokens=args.max_tokens,
            kv_transfer_params={
                "lmcache.request_kind": "online_blended_query",
            },
        )

    return ChunkBlendResult(
        chunk_size=chunk_size,
        prefill_requests=len(prefill_prompt_ids),
        prefill_wall_s=prefill_wall_s,
        prefill_mean_wall_s=_safe_mean(
            [measurement.wall_s for measurement in prefill_measurements]
        ),
        prefill_mean_ttft_s=_safe_mean(
            [
                measurement.ttft_s
                for measurement in prefill_measurements
                if measurement.ttft_s is not None
            ]
        ),
        query_ttft_s=final_request.ttft_s,
        query_wall_s=final_request.wall_s,
        query_cached_tokens=final_request.cached_tokens,
        total_including_prepare_s=prefill_wall_s + (final_request.ttft_s or 0.0),
    )


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

    system_ids = [bos_id] + build_exact_token_ids(
        tokenizer,
        seed_text="You are a retrieval QA assistant. Use the provided fragments only.",
        target_tokens=48,
    )
    engine_warmup_prompt_ids = [bos_id] + build_exact_token_ids(
        tokenizer,
        seed_text="engine warmup prompt",
        target_tokens=64,
    )
    blend_special_ids = tokenizer.encode(blend_special_str, add_special_tokens=False)
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
    unit_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not unit_ids:
        raise ValueError(f"Failed to tokenize seed_text={seed_text!r}.")

    repeat_count = math.ceil(target_tokens / len(unit_ids))
    return (unit_ids * repeat_count)[:target_tokens]


@contextmanager
def launch_server(
    model: str,
    port: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    enable_prefix_caching: bool,
    enable_blend: bool,
    startup_timeout_s: float,
    enforce_eager: bool,
    cuda_visible_devices: str | None,
    lmcache_env: dict[str, str] | None = None,
) -> Iterator[ServerProcess]:
    """Launch a vLLM server and tear it down automatically."""

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for proxy_key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        env.pop(proxy_key, None)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if lmcache_env is not None:
        env.update(lmcache_env)

    cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        dtype,
        "--disable-log-stats",
        "--enable-prompt-tokens-details",
    ]
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")
    if enforce_eager:
        cmd.append("--enforce-eager")
    if enable_blend:
        cmd.extend(
            [
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "LMCacheConnectorV1",
                        "kv_role": "kv_both",
                    }
                ),
            ]
        )

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    log_tail: deque[str] = deque(maxlen=400)

    def log_reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            log_tail.append(line)

    reader_thread = threading.Thread(target=log_reader, daemon=True)
    reader_thread.start()
    server = ServerProcess(process=process, log_tail=log_tail)

    try:
        wait_for_server_ready(server, port=port, timeout_s=startup_timeout_s)
        yield server
    finally:
        terminate_server(server)


def wait_for_server_ready(server: ServerProcess, port: int, timeout_s: float) -> None:
    """Wait until the server starts responding to /health."""

    session = requests.Session()
    session.trust_env = False
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(
                "vLLM server exited before becoming healthy.\n"
                f"exit_code={server.process.returncode}\n"
                f"tail:\n{server.tail_text()}"
            )
        try:
            response = session.get(health_url, timeout=2)
            if response.status_code == 200:
                return
            last_error = f"health={response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)

    raise TimeoutError(
        "Timed out waiting for the vLLM server to become healthy.\n"
        f"last_error={last_error}\n"
        f"tail:\n{server.tail_text()}"
    )


def terminate_server(server: ServerProcess) -> None:
    """Best-effort shutdown for the background server process."""

    if server.process.poll() is not None:
        return

    try:
        os.killpg(server.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        server.process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(server.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        server.process.wait(timeout=10)


def measure_streaming_request(
    session: requests.Session,
    port: int,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    kv_transfer_params: dict[str, Any] | None = None,
) -> RequestMeasurement:
    """Measure TTFT and total wall time using the streaming completions API."""

    payload = {
        "model": model,
        "prompt": prompt_ids,
        "add_special_tokens": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if kv_transfer_params is not None:
        payload["kv_transfer_params"] = kv_transfer_params

    url = f"http://127.0.0.1:{port}/v1/completions"
    start_time = time.perf_counter()
    first_token_time: float | None = None
    generated_parts: list[str] = []
    cached_tokens: int | None = None
    request_id: str | None = None

    with session.post(url, json=payload, stream=True, timeout=180) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"Request failed with status {response.status_code}: {response.text}"
            )

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data: "):
                continue

            data = raw_line[6:]
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            if request_id is None:
                request_id = chunk.get("id")
            choices = chunk.get("choices") or []
            if choices:
                text = choices[0].get("text", "")
                if text and first_token_time is None:
                    first_token_time = time.perf_counter()
                if text:
                    generated_parts.append(text)

            usage = chunk.get("usage")
            if usage is not None:
                prompt_details = usage.get("prompt_tokens_details") or {}
                cached_tokens = prompt_details.get("cached_tokens")

    wall_s = time.perf_counter() - start_time
    ttft_s = None
    if first_token_time is not None:
        ttft_s = first_token_time - start_time

    return RequestMeasurement(
        request_id=request_id,
        ttft_s=ttft_s,
        wall_s=wall_s,
        prompt_tokens=len(prompt_ids),
        cached_tokens=cached_tokens,
        generated_text="".join(generated_parts),
    )


def print_result(result: BenchmarkResult) -> None:
    baseline_no_prefix = result.baseline_no_prefix
    baseline_prefix_miss = result.baseline_prefix_miss
    prefix = result.vllm_prefix

    print("=" * 80)
    print("Synthetic Fragment Streaming TTFT Benchmark")
    print("=" * 80)
    print(f"model: {result.model}")
    print(
        f"prompt: {result.num_fragments} fragments x {result.fragment_tokens} tokens, "
        f"total_prompt_tokens={result.total_prompt_tokens}"
    )
    print(f"blend_fragment_order: {result.fragment_order}")
    print()
    print("[plain vLLM]")
    print(
        "no_prefix_compute_ttft_s: "
        f"{format_optional_float(baseline_no_prefix['ttft_s'])}"
    )
    print(f"no_prefix_compute_wall_s: {baseline_no_prefix['wall_s']:.4f}")
    print(f"no_prefix_compute_cached_tokens: {baseline_no_prefix['cached_tokens']}")
    print(
        "prefix_enabled_first_request_ttft_s: "
        f"{format_optional_float(baseline_prefix_miss['ttft_s'])}"
    )
    print(
        "prefix_enabled_first_request_wall_s: "
        f"{baseline_prefix_miss['wall_s']:.4f}"
    )
    print(
        "prefix_enabled_first_request_cached_tokens: "
        f"{baseline_prefix_miss['cached_tokens']}"
    )
    print(f"exact_prefix_hit_ttft_s: {format_optional_float(prefix['query_ttft_s'])}")
    print(f"exact_prefix_hit_wall_s: {prefix['query_wall_s']:.4f}")
    print(f"exact_prefix_hit_cached_tokens: {prefix['query_cached_tokens']}")
    print(
        "exact_prefix_total_including_prepare_s: "
        f"{prefix['total_including_prepare_s']:.4f}"
    )
    print()
    print("[LMCache CacheBlend on CPU RAM]")
    for item in result.blend_cpu_results:
        print(
            f"chunk={item['chunk_size']}: "
            f"query_ttft_s={format_optional_float(item['query_ttft_s'])}, "
            f"query_wall_s={item['query_wall_s']:.4f}, "
            f"query_cached_tokens={item['query_cached_tokens']}, "
            f"prefill_wall_s={item['prefill_wall_s']:.4f}, "
            f"total_including_prepare_s={item['total_including_prepare_s']:.4f}"
        )
    print("=" * 80)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
