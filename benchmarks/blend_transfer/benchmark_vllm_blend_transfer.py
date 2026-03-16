# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import random
import statistics
import time

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import VLLMBufferLayerwiseGPUConnector
from lmcache.v1.gpu_connector.utils import discover_gpu_kv_format
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    MemoryFormat,
    MemoryObj,
    TensorMemoryAllocator,
)

if not torch.cuda.is_available():
    raise RuntimeError("This benchmark requires CUDA.")

# First Party
import lmcache.c_ops as lmc_ops


@dataclass
class BenchmarkConfig:
    """Configuration for the blend transfer microbenchmark."""

    num_layers: int
    num_heads: int
    head_size: int
    num_blocks: int
    block_size: int
    num_fragments: int
    fragment_tokens: int
    gap_tokens: int
    dtype: torch.dtype
    warmup_rounds: int
    rounds: int
    seed: int

    @property
    def hidden_dim(self) -> int:
        """Return the flattened hidden dimension per token."""

        return self.num_heads * self.head_size

    @property
    def total_span_tokens(self) -> int:
        """Return the span size from the first reused token to the last one."""

        return self.num_fragments * self.fragment_tokens + (
            self.num_fragments - 1
        ) * self.gap_tokens


def parse_args() -> BenchmarkConfig:
    """Parse CLI arguments."""

    parser = ArgumentParser(
        description=(
            "Benchmark LMCache blend transfer hot path for "
            "CPU-pinned/GPU sources into GPU buffer and paged KV."
        )
    )
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-fragments", type=int, default=4)
    parser.add_argument("--fragment-tokens", type=int, default=256)
    parser.add_argument("--gap-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_size=args.head_size,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        num_fragments=args.num_fragments,
        fragment_tokens=args.fragment_tokens,
        gap_tokens=args.gap_tokens,
        dtype=_parse_dtype(args.dtype),
        warmup_rounds=args.warmup_rounds,
        rounds=args.rounds,
        seed=args.seed,
    )
    _validate_config(cfg)
    return cfg


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Map CLI dtype names to torch dtypes."""

    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _validate_config(cfg: BenchmarkConfig) -> None:
    """Validate the benchmark configuration."""

    if cfg.num_fragments <= 0:
        raise ValueError("num_fragments must be positive.")
    if cfg.fragment_tokens <= 0:
        raise ValueError("fragment_tokens must be positive.")
    if cfg.gap_tokens < 0:
        raise ValueError("gap_tokens cannot be negative.")
    page_capacity = cfg.num_blocks * cfg.block_size
    if cfg.total_span_tokens > page_capacity:
        raise ValueError(
            "total span exceeds paged KV capacity. "
            f"span={cfg.total_span_tokens}, capacity={page_capacity}."
        )


def main() -> None:
    """Entry point for the benchmark."""

    cfg = parse_args()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    starts, ends = build_fragment_spans(
        cfg.num_fragments,
        cfg.fragment_tokens,
        cfg.gap_tokens,
    )
    gap_positions = build_gap_positions(starts, ends, device="cuda")
    slot_mapping = build_slot_mapping(cfg.total_span_tokens, cfg.num_blocks, cfg.block_size)
    paged_kv = generate_vllm_paged_kv(cfg)
    gpu_kv_format = discover_gpu_kv_format(paged_kv, _vllm_engine_type())

    connector = VLLMBufferLayerwiseGPUConnector(
        cfg.hidden_dim,
        cfg.num_layers,
        use_gpu=True,
        dtype=cfg.dtype,
        device=torch.device("cuda"),
    )
    connector.cache_positions = False

    cpu_memory_objs, cpu_allocator = allocate_fragment_memory_objs(
        cfg, connector, starts, ends, source_device="cpu"
    )
    gpu_memory_objs, gpu_allocator = allocate_fragment_memory_objs(
        cfg, connector, starts, ends, source_device="cuda"
    )

    print_config(cfg, starts, ends)
    print()

    cpu_stage = benchmark_stage_breakdown(
        cfg,
        cpu_memory_objs,
        starts,
        ends,
        paged_kv,
        gap_positions,
        slot_mapping,
        gpu_kv_format,
    )
    gpu_stage = benchmark_stage_breakdown(
        cfg,
        gpu_memory_objs,
        starts,
        ends,
        paged_kv,
        gap_positions,
        slot_mapping,
        gpu_kv_format,
    )
    cpu_pipeline = benchmark_actual_pipeline(
        cfg,
        connector,
        cpu_memory_objs,
        starts,
        ends,
        paged_kv,
        slot_mapping,
    )
    gpu_pipeline = benchmark_actual_pipeline(
        cfg,
        connector,
        gpu_memory_objs,
        starts,
        ends,
        paged_kv,
        slot_mapping,
    )

    print_summary("cpu-pinned -> gpu buffer -> paged kv", cpu_stage, cpu_pipeline)
    print()
    print_summary("gpu -> gpu buffer -> paged kv", gpu_stage, gpu_pipeline)

    release_memory_objs(cpu_memory_objs)
    release_memory_objs(gpu_memory_objs)

    del cpu_allocator
    del gpu_allocator
    torch.cuda.synchronize()


def build_fragment_spans(
    num_fragments: int,
    fragment_tokens: int,
    gap_tokens: int,
) -> tuple[list[int], list[int]]:
    """Build `[start, end)` spans for reused fragments."""

    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for _ in range(num_fragments):
        starts.append(cursor)
        cursor += fragment_tokens
        ends.append(cursor)
        cursor += gap_tokens
    return starts, ends


def build_gap_positions(
    starts: list[int],
    ends: list[int],
    device: str,
) -> torch.Tensor:
    """Build the gap-position tensor used by the blend connector."""

    total_span = ends[-1] - starts[0]
    gap_mask = torch.ones(total_span, dtype=torch.bool, device=device)
    offset = starts[0]
    for start, end in zip(starts, ends, strict=False):
        gap_mask[start - offset : end - offset] = False
    return torch.where(gap_mask)[0]


def build_slot_mapping(
    total_span_tokens: int,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Build a random slot mapping covering the full span."""

    capacity = num_blocks * block_size
    slots = random.sample(range(capacity), total_span_tokens)
    return torch.tensor(slots, device="cuda", dtype=torch.int64)


def generate_vllm_paged_kv(cfg: BenchmarkConfig) -> list[torch.Tensor]:
    """Generate vLLM-style paged KV tensors."""

    shape = [
        2,
        cfg.num_blocks,
        cfg.block_size,
        cfg.num_heads,
        cfg.head_size,
    ]
    return [
        torch.rand(shape, dtype=cfg.dtype, device="cuda")
        for _ in range(cfg.num_layers)
    ]


def allocate_fragment_memory_objs(
    cfg: BenchmarkConfig,
    connector: VLLMBufferLayerwiseGPUConnector,
    starts: list[int],
    ends: list[int],
    source_device: str,
) -> tuple[list[list[MemoryObj]], object]:
    """Allocate fragment memory objects for all layers on the selected device."""

    allocator = create_allocator(cfg, source_device)
    layer_major_memory_objs: list[list[MemoryObj]] = []
    for _layer_id in range(cfg.num_layers):
        objs: list[MemoryObj] = []
        for start, end in zip(starts, ends, strict=False):
            shape = connector.get_shape(end - start)
            memory_obj = allocator.allocate(shape, cfg.dtype, fmt=MemoryFormat.KV_2TD)
            assert memory_obj is not None
            assert memory_obj.tensor is not None
            random_tensor = torch.rand(
                shape,
                dtype=cfg.dtype,
                device=source_device,
            )
            memory_obj.tensor.copy_(random_tensor)
            objs.append(memory_obj)
        layer_major_memory_objs.append(objs)
    return layer_major_memory_objs, allocator


def create_allocator(cfg: BenchmarkConfig, source_device: str) -> object:
    """Create a source allocator for fragment memory objects."""

    bytes_per_obj = (
        2
        * cfg.fragment_tokens
        * cfg.hidden_dim
        * torch.tensor([], dtype=cfg.dtype).element_size()
    )
    total_objs = cfg.num_layers * cfg.num_fragments
    pool_size = max(bytes_per_obj * total_objs + 4096, 1 << 20)

    if source_device == "cpu":
        pinned_buffer = torch.empty(pool_size, dtype=torch.uint8, pin_memory=True)
        return TensorMemoryAllocator(pinned_buffer)
    if source_device == "cuda":
        return GPUMemoryAllocator(pool_size, device="cuda")
    raise ValueError(f"Unsupported source device: {source_device}")


def benchmark_stage_breakdown(
    cfg: BenchmarkConfig,
    memory_objs: list[list[MemoryObj]],
    starts: list[int],
    ends: list[int],
    paged_kv: list[torch.Tensor],
    gap_positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    gpu_kv_format: object,
) -> dict[str, list[float]]:
    """Benchmark isolated copy, gap-zero, and paged-KV injection stages."""

    stats = {
        "copy_ms": [],
        "gap_zero_ms": [],
        "transfer_ms": [],
        "isolated_total_ms": [],
    }
    total_span = ends[-1] - starts[0]
    hidden_dim = cfg.hidden_dim
    tmp_allocator = GPUMemoryAllocator(
        2 * total_span * hidden_dim * torch.tensor([], dtype=cfg.dtype).element_size() + 4096,
        device="cuda",
    )
    tmp_gpu_buffer = tmp_allocator.allocate(
        torch.Size([2, total_span, hidden_dim]),
        cfg.dtype,
        fmt=MemoryFormat.KV_2TD,
    )
    assert tmp_gpu_buffer is not None
    assert tmp_gpu_buffer.tensor is not None

    for round_id in range(cfg.warmup_rounds + cfg.rounds):
        result = run_stage_breakdown_once(
            cfg,
            memory_objs,
            starts,
            ends,
            paged_kv,
            gap_positions,
            slot_mapping,
            gpu_kv_format,
            tmp_gpu_buffer,
        )
        if round_id >= cfg.warmup_rounds:
            for key, value in result.items():
                stats[key].append(value)

    tmp_gpu_buffer.ref_count_down()
    return stats


def run_stage_breakdown_once(
    cfg: BenchmarkConfig,
    memory_objs: list[list[MemoryObj]],
    starts: list[int],
    ends: list[int],
    paged_kv: list[torch.Tensor],
    gap_positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    gpu_kv_format: object,
    tmp_gpu_buffer: MemoryObj,
) -> dict[str, float]:
    """Run one isolated breakdown round without the connector pipeline."""

    assert tmp_gpu_buffer.tensor is not None
    copy_ms = 0.0
    gap_zero_ms = 0.0
    transfer_ms = 0.0
    offset = starts[0]
    stream = torch.cuda.Stream(device="cuda")

    for layer_id in range(cfg.num_layers):
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        gap_start = torch.cuda.Event(enable_timing=True)
        gap_end = torch.cuda.Event(enable_timing=True)
        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream):
            copy_start.record(stream)
            for start, end, memory_obj in zip(
                starts, ends, memory_objs[layer_id], strict=False
            ):
                assert memory_obj.tensor is not None
                tmp_gpu_buffer.tensor[0][start - offset : end - offset].copy_(
                    memory_obj.tensor[0], non_blocking=True
                )
                tmp_gpu_buffer.tensor[1][start - offset : end - offset].copy_(
                    memory_obj.tensor[1], non_blocking=True
                )
            copy_end.record(stream)

            if gap_positions.numel() > 0:
                gap_start.record(stream)
                tmp_gpu_buffer.tensor[:, gap_positions] = 0.0
                gap_end.record(stream)

            transfer_start.record(stream)
            lmc_ops.single_layer_kv_transfer(
                tmp_gpu_buffer.tensor,
                paged_kv[layer_id],
                slot_mapping,
                lmc_ops.TransferDirection.H2D,
                gpu_kv_format,
                token_major=False,
            )
            transfer_end.record(stream)

        stream.synchronize()
        copy_ms += copy_start.elapsed_time(copy_end)
        if gap_positions.numel() > 0:
            gap_zero_ms += gap_start.elapsed_time(gap_end)
        transfer_ms += transfer_start.elapsed_time(transfer_end)

    return {
        "copy_ms": copy_ms,
        "gap_zero_ms": gap_zero_ms,
        "transfer_ms": transfer_ms,
        "isolated_total_ms": copy_ms + gap_zero_ms + transfer_ms,
    }


def benchmark_actual_pipeline(
    cfg: BenchmarkConfig,
    connector: VLLMBufferLayerwiseGPUConnector,
    memory_objs: list[list[MemoryObj]],
    starts: list[int],
    ends: list[int],
    paged_kv: list[torch.Tensor],
    slot_mapping: torch.Tensor,
) -> list[float]:
    """Benchmark the actual `VLLMBufferLayerwiseGPUConnector.batched_to_gpu()` path."""

    samples: list[float] = []
    for round_id in range(cfg.warmup_rounds + cfg.rounds):
        elapsed_ms = run_actual_pipeline_once(
            connector,
            memory_objs,
            starts,
            ends,
            paged_kv,
            slot_mapping,
        )
        if round_id >= cfg.warmup_rounds:
            samples.append(elapsed_ms)
    return samples


def run_actual_pipeline_once(
    connector: VLLMBufferLayerwiseGPUConnector,
    memory_objs: list[list[MemoryObj]],
    starts: list[int],
    ends: list[int],
    paged_kv: list[torch.Tensor],
    slot_mapping: torch.Tensor,
) -> float:
    """Run one actual connector pipeline round and return wall time in ms."""

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=paged_kv,
        slot_mapping=slot_mapping,
    )
    next(consumer)
    for layer_memory_objs in memory_objs:
        consumer.send(layer_memory_objs)
    next(consumer)
    try:
        next(consumer)
    except StopIteration:
        pass
    torch.cuda.synchronize()
    return (time.perf_counter() - start_time) * 1000.0


def release_memory_objs(memory_objs: list[list[MemoryObj]]) -> None:
    """Release all allocated memory objects."""

    for layer_memory_objs in memory_objs:
        for memory_obj in layer_memory_objs:
            memory_obj.ref_count_down()


def print_config(
    cfg: BenchmarkConfig,
    starts: list[int],
    ends: list[int],
) -> None:
    """Print the benchmark configuration."""

    fragment_lengths = [end - start for start, end in zip(starts, ends, strict=False)]
    print("Benchmark configuration")
    print(f"  dtype: {cfg.dtype}")
    print(f"  num_layers: {cfg.num_layers}")
    print(f"  num_heads: {cfg.num_heads}")
    print(f"  head_size: {cfg.head_size}")
    print(f"  hidden_dim: {cfg.hidden_dim}")
    print(f"  paged_kv_capacity_tokens: {cfg.num_blocks * cfg.block_size}")
    print(f"  num_fragments: {cfg.num_fragments}")
    print(f"  fragment_lengths: {fragment_lengths}")
    print(f"  gap_tokens: {cfg.gap_tokens}")
    print(f"  total_span_tokens: {cfg.total_span_tokens}")
    print(f"  warmup_rounds: {cfg.warmup_rounds}")
    print(f"  measured_rounds: {cfg.rounds}")


def print_summary(
    title: str,
    stage_stats: dict[str, list[float]],
    pipeline_ms: list[float],
) -> None:
    """Print one source-path summary."""

    print(title)
    print(f"  source->gpu_buffer: {format_stats(stage_stats['copy_ms'])}")
    print(f"  gap_zero:           {format_stats(stage_stats['gap_zero_ms'])}")
    print(f"  gpu_buffer->paged:  {format_stats(stage_stats['transfer_ms'])}")
    print(f"  isolated_total:     {format_stats(stage_stats['isolated_total_ms'])}")
    print(f"  actual_pipeline:    {format_stats(pipeline_ms)}")


def format_stats(samples: list[float]) -> str:
    """Format a millisecond sample list as mean/p50/p95."""

    if not samples:
        return "n/a"
    ordered = sorted(samples)
    mean_ms = statistics.mean(samples)
    p50_ms = ordered[len(ordered) // 2]
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    p95_ms = ordered[p95_index]
    return f"mean={mean_ms:.3f} ms, p50={p50_ms:.3f} ms, p95={p95_ms:.3f} ms"


def _vllm_engine_type():
    """Return the vLLM engine enum without importing the full engine stack."""

    # First Party
    from lmcache.utils import EngineType

    return EngineType.VLLM


if __name__ == "__main__":
    main()
