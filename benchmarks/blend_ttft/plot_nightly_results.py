#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
import matplotlib.pyplot as plt  # noqa: E402


METHOD_ORDER = [
    "no_prefix",
    "native_prefix",
    "cacheblend_legacy",
    "cacheblend_fast",
]
METHOD_LABELS = {
    "no_prefix": "Recompute",
    "native_prefix": "Prefix",
    "cacheblend_legacy": "CacheBlend",
    "cacheblend_fast": "Ours",
}
METHOD_COLORS = {
    "no_prefix": "#4C9F70",
    "native_prefix": "#6C757D",
    "cacheblend_legacy": "#2E6FBB",
    "cacheblend_fast": "#BC4749",
}
WORKLOAD_ORDER = [
    "memos",
    "memoryos",
    "amem",
    "skillsbench",
]
WORKLOAD_LABELS = {
    "memos": "MemOS",
    "memoryos": "MemoryOS",
    "amem": "A-Mem",
    "skillsbench": "SkillsBench",
}


@dataclass(frozen=True)
class ResultPoint:
    method: str
    workload: str
    gpu_budget_gb: float
    mean_ttft_s: float
    p90_ttft_s: float
    mean_cached_tokens: float | None
    cache_hit_rate: float
    path: Path


def parse_args() -> argparse.Namespace:
    default_run_dir = (
        Path(__file__).resolve().parent
        / "analysis_results"
        / "nightly_all_methods_cpu0_gpu2_3_4_5_6_20260319_235702"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Plot nightly GPU sweep results as a multi-system grouped bar chart "
            "and a per-workload Pareto panel figure."
        )
    )
    parser.add_argument("--run-dir", type=str, default=str(default_run_dir))
    parser.add_argument("--multi-gpu-gb", type=float, default=4.0)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--annotate", action="store_true")
    return parser.parse_args()


def _load_point(path: Path) -> ResultPoint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = str(payload["method"])
    workload = str(payload["config"]["workload_kind"])
    gpu_budget_gb = float(payload["config"]["gpu_budget_gb"])
    result_kind = str(payload["result_kind"])
    result_obj = payload["result"]
    if result_kind == "latency_summary":
        summary = result_obj
    elif result_kind == "blend_chunk_result":
        summary = result_obj["online"]
    else:
        raise ValueError(f"Unsupported result_kind={result_kind!r} in {path}")
    return ResultPoint(
        method=method,
        workload=workload,
        gpu_budget_gb=gpu_budget_gb,
        mean_ttft_s=float(summary["mean_ttft_s"]),
        p90_ttft_s=float(summary["p90_ttft_s"]),
        mean_cached_tokens=(
            None
            if summary.get("mean_cached_tokens") is None
            else float(summary["mean_cached_tokens"])
        ),
        cache_hit_rate=float(summary["cache_hit_rate"]),
        path=path,
    )


def load_results(run_dir: Path) -> Dict[str, Dict[float, Dict[str, ResultPoint]]]:
    table: Dict[str, Dict[float, Dict[str, ResultPoint]]] = {}
    for path in sorted(run_dir.rglob("*.json")):
        point = _load_point(path)
        table.setdefault(point.workload, {}).setdefault(point.gpu_budget_gb, {})[
            point.method
        ] = point
    return table


def _save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_multisystem(
    table: Dict[str, Dict[float, Dict[str, ResultPoint]]],
    *,
    gpu_budget_gb: float,
    output_base: Path,
    dpi: int,
    annotate: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.6), constrained_layout=False)
    x = list(range(len(WORKLOAD_ORDER)))
    width = 0.18
    center = (len(METHOD_ORDER) - 1) / 2.0
    offsets = {method: (idx - center) * width for idx, method in enumerate(METHOD_ORDER)}

    for method in METHOD_ORDER:
        xs: List[float] = []
        ys: List[float] = []
        for i, workload in enumerate(WORKLOAD_ORDER):
            point = table.get(workload, {}).get(gpu_budget_gb, {}).get(method)
            baseline = table.get(workload, {}).get(gpu_budget_gb, {}).get("no_prefix")
            if point is None or baseline is None:
                continue
            normalized_ttft = point.mean_ttft_s / baseline.mean_ttft_s
            xs.append(i + offsets[method])
            ys.append(normalized_ttft)

        bars = ax.bar(
            xs,
            ys,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        if annotate:
            for bar in bars:
                height = float(bar.get_height())
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_LABELS[w] for w in WORKLOAD_ORDER])
    ax.set_ylabel("Normalized Mean TTFT")
    ax.set_title(
        f"Normalized Mean TTFT Across Workloads (Recompute = 1.0, GPU Budget = {gpu_budget_gb:.0f} GB)"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.axhline(1.0, color="#444444", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    _save_figure(fig, output_base, dpi)


def plot_pareto_panels(
    table: Dict[str, Dict[float, Dict[str, ResultPoint]]],
    *,
    output_base: Path,
    dpi: int,
    annotate: bool,
) -> None:
    fig, axs = plt.subplots(
        1,
        len(WORKLOAD_ORDER),
        figsize=(4.2 * len(WORKLOAD_ORDER), 4.1),
        constrained_layout=False,
    )
    if len(WORKLOAD_ORDER) == 1:
        axs = [axs]

    for ax, workload in zip(axs, WORKLOAD_ORDER):
        for method in METHOD_ORDER:
            points = []
            for gpu_budget_gb in sorted(table.get(workload, {}).keys()):
                point = table.get(workload, {}).get(gpu_budget_gb, {}).get(method)
                if point is None:
                    continue
                points.append(point)
            if not points:
                continue

            xs = [p.mean_ttft_s for p in points]
            ys = [p.gpu_budget_gb for p in points]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                markersize=4.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            if annotate:
                for point in points:
                    ax.text(
                        point.mean_ttft_s,
                        point.gpu_budget_gb,
                        f"{point.gpu_budget_gb:.0f}G",
                        fontsize=7,
                        color=METHOD_COLORS[method],
                    )

        ax.set_title(WORKLOAD_LABELS[workload])
        ax.set_xlabel("Mean TTFT (s)")
        ax.grid(True, linestyle="--", alpha=0.28)

    axs[0].set_ylabel("GPU RAM Budget (GB)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save_figure(fig, output_base, dpi)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "figures")
    )
    table = load_results(run_dir)

    plot_multisystem(
        table,
        gpu_budget_gb=float(args.multi_gpu_gb),
        output_base=output_dir / f"multisystem_mean_ttft_gpu{int(args.multi_gpu_gb)}g",
        dpi=int(args.dpi),
        annotate=bool(args.annotate),
    )
    plot_pareto_panels(
        table,
        output_base=output_dir / "pareto_mean_ttft_gpu_budget_panels",
        dpi=int(args.dpi),
        annotate=bool(args.annotate),
    )
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
