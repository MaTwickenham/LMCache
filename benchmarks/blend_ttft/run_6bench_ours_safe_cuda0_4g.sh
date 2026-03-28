#!/usr/bin/env bash
set -u -o pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
BENCH_PY="$BENCH_DIR/benchmark_explicit_fragments.py"

export PATH="$ROOT/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-/AI/HF_MODELS/Mistral-7B-Instruct-v0.2}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/processed}"
CUDA_DEV="${CUDA_DEV:-0}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.80}"
DTYPE="${DTYPE:-bfloat16}"
MAX_QA_PER_DATASET="${MAX_QA_PER_DATASET:-400}"
MAX_TOKENS="${MAX_TOKENS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-8h}"
CPU_BUDGET="${CPU_BUDGET:-4}"
GPU_BUDGET="${GPU_BUDGET:-4}"
WORKLOADS=(${WORKLOADS:-memoryos memos amem skillsbench dspy_locomo memgas})
STARTUP_PREFILL_PLACEMENT="${STARTUP_PREFILL_PLACEMENT:-all_gpu}"
STARTUP_INITIAL_FILL_MODE="${STARTUP_INITIAL_FILL_MODE:-gpu_then_cpu}"

# Current runtime TTFT path already uses the cleaned, future-leak-free semantics.
OURS_BLEND_CONNECTOR_IMPL="${OURS_BLEND_CONNECTOR_IMPL:-fast}"
OURS_ONLINE_EXECUTION_POLICY="${OURS_ONLINE_EXECUTION_POLICY:-utility}"
OURS_ONLINE_EXECUTION_MODE="${OURS_ONLINE_EXECUTION_MODE:-blend}"
OURS_FRAGMENT_MANAGEMENT_POLICY="${OURS_FRAGMENT_MANAGEMENT_POLICY:-utility}"
OURS_ENABLE_SEMANTIC_HINTS="${OURS_ENABLE_SEMANTIC_HINTS:-1}"
OURS_USE_HISTORICAL_STATS="${OURS_USE_HISTORICAL_STATS:-0}"
OURS_ONLINE_GAP_ALPHA="${OURS_ONLINE_GAP_ALPHA:-0.5}"

# Keep this optional. If provided, we will materialize a direct comparison table.
CACHEBLEND_SUMMARY_TSV="${CACHEBLEND_SUMMARY_TSV:-}"

# Keep Skills configurable because the repo has multiple settled variants.
SKILLSBENCH_DATASETS="${SKILLSBENCH_DATASETS:-dense_core}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$BENCH_DIR/analysis_results/run6_ours_safe_cpu${CPU_BUDGET}_gpu${GPU_BUDGET}_cuda${CUDA_DEV}_${STAMP}}"

mkdir -p "$OUT_ROOT"/{logs,ours_with}
printf "tag\trc\tlog\n" > "$OUT_ROOT/failures.tsv"
printf "tag\tlog\n" > "$OUT_ROOT/success.tsv"

max_model_len_for() {
  local workload="$1"
  case "$workload" in
    skillsbench|swe_skillsbench) echo 16384 ;;
    *) echo 4608 ;;
  esac
}

datasets_for() {
  local workload="$1"
  case "$workload" in
    memoryos) echo "conv26,conv30,conv41" ;;
    amem) echo "conv26,conv30,conv41" ;;
    memos) echo "locomo_conv0,locomo_conv1,locomo_conv2" ;;
    skillsbench) echo "$SKILLSBENCH_DATASETS" ;;
    dspy_locomo) echo "conv-26,conv-30,conv-41" ;;
    memgas) echo "locomo10_minilm" ;;
    *)
      echo "Unsupported workload: $workload" >&2
      return 1
      ;;
  esac
}

run_logged() {
  local tag="$1"
  shift
  local log="$OUT_ROOT/logs/${tag}.log"

  echo "[$(date '+%F %T')] START $tag"
  if timeout -k 5m "$TIMEOUT_PER_RUN" "$@" >"$log" 2>&1; then
    echo -e "${tag}\t${log}" >> "$OUT_ROOT/success.tsv"
    echo "[$(date '+%F %T')] OK $tag"
  else
    local rc=$?
    echo -e "${tag}\t${rc}\t${log}" >> "$OUT_ROOT/failures.tsv"
    echo "[$(date '+%F %T')] FAIL $tag rc=$rc log=$log"
  fi
  return 0
}

if [[ ! -x "$PY" ]]; then
  echo "Missing python executable: $PY" >&2
  exit 1
fi

if [[ ! -f "$BENCH_PY" ]]; then
  echo "Missing benchmark script: $BENCH_PY" >&2
  exit 1
fi

echo "OUT_ROOT=$OUT_ROOT"
echo "CUDA_DEV=$CUDA_DEV"
echo "MODEL=$MODEL"
echo "DATA_ROOT=$DATA_ROOT"
echo "TIMEOUT_PER_RUN=$TIMEOUT_PER_RUN"
echo "WORKLOADS=${WORKLOADS[*]}"
echo "CPU_BUDGET=$CPU_BUDGET"
echo "GPU_BUDGET=$GPU_BUDGET"
echo "MAX_QA_PER_DATASET=$MAX_QA_PER_DATASET"
echo "STARTUP_PREFILL_PLACEMENT=$STARTUP_PREFILL_PLACEMENT"
echo "STARTUP_INITIAL_FILL_MODE=$STARTUP_INITIAL_FILL_MODE"
echo "OURS_BLEND_CONNECTOR_IMPL=$OURS_BLEND_CONNECTOR_IMPL"
echo "OURS_ONLINE_EXECUTION_POLICY=$OURS_ONLINE_EXECUTION_POLICY"
echo "OURS_ONLINE_EXECUTION_MODE=$OURS_ONLINE_EXECUTION_MODE"
echo "OURS_FRAGMENT_MANAGEMENT_POLICY=$OURS_FRAGMENT_MANAGEMENT_POLICY"
echo "OURS_ENABLE_SEMANTIC_HINTS=$OURS_ENABLE_SEMANTIC_HINTS"
echo "OURS_USE_HISTORICAL_STATS=$OURS_USE_HISTORICAL_STATS"
echo "OURS_ONLINE_GAP_ALPHA=$OURS_ONLINE_GAP_ALPHA"
echo "SKILLSBENCH_DATASETS=$SKILLSBENCH_DATASETS"
if [[ -n "$CACHEBLEND_SUMMARY_TSV" ]]; then
  echo "CACHEBLEND_SUMMARY_TSV=$CACHEBLEND_SUMMARY_TSV"
fi

for workload in "${WORKLOADS[@]}"; do
  max_model_len="$(max_model_len_for "$workload")"
  datasets="$(datasets_for "$workload")"

  shared_args=(
    --workload-kind "$workload"
    --datasets "$datasets"
    --model "$MODEL"
    --data-root "$DATA_ROOT"
    --cuda-visible-devices "$CUDA_DEV"
    --chunk-size "$CHUNK_SIZE"
    --prompt-layout memory_first
    --fragment-order-policy auto
    --prefill-order-policy sorted
    --max-qa-per-dataset "$MAX_QA_PER_DATASET"
    --max-model-len "$max_model_len"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --dtype "$DTYPE"
    --max-tokens "$MAX_TOKENS"
    --blend-check-layers 1
    --blend-recompute-ratios 0.15
    --no-blend-internal-timing
    --no-blend-engine-enable-prefix-caching
    --max-local-gpu-size "$GPU_BUDGET"
    --max-local-cpu-size "$CPU_BUDGET"
    --gpu-lookahead 0
    --prefill-placement-policy "$STARTUP_PREFILL_PLACEMENT"
    --initial-fill-mode "$STARTUP_INITIAL_FILL_MODE"
    --admit-misses-to auto
    --methods explicit_blend
  )

  ours_out="$OUT_ROOT/ours_with/${workload}"
  mkdir -p "$ours_out"

  run_logged "ours_with__${workload}" \
    "$PY" "$BENCH_PY" \
    "${shared_args[@]}" \
    --blend-connector-impl "$OURS_BLEND_CONNECTOR_IMPL" \
    --online-execution-policy "$OURS_ONLINE_EXECUTION_POLICY" \
    --online-execution-mode "$OURS_ONLINE_EXECUTION_MODE" \
    --fragment-management-policy "$OURS_FRAGMENT_MANAGEMENT_POLICY" \
    --online-gap-alpha "$OURS_ONLINE_GAP_ALPHA" \
    $(if [[ "$OURS_ENABLE_SEMANTIC_HINTS" == "1" ]]; then
        printf '%s' "--utility-enable-semantic-hints"
      else
        printf '%s' "--no-utility-enable-semantic-hints"
      fi) \
    $(if [[ "$OURS_USE_HISTORICAL_STATS" == "1" ]]; then
        printf '%s' "--online-use-historical-stats"
      else
        printf '%s' "--no-online-use-historical-stats"
      fi) \
    --out-dir "$ours_out"
done

"$PY" - <<PY
from __future__ import annotations

import csv
import json
from pathlib import Path

out_root = Path(${OUT_ROOT@Q})
compare_path = Path(${CACHEBLEND_SUMMARY_TSV@Q}) if ${CACHEBLEND_SUMMARY_TSV@Q} else None

rows = [
    "workload\tconfig\tmean_ttft_s\tp90_ttft_s\ttotal_wall_s\tthroughput_qps\thit_rate\tmean_shadow_hit_tokens\tmean_shadow_miss_tokens\tblend_requests\trecompute_requests",
]


def append_explicit_rows(config: str) -> None:
    config_dir = out_root / config
    if not config_dir.exists():
        return
    for workload_dir in sorted(path for path in config_dir.iterdir() if path.is_dir()):
        result_path = workload_dir / "result.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary = payload["explicit_blend"]
        online = summary["online"]
        request_count = float(online.get("request_count", 0) or 0)
        total_wall_s = float(online.get("total_wall_s", 0.0) or 0.0)
        throughput_qps = (request_count / total_wall_s) if total_wall_s > 0.0 else 0.0
        hit_tokens = float(summary.get("mean_shadow_hit_tokens", 0.0) or 0.0)
        miss_tokens = float(summary.get("mean_shadow_miss_tokens", 0.0) or 0.0)
        denom = hit_tokens + miss_tokens
        hit_rate = (hit_tokens / denom) if denom > 0.0 else 0.0
        rows.append(
            "\t".join(
                [
                    workload_dir.name,
                    config,
                    f"{float(online.get('mean_ttft_s', 0.0) or 0.0):.6f}",
                    f"{float(online.get('p90_ttft_s', 0.0) or 0.0):.6f}",
                    f"{total_wall_s:.6f}",
                    f"{throughput_qps:.6f}",
                    f"{hit_rate:.6f}",
                    f"{hit_tokens:.1f}",
                    f"{miss_tokens:.1f}",
                    str(int(summary.get('utility_mode_blend_requests', 0) or 0)),
                    str(int(summary.get('utility_mode_recompute_requests', 0) or 0)),
                ]
            )
        )


append_explicit_rows("ours_with")

summary_path = out_root / "summary.tsv"
summary_path.write_text("\\n".join(rows) + "\\n", encoding="utf-8")
print(f"summary_tsv: {summary_path}")

if compare_path is not None and compare_path.exists():
    baseline_rows: dict[str, dict[str, str]] = {}
    with compare_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\\t")
        for row in reader:
            if row.get("config") != "legacy_without":
                continue
            workload = str(row.get("workload", "") or "").strip()
            if workload:
                baseline_rows[workload] = row

    ours_rows: dict[str, dict[str, str]] = {}
    with summary_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\\t")
        for row in reader:
            workload = str(row.get("workload", "") or "").strip()
            if workload:
                ours_rows[workload] = row

    compare_lines = [
        "workload\tcacheblend_p90_ms\tours_p90_ms\ttpft_delta_pct\tcacheblend_thrpt\tours_thrpt\tthrpt_delta_pct\tcacheblend_hit_rate\tours_hit_rate"
    ]
    for workload in sorted(ours_rows):
        base = baseline_rows.get(workload)
        ours = ours_rows[workload]
        if base is None:
            continue
        base_p90 = float(base.get("p90_ttft_s", 0.0) or 0.0) * 1000.0
        ours_p90 = float(ours.get("p90_ttft_s", 0.0) or 0.0) * 1000.0
        ttft_delta_pct = ((base_p90 - ours_p90) / base_p90 * 100.0) if base_p90 > 0.0 else 0.0

        base_thrpt = float(base.get("throughput_qps", 0.0) or 0.0)
        ours_thrpt = float(ours.get("throughput_qps", 0.0) or 0.0)
        thrpt_delta_pct = ((ours_thrpt - base_thrpt) / base_thrpt * 100.0) if base_thrpt > 0.0 else 0.0

        compare_lines.append(
            "\t".join(
                [
                    workload,
                    f"{base_p90:.1f}",
                    f"{ours_p90:.1f}",
                    f"{ttft_delta_pct:+.2f}",
                    f"{base_thrpt:.6f}",
                    f"{ours_thrpt:.6f}",
                    f"{thrpt_delta_pct:+.2f}",
                    str(base.get("hit_rate", "0")),
                    str(ours.get("hit_rate", "0")),
                ]
            )
        )

    compare_out = out_root / "compare_vs_cacheblend.tsv"
    compare_out.write_text("\\n".join(compare_lines) + "\\n", encoding="utf-8")
    print(f"compare_tsv: {compare_out}")
elif compare_path is not None:
    print(f"compare_skipped_missing_baseline: {compare_path}")
PY

echo
echo "DONE"
echo "OUT_ROOT=$OUT_ROOT"
echo "SUCCESS_TSV=$OUT_ROOT/success.tsv"
echo "FAILURE_TSV=$OUT_ROOT/failures.tsv"
