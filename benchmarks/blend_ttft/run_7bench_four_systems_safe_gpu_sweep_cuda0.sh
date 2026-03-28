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
GPU_BUDGETS=(${GPU_BUDGETS:-2 2.5 3 3.5 4})
WORKLOADS=(${WORKLOADS:-memoryos memos amem skillsbench dspy_locomo memgas swe_skillsbench})
STARTUP_PREFILL_PLACEMENT="${STARTUP_PREFILL_PLACEMENT:-all_gpu}"
STARTUP_INITIAL_FILL_MODE="${STARTUP_INITIAL_FILL_MODE:-gpu_then_cpu}"

# Plain baselines do not depend on LMCache local budgets. Run them once and
# fan out the results to every budget point in summary.tsv.
PLAIN_BASELINE_GPU_BUDGET="${PLAIN_BASELINE_GPU_BUDGET:-4}"
PLAIN_BASELINE_CPU_BUDGET="${PLAIN_BASELINE_CPU_BUDGET:-4}"

# Current leak-free runtime path:
# - semantic hints are static/request-local canonical signals
# - optional online stats use past-only observations
OURS_BLEND_CONNECTOR_IMPL="${OURS_BLEND_CONNECTOR_IMPL:-fast}"
OURS_ONLINE_EXECUTION_POLICY="${OURS_ONLINE_EXECUTION_POLICY:-utility}"
OURS_ONLINE_EXECUTION_MODE="${OURS_ONLINE_EXECUTION_MODE:-blend}"
OURS_FRAGMENT_MANAGEMENT_POLICY="${OURS_FRAGMENT_MANAGEMENT_POLICY:-utility}"
OURS_ENABLE_SEMANTIC_HINTS="${OURS_ENABLE_SEMANTIC_HINTS:-1}"
OURS_USE_HISTORICAL_STATS="${OURS_USE_HISTORICAL_STATS:-1}"
OURS_ONLINE_GAP_ALPHA="${OURS_ONLINE_GAP_ALPHA:-0.5}"

SKILLSBENCH_DATASETS="${SKILLSBENCH_DATASETS:-dense_core}"
SWE_SKILLSBENCH_DATASETS="${SWE_SKILLSBENCH_DATASETS:-swe_obs_dense}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$BENCH_DIR/analysis_results/run7_four_systems_safe_gpuSweep_cpu${CPU_BUDGET}_cuda${CUDA_DEV}_${STAMP}}"

mkdir -p "$OUT_ROOT"/{logs,plain_baselines}
printf "tag\trc\tlog\n" > "$OUT_ROOT/failures.tsv"
printf "tag\tlog\n" > "$OUT_ROOT/success.tsv"
printf "budget_tag\tgpu_budget_gb\tcpu_budget_gb\n" > "$OUT_ROOT/budgets.tsv"

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
    swe_skillsbench) echo "$SWE_SKILLSBENCH_DATASETS" ;;
    dspy_locomo) echo "conv-26,conv-30,conv-41" ;;
    memgas) echo "locomo10_minilm" ;;
    *)
      echo "Unsupported workload: $workload" >&2
      return 1
      ;;
  esac
}

budget_tag_for() {
  local gpu_budget="$1"
  local cpu_budget="$2"
  local gpu_tag="${gpu_budget//./p}"
  local cpu_tag="${cpu_budget//./p}"
  echo "gpu${gpu_tag}_cpu${cpu_tag}"
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
echo "GPU_BUDGETS=${GPU_BUDGETS[*]}"
echo "CPU_BUDGET=$CPU_BUDGET"
echo "MAX_QA_PER_DATASET=$MAX_QA_PER_DATASET"
echo "STARTUP_PREFILL_PLACEMENT=$STARTUP_PREFILL_PLACEMENT"
echo "STARTUP_INITIAL_FILL_MODE=$STARTUP_INITIAL_FILL_MODE"
echo "PLAIN_BASELINE_GPU_BUDGET=$PLAIN_BASELINE_GPU_BUDGET"
echo "PLAIN_BASELINE_CPU_BUDGET=$PLAIN_BASELINE_CPU_BUDGET"
echo "OURS_BLEND_CONNECTOR_IMPL=$OURS_BLEND_CONNECTOR_IMPL"
echo "OURS_ONLINE_EXECUTION_POLICY=$OURS_ONLINE_EXECUTION_POLICY"
echo "OURS_ONLINE_EXECUTION_MODE=$OURS_ONLINE_EXECUTION_MODE"
echo "OURS_FRAGMENT_MANAGEMENT_POLICY=$OURS_FRAGMENT_MANAGEMENT_POLICY"
echo "OURS_ENABLE_SEMANTIC_HINTS=$OURS_ENABLE_SEMANTIC_HINTS"
echo "OURS_USE_HISTORICAL_STATS=$OURS_USE_HISTORICAL_STATS"
echo "OURS_ONLINE_GAP_ALPHA=$OURS_ONLINE_GAP_ALPHA"
echo "SKILLSBENCH_DATASETS=$SKILLSBENCH_DATASETS"
echo "SWE_SKILLSBENCH_DATASETS=$SWE_SKILLSBENCH_DATASETS"

for gpu_budget in "${GPU_BUDGETS[@]}"; do
  budget_tag="$(budget_tag_for "$gpu_budget" "$CPU_BUDGET")"
  budget_root="$OUT_ROOT/$budget_tag"
  mkdir -p "$budget_root"/{legacy_without,ours_with}
  echo -e "${budget_tag}\t${gpu_budget}\t${CPU_BUDGET}" >> "$OUT_ROOT/budgets.tsv"
done

# Run recompute / vLLM prefix once per workload.
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
    --max-local-gpu-size "$PLAIN_BASELINE_GPU_BUDGET"
    --max-local-cpu-size "$PLAIN_BASELINE_CPU_BUDGET"
    --gpu-lookahead 0
  )

  plain_out="$OUT_ROOT/plain_baselines/${workload}"
  mkdir -p "$plain_out"

  run_logged "plain_baselines__${workload}" \
    "$PY" "$BENCH_PY" \
    "${shared_args[@]}" \
    --methods no_prefix,native_prefix \
    --out-dir "$plain_out"
done

for gpu_budget in "${GPU_BUDGETS[@]}"; do
  budget_tag="$(budget_tag_for "$gpu_budget" "$CPU_BUDGET")"
  budget_root="$OUT_ROOT/$budget_tag"

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
      --max-local-gpu-size "$gpu_budget"
      --max-local-cpu-size "$CPU_BUDGET"
      --gpu-lookahead 0
      --methods explicit_blend
    )

    legacy_out="$budget_root/legacy_without/${workload}"
    ours_out="$budget_root/ours_with/${workload}"
    mkdir -p "$legacy_out" "$ours_out"

    run_logged "${budget_tag}__legacy_without__${workload}" \
      "$PY" "$BENCH_PY" \
      "${shared_args[@]}" \
      --blend-connector-impl legacy \
      --online-execution-policy fixed \
      --online-execution-mode blend \
      --fragment-management-policy static \
      --prefill-placement-policy "$STARTUP_PREFILL_PLACEMENT" \
      --initial-fill-mode "$STARTUP_INITIAL_FILL_MODE" \
      --admit-misses-to auto \
      --no-utility-enable-semantic-hints \
      --out-dir "$legacy_out"

    run_logged "${budget_tag}__ours_with__${workload}" \
      "$PY" "$BENCH_PY" \
      "${shared_args[@]}" \
      --blend-connector-impl "$OURS_BLEND_CONNECTOR_IMPL" \
      --online-execution-policy "$OURS_ONLINE_EXECUTION_POLICY" \
      --online-execution-mode "$OURS_ONLINE_EXECUTION_MODE" \
      --fragment-management-policy "$OURS_FRAGMENT_MANAGEMENT_POLICY" \
      --prefill-placement-policy "$STARTUP_PREFILL_PLACEMENT" \
      --initial-fill-mode "$STARTUP_INITIAL_FILL_MODE" \
      --admit-misses-to auto \
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
done

"$PY" - <<PY
from __future__ import annotations

import csv
import json
from pathlib import Path

out_root = Path(${OUT_ROOT@Q})
budget_rows = list(csv.DictReader((out_root / "budgets.tsv").open("r", encoding="utf-8"), delimiter="\t"))

rows = [
    "budget_tag\tgpu_budget_gb\tcpu_budget_gb\tworkload\tconfig\tmean_ttft_s\tp90_ttft_s\ttotal_wall_s\tthroughput_qps\thit_rate\tmean_shadow_hit_tokens\tmean_shadow_miss_tokens\tblend_requests\trecompute_requests"
]

plain_rows: dict[str, dict[str, dict[str, float | str]]] = {}
plain_root = out_root / "plain_baselines"
if plain_root.exists():
    for workload_dir in sorted(path for path in plain_root.iterdir() if path.is_dir()):
        result_path = workload_dir / "result.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for config in ("no_prefix", "native_prefix"):
            summary = payload.get(config)
            if summary is None:
                continue
            request_count = float(summary.get("request_count", 0) or 0)
            total_wall_s = float(summary.get("total_wall_s", 0.0) or 0.0)
            throughput_qps = (request_count / total_wall_s) if total_wall_s > 0.0 else 0.0
            plain_rows.setdefault(workload_dir.name, {})[config] = {
                "mean_ttft_s": float(summary.get("mean_ttft_s", 0.0) or 0.0),
                "p90_ttft_s": float(summary.get("p90_ttft_s", 0.0) or 0.0),
                "total_wall_s": total_wall_s,
                "throughput_qps": throughput_qps,
                "hit_rate": float(summary.get("cache_hit_rate", 0.0) or 0.0),
            }

for budget_row in budget_rows:
    budget_tag = str(budget_row["budget_tag"])
    gpu_budget_gb = str(budget_row["gpu_budget_gb"])
    cpu_budget_gb = str(budget_row["cpu_budget_gb"])

    for workload, configs in sorted(plain_rows.items()):
        for config, summary in sorted(configs.items()):
            rows.append(
                "\t".join(
                    [
                        budget_tag,
                        gpu_budget_gb,
                        cpu_budget_gb,
                        workload,
                        config,
                        f"{float(summary['mean_ttft_s']):.6f}",
                        f"{float(summary['p90_ttft_s']):.6f}",
                        f"{float(summary['total_wall_s']):.6f}",
                        f"{float(summary['throughput_qps']):.6f}",
                        f"{float(summary['hit_rate']):.6f}",
                        "0.0",
                        "0.0",
                        "0",
                        "0",
                    ]
                )
            )

    budget_root = out_root / budget_tag
    for config in ("legacy_without", "ours_with"):
        config_dir = budget_root / config
        if not config_dir.exists():
            continue
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
                        budget_tag,
                        gpu_budget_gb,
                        cpu_budget_gb,
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

summary_path = out_root / "summary.tsv"
summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"summary_tsv: {summary_path}")
PY

echo
echo "DONE"
echo "OUT_ROOT=$OUT_ROOT"
echo "SUCCESS_TSV=$OUT_ROOT/success.tsv"
echo "FAILURE_TSV=$OUT_ROOT/failures.tsv"
