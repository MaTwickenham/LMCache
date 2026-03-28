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
MAX_QA_PER_DATASET="${MAX_QA_PER_DATASET:-100}"
MAX_TOKENS="${MAX_TOKENS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-8h}"
CPU_BUDGET="${CPU_BUDGET:-4}"
GPU_BUDGET="${GPU_BUDGET:-4}"
WORKLOADS=(${WORKLOADS:-skillsbench swe_skillsbench})
STARTUP_PREFILL_PLACEMENT="${STARTUP_PREFILL_PLACEMENT:-all_gpu}"
STARTUP_INITIAL_FILL_MODE="${STARTUP_INITIAL_FILL_MODE:-gpu_then_cpu}"
OURS_USE_HISTORICAL_STATS="${OURS_USE_HISTORICAL_STATS:-0}"
OURS_ONLINE_GAP_ALPHA="${OURS_ONLINE_GAP_ALPHA:-0.5}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$BENCH_DIR/analysis_results/skills_four_systems_fair_cpu${CPU_BUDGET}_gpu${GPU_BUDGET}_cuda${CUDA_DEV}_${STAMP}}"

mkdir -p "$OUT_ROOT"/{logs,plain_baselines,legacy_without,ours_with}
printf "tag\trc\tlog\n" > "$OUT_ROOT/failures.tsv"
printf "tag\tlog\n" > "$OUT_ROOT/success.tsv"

max_model_len_for() {
  local workload="$1"
  case "$workload" in
    skillsbench|swe_skillsbench) echo 16384 ;;
    *)
      echo "Unsupported workload for this script: $workload" >&2
      return 1
      ;;
  esac
}

datasets_for() {
  local workload="$1"
  case "$workload" in
    skillsbench)
      echo "stoch100_exact_seed0,stoch100_exact_seed1,stoch100_exact_seed2,stoch100_exact_seed3,stoch100_exact_seed4"
      ;;
    swe_skillsbench)
      echo "swe_stoch100_seed0,swe_stoch100_seed1,swe_stoch100_seed2,swe_stoch100_seed3,swe_stoch100_seed4"
      ;;
    *)
      echo "Unsupported workload for this script: $workload" >&2
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
echo "OURS_USE_HISTORICAL_STATS=$OURS_USE_HISTORICAL_STATS"
echo "OURS_ONLINE_GAP_ALPHA=$OURS_ONLINE_GAP_ALPHA"

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
  )

  plain_out="$OUT_ROOT/plain_baselines/${workload}"
  legacy_out="$OUT_ROOT/legacy_without/${workload}"
  ours_out="$OUT_ROOT/ours_with/${workload}"
  mkdir -p "$plain_out" "$legacy_out" "$ours_out"

  run_logged "plain_baselines__${workload}" \
    "$PY" "$BENCH_PY" \
    "${shared_args[@]}" \
    --methods no_prefix,native_prefix \
    --out-dir "$plain_out"

  run_logged "legacy_without__${workload}" \
    "$PY" "$BENCH_PY" \
    "${shared_args[@]}" \
    --methods explicit_blend \
    --blend-connector-impl legacy \
    --online-execution-policy fixed \
    --online-execution-mode blend \
    --fragment-management-policy static \
    --prefill-placement-policy "$STARTUP_PREFILL_PLACEMENT" \
    --initial-fill-mode "$STARTUP_INITIAL_FILL_MODE" \
    --admit-misses-to auto \
    --no-utility-enable-semantic-hints \
    --out-dir "$legacy_out"

  run_logged "ours_with__${workload}" \
    "$PY" "$BENCH_PY" \
    "${shared_args[@]}" \
    --methods explicit_blend \
    --blend-connector-impl fast \
    --online-execution-policy utility \
    --online-execution-mode blend \
    --fragment-management-policy utility \
    --prefill-placement-policy "$STARTUP_PREFILL_PLACEMENT" \
    --initial-fill-mode "$STARTUP_INITIAL_FILL_MODE" \
    --admit-misses-to auto \
    --online-gap-alpha "$OURS_ONLINE_GAP_ALPHA" \
    --utility-enable-semantic-hints \
    $(if [[ "$OURS_USE_HISTORICAL_STATS" == "1" ]]; then
        printf '%s' "--online-use-historical-stats"
      else
        printf '%s' "--no-online-use-historical-stats"
      fi) \
    --out-dir "$ours_out"
done

"$PY" - <<PY
from __future__ import annotations

import json
from pathlib import Path

out_root = Path(${OUT_ROOT@Q})
rows = [
    "workload\tconfig\tmean_ttft_s\tp90_ttft_s\ttotal_wall_s\tthroughput_qps\thit_rate\tmean_shadow_hit_tokens\tmean_shadow_miss_tokens\tblend_requests\trecompute_requests",
]


def append_plain_rows(config_dir: Path) -> None:
    if not config_dir.exists():
        return
    for workload_dir in sorted(path for path in config_dir.iterdir() if path.is_dir()):
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
            rows.append(
                "\t".join(
                    [
                        workload_dir.name,
                        config,
                        f"{float(summary.get('mean_ttft_s', 0.0) or 0.0):.6f}",
                        f"{float(summary.get('p90_ttft_s', 0.0) or 0.0):.6f}",
                        f"{total_wall_s:.6f}",
                        f"{throughput_qps:.6f}",
                        f"{float(summary.get('cache_hit_rate', 0.0) or 0.0):.6f}",
                        "0.0",
                        "0.0",
                        "0",
                        "0",
                    ]
                )
            )


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
                    str(int(summary.get("utility_mode_blend_requests", 0) or 0)),
                    str(int(summary.get("utility_mode_recompute_requests", 0) or 0)),
                ]
            )
        )


append_plain_rows(out_root / "plain_baselines")
append_explicit_rows("legacy_without")
append_explicit_rows("ours_with")

summary_path = out_root / "summary.tsv"
summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"summary_tsv: {summary_path}")
PY

echo
echo "DONE"
echo "OUT_ROOT=$OUT_ROOT"
echo "SUCCESS_TSV=$OUT_ROOT/success.tsv"
echo "FAILURE_TSV=$OUT_ROOT/failures.tsv"
