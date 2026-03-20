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
CUDA_DEV="${CUDA_DEV:-2}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.80}"
DTYPE="${DTYPE:-bfloat16}"
MAX_QA_PER_DATASET="${MAX_QA_PER_DATASET:-0}"
MAX_TOKENS="${MAX_TOKENS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-6h}"
CPU_BUDGET="${CPU_BUDGET:-2}"

# This script drives the canonical TTFT benchmark entrypoint.
# Both methods run under the same memory configuration:
# CPU budget is fixed and GPU budget is swept.
# The method difference is connector/runtime policy, not the configured capacity.
WORKLOADS=(${WORKLOADS:-memoryos memos amem skillsbench})
GPU_BUDGETS=(${GPU_BUDGETS:-3 4 5})

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$BENCH_DIR/analysis_results/canonical_ttft_sweep_cpu${CPU_BUDGET}_gpu3_4_5_${STAMP}}"

mkdir -p "$OUT_ROOT"/{logs,legacy,ours}
printf "tag\trc\tlog\n" > "$OUT_ROOT/failures.tsv"
printf "tag\tlog\n" > "$OUT_ROOT/success.tsv"

max_model_len_for() {
  local workload="$1"
  case "$workload" in
    skillsbench) echo 16384 ;;
    *) echo 4608 ;;
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
echo "GPU_BUDGETS=${GPU_BUDGETS[*]}"

for workload in "${WORKLOADS[@]}"; do
  max_model_len="$(max_model_len_for "$workload")"
  shared_args=(
    --workload-kind "$workload"
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
    --no-blend-internal-timing
    --no-blend-engine-enable-prefix-caching
    --initial-fill-mode cpu_then_gpu
    --gpu-lookahead 0
  )

  for budget in "${GPU_BUDGETS[@]}"; do
    legacy_out="$OUT_ROOT/legacy/${workload}/cpu${CPU_BUDGET}g_gpu${budget}g"
    ours_out="$OUT_ROOT/ours/${workload}/cpu${CPU_BUDGET}g_gpu${budget}g"
    mkdir -p "$legacy_out" "$ours_out"

    run_logged "legacy__${workload}__cpu${CPU_BUDGET}g__gpu${budget}g" \
      "$PY" "$BENCH_PY" \
      "${shared_args[@]}" \
      --max-local-gpu-size "$budget" \
      --max-local-cpu-size "$CPU_BUDGET" \
      --prefill-placement-policy all_gpu \
      --blend-connector-impl legacy \
      --online-execution-policy fixed \
      --online-execution-mode blend \
      --admit-misses-to none \
      --out-dir "$legacy_out"

    run_logged "ours__${workload}__cpu${CPU_BUDGET}g__gpu${budget}g" \
      "$PY" "$BENCH_PY" \
      "${shared_args[@]}" \
      --max-local-gpu-size "$budget" \
      --max-local-cpu-size "$CPU_BUDGET" \
      --prefill-placement-policy all_gpu \
      --blend-connector-impl fast \
      --online-execution-policy utility \
      --online-execution-mode blend \
      --admit-misses-to auto \
      --out-dir "$ours_out"
  done
done

echo
echo "DONE"
echo "OUT_ROOT=$OUT_ROOT"
echo "SUCCESS_TSV=$OUT_ROOT/success.tsv"
echo "FAILURE_TSV=$OUT_ROOT/failures.tsv"
