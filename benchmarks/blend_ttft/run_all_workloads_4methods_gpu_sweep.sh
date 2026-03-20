#!/usr/bin/env bash
set -u -o pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
HELPER="$BENCH_DIR/run_single_method_gpu_workload.py"

export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$BENCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-/AI/HF_MODELS/Mistral-7B-Instruct-v0.2}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/processed}"
CUDA_DEV="${CUDA_DEV:-3}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.80}"
DTYPE="${DTYPE:-bfloat16}"
MAX_QA_PER_DATASET="${MAX_QA_PER_DATASET:-0}"
MAX_TOKENS="${MAX_TOKENS:-1}"
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-4h}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"

PORT_NO_PREFIX="${PORT_NO_PREFIX:-8014}"
PORT_PREFIX="${PORT_PREFIX:-8015}"
PORT_LEGACY="${PORT_LEGACY:-8016}"
PORT_FAST="${PORT_FAST:-8017}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$BENCH_DIR/analysis_results/nightly_all_methods_cpu0_gpu2_3_4_5_6_${STAMP}}"

mkdir -p "$OUT_ROOT"/{logs,no_prefix,native_prefix,cacheblend_legacy,cacheblend_fast}
printf "tag\trc\tlog\n" > "$OUT_ROOT/failures.tsv"
printf "tag\tlog\n" > "$OUT_ROOT/success.tsv"

cleanup_ports() {
  local p
  if command -v fuser >/dev/null 2>&1; then
    for p in "$PORT_NO_PREFIX" "$PORT_PREFIX" "$PORT_LEGACY" "$PORT_FAST"; do
      fuser -k "${p}/tcp" >/dev/null 2>&1 || true
    done
  fi
}

trap cleanup_ports EXIT

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
  cleanup_ports

  if timeout -k 5m "$TIMEOUT_PER_RUN" "$@" >"$log" 2>&1; then
    echo -e "${tag}\t${log}" >> "$OUT_ROOT/success.tsv"
    echo "[$(date '+%F %T')] OK $tag"
  else
    local rc=$?
    echo -e "${tag}\t${rc}\t${log}" >> "$OUT_ROOT/failures.tsv"
    echo "[$(date '+%F %T')] FAIL $tag rc=$rc log=$log"
  fi

  cleanup_ports
  sleep 3
  return 0
}

if [[ ! -x "$PY" ]]; then
  echo "Missing python executable: $PY" >&2
  exit 1
fi

if [[ ! -f "$HELPER" ]]; then
  echo "Missing helper script: $HELPER" >&2
  exit 1
fi

WORKLOADS=(memoryos memos amem skillsbench)
GPU_BUDGETS=(2 3 4 5 6)

echo "OUT_ROOT=$OUT_ROOT"
echo "CUDA_DEV=$CUDA_DEV"
echo "MODEL=$MODEL"
echo "DATA_ROOT=$DATA_ROOT"
echo "TIMEOUT_PER_RUN=$TIMEOUT_PER_RUN"

for workload in "${WORKLOADS[@]}"; do
  max_model_len="$(max_model_len_for "$workload")"

  for budget in "${GPU_BUDGETS[@]}"; do
    common_args=(
      --model "$MODEL"
      --data-root "$DATA_ROOT"
      --workload-kind "$workload"
      --max-qa-per-dataset "$MAX_QA_PER_DATASET"
      --chunk-size "$CHUNK_SIZE"
      --max-model-len "$max_model_len"
      --gpu-memory-utilization "$GPU_MEMORY_UTIL"
      --dtype "$DTYPE"
      --max-tokens "$MAX_TOKENS"
      --gpu-budget-gb "$budget"
      --cpu-budget-gb 0
      --prefill-placement-policy all_gpu
      --cuda-visible-devices "$CUDA_DEV"
    )

    run_logged "no_prefix__${workload}__gpu${budget}g" \
      "$PY" "$HELPER" \
      --method no_prefix \
      "${common_args[@]}" \
      --port "$PORT_NO_PREFIX" \
      --output-json "$OUT_ROOT/no_prefix/no_prefix__${workload}__gpu${budget}g.json"

    run_logged "native_prefix__${workload}__gpu${budget}g" \
      "$PY" "$HELPER" \
      --method native_prefix \
      "${common_args[@]}" \
      --port "$PORT_PREFIX" \
      --output-json "$OUT_ROOT/native_prefix/native_prefix__${workload}__gpu${budget}g.json"

    run_logged "cacheblend_legacy__${workload}__gpu${budget}g" \
      "$PY" "$HELPER" \
      --method cacheblend \
      "${common_args[@]}" \
      --blend-connector-impl legacy \
      --no-blend-internal-timing \
      --port "$PORT_LEGACY" \
      --output-json "$OUT_ROOT/cacheblend_legacy/cacheblend_legacy__${workload}__gpu${budget}g.json"

    run_logged "cacheblend_fast__${workload}__gpu${budget}g" \
      "$PY" "$HELPER" \
      --method cacheblend \
      "${common_args[@]}" \
      --blend-connector-impl fast \
      --no-blend-internal-timing \
      --port "$PORT_FAST" \
      --output-json "$OUT_ROOT/cacheblend_fast/cacheblend_fast__${workload}__gpu${budget}g.json"
  done
done

echo
echo "DONE"
echo "OUT_ROOT=$OUT_ROOT"
echo "SUCCESS_TSV=$OUT_ROOT/success.tsv"
echo "FAILURE_TSV=$OUT_ROOT/failures.tsv"
