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
MAX_QA_PER_DATASET="${MAX_QA_PER_DATASET:-0}"
MAX_TOKENS="${MAX_TOKENS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-6h}"
CPU_BUDGET="${CPU_BUDGET:-2}"
GPU_BUDGET="${GPU_BUDGET:-4}"
WORKLOADS=(${WORKLOADS:-memoryos memos})

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$BENCH_DIR/analysis_results/semantic_hints_ablation_cpu${CPU_BUDGET}_gpu${GPU_BUDGET}_${STAMP}}"

mkdir -p "$OUT_ROOT"/{logs,with_hints,without_hints}
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
echo "WORKLOADS=${WORKLOADS[*]}"
echo "CPU_BUDGET=$CPU_BUDGET"
echo "GPU_BUDGET=$GPU_BUDGET"

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
    --prefill-placement-policy all_cpu
    --max-qa-per-dataset "$MAX_QA_PER_DATASET"
    --max-model-len "$max_model_len"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --dtype "$DTYPE"
    --max-tokens "$MAX_TOKENS"
    --blend-connector-impl fast
    --no-blend-internal-timing
    --no-blend-engine-enable-prefix-caching
    --online-execution-policy utility
    --online-execution-mode blend
    --fragment-management-policy utility
    --initial-fill-mode none
    --admit-misses-to auto
    --gpu-lookahead 0
    --max-local-gpu-size "$GPU_BUDGET"
    --max-local-cpu-size "$CPU_BUDGET"
  )

  with_hints_out="$OUT_ROOT/with_hints/${workload}"
  without_hints_out="$OUT_ROOT/without_hints/${workload}"
  mkdir -p "$with_hints_out" "$without_hints_out"

  run_logged "with_hints__${workload}" \
    "$PY" "$BENCH_PY" \
    "${shared_args[@]}" \
    --utility-enable-semantic-hints \
    --out-dir "$with_hints_out"

  run_logged "without_hints__${workload}" \
    "$PY" "$BENCH_PY" \
    "${shared_args[@]}" \
    --no-utility-enable-semantic-hints \
    --out-dir "$without_hints_out"
done

"$PY" - <<PY
from __future__ import annotations

import json
from pathlib import Path

out_root = Path(${OUT_ROOT@Q})
rows = [
    "workload\ttag\tmean_ttft_s\tp90_ttft_s\tshadow_hit_rate\tmean_shadow_hit_tokens\tmean_shadow_miss_tokens",
]

for tag in ("with_hints", "without_hints"):
    tag_dir = out_root / tag
    if not tag_dir.exists():
        continue
    for workload_dir in sorted(path for path in tag_dir.iterdir() if path.is_dir()):
        result_path = workload_dir / "result.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary = payload["explicit_blend"]
        hit_tokens = float(summary.get("mean_shadow_hit_tokens", 0.0) or 0.0)
        miss_tokens = float(summary.get("mean_shadow_miss_tokens", 0.0) or 0.0)
        denom = hit_tokens + miss_tokens
        hit_rate = (hit_tokens / denom) if denom > 0.0 else 0.0
        online = summary["online"]
        rows.append(
            "\t".join(
                [
                    workload_dir.name,
                    tag,
                    f"{float(online.get('mean_ttft_s', 0.0) or 0.0):.6f}",
                    f"{float(online.get('p90_ttft_s', 0.0) or 0.0):.6f}",
                    f"{hit_rate:.6f}",
                    f"{hit_tokens:.1f}",
                    f"{miss_tokens:.1f}",
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
