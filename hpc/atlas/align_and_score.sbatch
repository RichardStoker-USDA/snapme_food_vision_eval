#!/bin/bash
# Grade current runs against ASA24 lines with align_v1 (gpt-oss-120b), then rescore them.
# Start one grader server per GPU and distribute requests across the servers.
#   sbatch -p gpu-a100 --gres=gpu:a100:4 --export=ALL,SNAPME_REPO=... hpc/atlas/align_and_score.sbatch
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
set -euo pipefail
source "$SNAPME_REPO/hpc/atlas/env.sh"
module load apptainer/1.4.3
cd "$SNAPME_REPO"
PY="$SNAPME_REPO/.pixi/envs/default/bin/python"
SIF="$SNAPME_CONTAINERS/vllm_v0.32.0.sif"
GRADER=$("$PY" -c "import tomllib;m=tomllib.load(open('configs/aux_models/gpt-oss-120b.toml','rb'));o,n=m['hf_repo'].split('/');print(f\"$HF_HUB_CACHE/models--{o}--{n}/snapshots/{m['revision']}\")")
python3 -c "import ctypes,sys; sys.exit(ctypes.CDLL('libcuda.so.1').cuInit(0))" || { echo "CUDA init failed on $(hostname)"; exit 5; }
LOG="$SNAPME_ROOT/logs/align_${SLURM_JOB_ID}"
mkdir -p "$LOG"

GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
BASE=$(( 20000 + SLURM_JOB_ID % 20000 ))
URLS=()
for k in "${!GPUS[@]}"; do
  port=$(( BASE + k ))
  APPTAINERENV_CUDA_VISIBLE_DEVICES=${GPUS[$k]} APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_HF_HOME="$HF_HOME" \
  APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT/grader$k" APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR/grader$k" \
  APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME/grader$k" APPTAINERENV_TMPDIR="$TMPDIR" \
  apptainer exec --nv --cleanenv --home "$TMPDIR/home" --env PYTHONNOUSERSITE=1 --bind "$SNAPME_ROOT,$TMPDIR" "$SIF" \
    vllm serve "$GRADER" --served-model-name gpt-oss-120b --host 127.0.0.1 --port $port \
    --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 128 > "$LOG/grader$k.log" 2>&1 &
  URLS+=(--url "http://127.0.0.1:$port/v1")
done
trap 'pkill -P $$ || true' EXIT

for k in "${!GPUS[@]}"; do
  port=$(( BASE + k ))
  for i in $(seq 1 360); do curl -sf --noproxy "*" "http://127.0.0.1:$port/health" >/dev/null && break; sleep 5; done
  curl -sf --noproxy "*" "http://127.0.0.1:$port/health" >/dev/null || { echo "grader $k failed"; tail -n 40 "$LOG/grader$k.log"; exit 3; }
done
echo "graders ready $(date -u +%FT%TZ)"

# SNAPME_ALIGN_PIPELINES adds space-separated method folders to the set graded and rescored.
ALSO=()
for p in ${SNAPME_ALIGN_PIPELINES:-}; do ALSO+=(--pipeline "$p"); done
"$PY" -m snapme_eval.cli align "${URLS[@]}" ${ALSO[@]+"${ALSO[@]}"} ${SNAPME_ALIGN_ARGS:-}
# With SNAPME_ALIGN_ARGS set for a pilot or share, skip rescoring; run score_all.sbatch after grading finishes.
[ -n "${SNAPME_ALIGN_ARGS:-}" ] && { echo "grading done, no rescore $(date -u +%FT%TZ)"; exit 0; }
CURRENT=$("$PY" -c "
import os
from snapme_eval.paths import runs_dir
from snapme_eval.score.report import current
also = set(os.environ.get('SNAPME_ALIGN_PIPELINES', '').split())
for p in sorted(runs_dir().glob('*/*/*/predictions.jsonl')):
    n = p.parent.parent.name
    if not n.startswith('.') and (current(n) or n in also): print(p.parent)")
for run in $CURRENT; do
  echo "score $run"; "$PY" -m snapme_eval.cli score --run "$run" || echo "score failed: $run"
done
date -u +%FT%TZ
