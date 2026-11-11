#!/bin/bash
# Map every new predicted food name to an FNDDS code (mapper v1), then rescore all runs.
# First GPU: Qwen3-Embedding-8B search. Remaining GPUs: Qwen3.8-27B text selection.
#   sbatch -p gpu-a100 --gres=gpu:a100:2 --export=ALL,SNAPME_REPO=... hpc/atlas/map_and_score.sbatch
#   sbatch -p gpu-l40s --gres=gpu:l40s:3 ${SNAPME_EXCLUDE:+--exclude=$SNAPME_EXCLUDE} --export=ALL,SNAPME_REPO=... hpc/atlas/map_and_score.sbatch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=snapme_map
#SBATCH --output=%x_%j.out
set -euo pipefail
source "$SNAPME_REPO/hpc/atlas/env.sh"
module load apptainer/1.4.3
cd "$SNAPME_REPO"
PY="$SNAPME_REPO/.pixi/envs/default/bin/python"
SIF="$SNAPME_CONTAINERS/vllm_v0.32.0.sif"
EP=$(( 20000 + SLURM_JOB_ID % 20000 )); JP=$(( EP + 1 ))
EMB=$("$PY" -c "import tomllib;m=tomllib.load(open('configs/aux_models/qwen3-embedding-8b.toml','rb'));o,n=m['hf_repo'].split('/');print(f\"$HF_HUB_CACHE/models--{o}--{n}/snapshots/{m['revision']}\")")
python3 -c "import ctypes,sys; sys.exit(ctypes.CDLL('libcuda.so.1').cuInit(0))" || { echo "CUDA init failed on $(hostname)"; exit 5; }
LOG="$SNAPME_ROOT/logs/map_${SLURM_JOB_ID}"
mkdir -p "$LOG"

GPUS=(${CUDA_VISIBLE_DEVICES//,/ })   # GPUs allocated by Slurm
run_engine() {  # gpu index, log name, command...
  local gpu=$1 name=$2; shift 2
  APPTAINERENV_CUDA_VISIBLE_DEVICES=$gpu APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_HF_HOME="$HF_HOME" \
  APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT/$name" APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR/$name" \
  APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME/$name" APPTAINERENV_TMPDIR="$TMPDIR" \
  apptainer exec --nv --cleanenv --home "$TMPDIR/home" --env PYTHONNOUSERSITE=1 --bind "$SNAPME_ROOT,$TMPDIR" "$SIF" "$@" > "$LOG/$name.log" 2>&1 &
}
run_engine "${GPUS[0]}" embed vllm serve "$EMB" --served-model-name qwen3-embedding-8b --runner pooling \
  --host 127.0.0.1 --port $EP --max-model-len 2048 --gpu-memory-utilization 0.85
JUDGE_GPUS=$(IFS=,; echo "${GPUS[*]:1}")          # every GPU after the first; 1 A100 or 2 L40S
eval "$("$PY" -m snapme_eval.cli serve-cmd --model qwen3-8-27b --gpus $(( ${#GPUS[@]} - 1 )) --port $JP)"
run_engine "$JUDGE_GPUS" judge "${SNAPME_SERVE_CMD[@]}"
trap 'pkill -P $$ || true' EXIT

for p in $EP $JP; do
  for i in $(seq 1 240); do curl -sf --noproxy "*" "http://127.0.0.1:$p/health" >/dev/null && break; sleep 5; done
  curl -sf --noproxy "*" "http://127.0.0.1:$p/health" >/dev/null || { echo "engine on $p failed"; tail -n 40 "$LOG"/*.log; exit 3; }
done
echo "engines ready $(date -u +%FT%TZ)"

"$PY" -m snapme_eval.cli map --embed-url "http://127.0.0.1:$EP/v1" --judge-url "http://127.0.0.1:$JP/v1"
# Rescore runs of each pipeline's current version only (older versions and set-aside runs stay as they are).
CURRENT=$("$PY" -c "
import pathlib
from snapme_eval.paths import runs_dir
from snapme_eval.score.report import current
for p in sorted(runs_dir().glob('*/*/*/predictions.jsonl')):
    if not p.parent.parent.name.startswith('.') and current(p.parent.parent.name): print(p.parent)")
for run in $CURRENT; do
  echo "score $run"; "$PY" -m snapme_eval.cli score --run "$run" || echo "score failed: $run"
done
date -u +%FT%TZ
