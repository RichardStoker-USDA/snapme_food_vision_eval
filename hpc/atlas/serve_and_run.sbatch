#!/bin/bash
# Serve one registry model inside its engine container and run one pipeline against it.
# Submit through hpc/atlas/submit.py, which sets the partition, GPU count and names:
#   sbatch -p gpu-l40s --gres=gpu:l40s:2 --export=ALL,SNAPME_REPO=...,SNAPME_MODEL=...,SNAPME_SYSTEM=direct,SNAPME_GPUS=2 \
#          hpc/atlas/serve_and_run.sbatch
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=snapme_run
#SBATCH --output=%x_%j.out
set -euo pipefail
source "$SNAPME_REPO/hpc/atlas/env.sh"
module load apptainer/1.4.3
cd "$SNAPME_REPO"

: "${SNAPME_MODEL:?}" "${SNAPME_SYSTEM:=direct}" "${SNAPME_GPUS:=1}" "${SNAPME_GT:=snapme_v1}" "${SNAPME_SPLIT:=all}"
: "${SNAPME_CONCURRENCY:=48}"
PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
PY="$SNAPME_REPO/.pixi/envs/default/bin/python"

eval "$("$PY" -m snapme_eval.cli serve-cmd --model "$SNAPME_MODEL" --gpus "$SNAPME_GPUS" --port "$PORT")"
export SNAPME_ENGINE SNAPME_ENGINE_IMAGE SNAPME_GPUS
SIF="$SNAPME_CONTAINERS/$SNAPME_SIF"
# Join methods with "+" in SNAPME_SYSTEM ("marker:masked+marker:untold") to run them in turn on one engine.
# Each method writes to its own run folder.
IFS=+ read -ra SYSTEMS <<< "$SNAPME_SYSTEM"
run_dir() { "$PY" -c "from snapme_eval.systems import load; m = load('$1'); print(f'$SNAPME_ROOT/runs/$SNAPME_GT/{m.NAME}_{m.VERSION}/$SNAPME_MODEL')"; }
if [ "${#SYSTEMS[@]}" -gt 1 ]; then unset SNAPME_OUT; fi
OUT="${SNAPME_OUT:-$(run_dir "${SYSTEMS[0]}")}"
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname) partition $SLURM_JOB_PARTITION gpus $SNAPME_GPUS model $SNAPME_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Preflight: some nodes pass nvidia-smi but CUDA cannot initialize (seen on Atlas, 2026-11-08).
python3 -c "import ctypes,sys; sys.exit(ctypes.CDLL('libcuda.so.1').cuInit(0))" || {
  echo "CUDA init failed on $(hostname); resubmit with SNAPME_EXCLUDE including this node"; exit 5; }

# Extra Python packages a model's own code needs, installed once per engine and package set.
if [ -n "${SNAPME_EXTRA_PIP:-}" ]; then
  EXTRA="$SNAPME_ROOT/env/engine_extras/$SNAPME_ENGINE/$(printf %s "$SNAPME_EXTRA_PIP" | sha256sum | cut -c1-12)"
  if [ ! -f "$EXTRA/.done" ]; then
    apptainer exec --cleanenv --home "$TMPDIR/home" "$SIF" python3 -m pip install --quiet --no-deps --target "$EXTRA" $SNAPME_EXTRA_PIP && touch "$EXTRA/.done"
  fi
  export APPTAINERENV_PYTHONPATH="$EXTRA"
fi

# Per-model Python shim (sitecustomize.py) ahead of everything else on the engine's path.
if [ -n "${SNAPME_SHIM:-}" ]; then
  export APPTAINERENV_PYTHONPATH="$SNAPME_SHIM${APPTAINERENV_PYTHONPATH:+:$APPTAINERENV_PYTHONPATH}"
fi

# Engine: offline (weights are local), caches on node-local NVMe. --cleanenv drops Slurm's GPU list, so pass it.
APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_HF_HOME="$HF_HOME" APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR" APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME" \
APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" APPTAINERENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH" \
APPTAINERENV_FLASHINFER_WORKSPACE_BASE="$FLASHINFER_WORKSPACE_BASE" APPTAINERENV_TMPDIR="$TMPDIR" \
apptainer exec --nv --cleanenv --home "$TMPDIR/home" --env PYTHONNOUSERSITE=1 --bind "$SNAPME_ROOT,$TMPDIR" "$SIF" "${SNAPME_SERVE_CMD[@]}" \
  > "$OUT/engine_${SLURM_JOB_ID}.log" 2>&1 &
ENGINE_PID=$!
trap 'kill $ENGINE_PID 2>/dev/null || true; wait $ENGINE_PID 2>/dev/null || true' EXIT

# Wait for the engine (large models can take 20+ minutes to load and compile).
for i in $(seq 1 360); do
  if curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" >/dev/null; then echo "engine ready after $((i*5))s"; break; fi
  if ! kill -0 $ENGINE_PID 2>/dev/null; then echo "engine exited"; tail -n 60 "$OUT/engine_${SLURM_JOB_ID}.log"; exit 3; fi
  sleep 5
done
curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" >/dev/null || { echo "engine not ready"; tail -n 60 "$OUT/engine_${SLURM_JOB_ID}.log"; exit 4; }

for S in "${SYSTEMS[@]}"; do
  O="$OUT"
  if [ "${#SYSTEMS[@]}" -gt 1 ]; then O=$(run_dir "$S"); mkdir -p "$O"; echo "method $S -> $O"; fi
  "$PY" -m snapme_eval.cli run --system "$S" --model "$SNAPME_MODEL" --base-url "http://127.0.0.1:$PORT/v1" \
    --gt "$SNAPME_GT" --split "$SNAPME_SPLIT" --out "$O" --concurrency "$SNAPME_CONCURRENCY" ${SNAPME_LIMIT:+--limit $SNAPME_LIMIT}
  "$PY" -m snapme_eval.cli score --run "$O" --gt "$SNAPME_GT" || echo "scoring deferred"
done
date -u +%FT%TZ
