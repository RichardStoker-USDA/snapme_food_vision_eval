#!/bin/bash
# EmbeddingGemma 2 vectors for FNDDS foods and SNAPMe photos (src/snapme_eval/embed.py), on one GPU in the
# SGLang image with sentence-transformers installed once into $SNAPME_ROOT/env/sentence_transformers.
#   sbatch -p gpu-l40s,gpu-a100 --gres=gpu:1 --export=ALL,SNAPME_REPO=$PWD hpc/atlas/embed.sbatch
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=snapme_embed
set -euo pipefail
source "$SNAPME_REPO/hpc/atlas/env.sh"
module load apptainer/1.4.3
umask 007
SIF="$SNAPME_CONTAINERS/sglang_v0.5.23.sif"
E="$SNAPME_ROOT/env/sentence_transformers"
if [ ! -f "$E/.done_v2" ]; then
  apptainer exec --cleanenv --home "$TMPDIR" "$SIF" python3 -m pip install --quiet --no-deps --target "$E" \
    "sentence-transformers==6.2.0" "scikit-learn==1.7.2" "threadpoolctl" "joblib" "pillow-heif==1.1.1"
  touch "$E/.done_v2"
fi
SNAP=$(ls -d "$HF_HUB_CACHE"/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e)
apptainer exec --nv --cleanenv --home "$TMPDIR" --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env PYTHONPATH="$E" --env HF_HUB_OFFLINE=1 --env PYTHONNOUSERSITE=1 --bind "$SNAPME_ROOT,$TMPDIR" "$SIF" \
  python3 "$SNAPME_REPO/src/snapme_eval/embed.py" --model-dir "$SNAP" --root "$SNAPME_ROOT"
