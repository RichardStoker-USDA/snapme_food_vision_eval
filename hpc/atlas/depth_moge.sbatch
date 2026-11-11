#!/bin/bash
# MoGe-2 point maps for every SNAPMe photo (geometry pipeline, stage 1). Configured for one L40S.
#   source hpc/atlas/env.sh; sbatch ${SNAPME_EXCLUDE:+--exclude=$SNAPME_EXCLUDE} --export=ALL,SNAPME_REPO=$PWD hpc/atlas/depth_moge.sbatch
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=snapme_depth_moge2
#SBATCH --output=%x_%j.out
set -euo pipefail
source "$SNAPME_REPO/hpc/atlas/env.sh"
module load apptainer/1.4.3
PY="$SNAPME_REPO/.pixi/envs/default/bin/python"
SIF="$SNAPME_CONTAINERS/vllm_v0.32.0.sif"
OUT="$SNAPME_ROOT/derived/moge2_vitl_normal"
mkdir -p "$OUT"

# Model code and checkpoint, both pinned in configs/aux_models/moge-2-vitl-normal.toml
eval "$("$PY" - <<'PYEOF'
import shlex, tomllib, pathlib, os
m = tomllib.loads(pathlib.Path(os.environ["SNAPME_REPO"], "configs/aux_models/moge-2-vitl-normal.toml").read_text())
org, name = m["hf_repo"].split("/")
ck = pathlib.Path(os.environ["HF_HUB_CACHE"], f"models--{org}--{name}", "snapshots", m["revision"], "model.pt")
print(f"CKPT={shlex.quote(str(ck))}")
print(f"PKGS={shlex.quote(' '.join([m['code'], *m['code_deps'], 'pillow-heif==1.8.1', 'scipy==1.18.1']))}")
PYEOF
)"
EXTRA="$SNAPME_ROOT/env/engine_extras/moge2/$(printf %s "$PKGS" | sha256sum | cut -c1-12)"
if [ ! -f "$EXTRA/.done" ]; then
  apptainer exec --cleanenv --home "$TMPDIR/home" --env PYTHONNOUSERSITE=1 --bind "$SNAPME_ROOT,$TMPDIR" "$SIF" \
    python3 -m pip install --quiet --no-deps --target "$EXTRA" $PKGS
  touch "$EXTRA/.done"
fi

"$PY" - > "$TMPDIR/photos.tsv" <<'PYEOF'
import pandas as pd
from snapme_eval.paths import gt_dir, data_dir
im = pd.read_parquet(gt_dir("snapme_v1") / "images.parquet")
rel = data_dir() / "snapme" / "snapme_db_09Dec2022"
for r in im.itertuples():
    print(f"{r.image_id}\t{rel / r.path}")
PYEOF
echo "photos: $(wc -l < "$TMPDIR/photos.tsv")"

APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" APPTAINERENV_PYTHONPATH="$EXTRA" \
apptainer exec --nv --cleanenv --home "$TMPDIR/home" --env PYTHONNOUSERSITE=1 --bind "$SNAPME_ROOT,$TMPDIR" "$SIF" \
  python3 "$SNAPME_REPO/src/snapme_eval/depth/moge_export.py" "$CKPT" "$TMPDIR/photos.tsv" "$OUT"
ls "$OUT" | wc -l
