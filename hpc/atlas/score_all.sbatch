#!/bin/bash
# Rescore current runs and methods in SNAPME_ALIGN_PIPELINES concurrently on a CPU node.
# Submit after all jobs finish grading their shares:
#   sbatch -p atlas --dependency=afterok:A:B --export=ALL,SNAPME_REPO=... hpc/atlas/score_all.sbatch
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
set -euo pipefail
source "$SNAPME_REPO/hpc/atlas/env.sh"
cd "$SNAPME_REPO"
PY="$SNAPME_REPO/.pixi/envs/default/bin/python"
"$PY" -c "
import os
from snapme_eval.paths import runs_dir
from snapme_eval.score.report import current
also = set(os.environ.get('SNAPME_ALIGN_PIPELINES', '').split())
for p in sorted(runs_dir().glob('*/*/*/predictions.jsonl')):
    n = p.parent.parent.name
    if not n.startswith('.') and (current(n) or n in also): print(p.parent)" > "$TMPDIR/runs.txt"
echo "$(wc -l < "$TMPDIR/runs.txt") runs $(date -u +%FT%TZ)"
OMP_NUM_THREADS=2 xargs -P 8 -I{} sh -c '"$0" -m snapme_eval.cli score --run "$1" > /dev/null 2>&1 && echo "scored $1" || echo "score failed: $1"' "$PY" {} < "$TMPDIR/runs.txt"
date -u +%FT%TZ
