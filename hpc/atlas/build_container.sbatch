#!/bin/bash
# Convert a Docker image to an Apptainer SIF on a CPU node.
# Usage: sbatch hpc/atlas/build_container.sbatch <docker-uri-with-digest> <output.sif>
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=snapme_sif
#SBATCH --output=%x_%j.out
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SNAPME_REPO:-$here/../..}/hpc/atlas/env.sh"
module load apptainer/1.4.3

uri="$1"; out="$SNAPME_CONTAINERS/$2"
[ -s "$out" ] && { echo "exists: $out"; exit 0; }
echo "building $out from $uri on $(hostname), TMPDIR=$TMPDIR"
# Build on local NVMe, then move: squashfs writes are slow on the shared filesystem.
apptainer pull "$TMPDIR/$2" "$uri"
mv "$TMPDIR/$2" "$out.partial" && mv "$out.partial" "$out"
[ -n "${SNAPME_GROUP:-}" ] && chgrp "$SNAPME_GROUP" "$out"; chmod 0770 "$out"
sha256sum "$out" | tee "$out.sha256"
apptainer inspect "$out" | head -20 || true
date -u +%FT%TZ
