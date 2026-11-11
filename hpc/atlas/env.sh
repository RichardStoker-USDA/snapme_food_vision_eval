# Source this in every Atlas job and shell that touches the benchmark.
# Keeps every cache, model and temporary file off /home (30 GB quota) and on
# /90daydata (shared, purged by access time) or node-local NVMe ($TMPDIR).

# Site settings (data root, Slurm account, shared group, nodes to avoid) live in site.env next to this
# file; it is not tracked. Copy site.env.example to start.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_here/site.env" ] && source "$_here/site.env"
: "${SNAPME_ROOT:?set SNAPME_ROOT in hpc/atlas/site.env}"
export SNAPME_ROOT

# Group-writable files so lab members can read and extend runs.
umask 007

# Node-local scratch exists inside Slurm jobs ($TMPDIR=/local/scratch/$USER/$SLURM_JOB_ID).
# Outside a job fall back to a per-user folder under the 90-day space.
if [ -z "${TMPDIR:-}" ] || [ ! -d "${TMPDIR}" ]; then
  export TMPDIR="$SNAPME_ROOT/tmp/$USER"
  mkdir -p "$TMPDIR"
fi

# Model weights: one shared Hugging Face cache for the whole lab project.
export HF_HOME="$SNAPME_ROOT/models/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# Containers: images on 90-day space, build layers on local NVMe.
export APPTAINER_CACHEDIR="$SNAPME_ROOT/containers/cache"
export APPTAINER_TMPDIR="$TMPDIR/apptainer"
mkdir -p "$APPTAINER_TMPDIR" "$TMPDIR/home"   # separate container home avoids host ~/.local packages

# Compile and kernel caches. Use node-local storage inside jobs.
export XDG_CACHE_HOME="$TMPDIR/xdg"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export CUDA_CACHE_PATH="$TMPDIR/nv"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer"
export PIP_CACHE_DIR="$TMPDIR/pip"

# pixi for the harness environment, kept under the project root.
export PIXI_HOME="$SNAPME_ROOT/env/pixi_home"
export PIXI_CACHE_DIR="$SNAPME_ROOT/env/pixi_cache"
export RATTLER_CACHE_DIR="$PIXI_CACHE_DIR"
export PATH="$PIXI_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1

export SNAPME_CONTAINERS="$SNAPME_ROOT/containers"

# WordNet for the token F1 lemmatizer
export NLTK_DATA="$SNAPME_ROOT/env/nltk_data"

# Atlas sends HTTP through a proxy whose no_proxy lists 127.0.0.0/8, which curl and httpx do not read as a
# range, so local engine calls went to the proxy (503). Name the loopback hosts explicitly.
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
