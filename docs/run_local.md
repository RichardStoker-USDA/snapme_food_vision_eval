# Running on one machine with NVIDIA GPUs

The local workflow runs a registered food-list model on the SNAPMe reference photos. It requires Docker,
the NVIDIA Container Toolkit, storage for the source and derived data, and space for model weights. Model weight file
sizes are recorded as `weights_gb` in `configs/models/`; request-time GPU memory use is larger than model weight
storage alone.

The harness sends photos and scores replies. The separate inference engine serves the model. Its image is
fixed by digest, and its command-line options come from the model configuration. The harness tag `latest`
changes over time; a commit tag or a local build identifies the code used for reproduction.

## 1. Repository and data directory

```bash
git clone https://github.com/RichardStoker-USDA/snapme_food_vision_eval.git
cd snapme_food_vision_eval
export SNAPME_DATA=/path/with/space          # data, model weights and runs go here
mkdir -p "$SNAPME_DATA"
H="docker run --rm --user $(id -u):$(id -g) -v $SNAPME_DATA:/data -v $PWD/containers:/out \
   ghcr.io/richardstoker-usda/snapme_food_vision_eval:latest"
```

`$H` runs one `snapme-eval` command in the harness container. A local image can be built with:
`docker build -f containers/harness.Dockerfile -t snapme_eval .` and use `snapme_eval` in place of the
ghcr.io name.

## 2. Data and reference tables

```bash
$H fetch                                   # SNAPMe DB v1.0 and FNDDS 2017-2018, checked against pinned hashes
$H fndds --src /data/data/fndds            # workbooks to parquet
$H gt --version snapme_v1                  # ground truth tables in /data/gt/snapme_v1
```

`fetch` checks the SNAPMe archive, FNDDS workbooks and 2023 baseline files against recorded hashes.
A mismatch stops the download because the file differs from the recorded input; either changed content or an
incomplete transfer can cause this. WordNet is downloaded separately for reducing words to their base forms.

## 3. Model weights

Model ids are the file names in `configs/models/`. Each file pins a Hugging Face commit and records size,
license and GPU allocations. The Results page lists how each one scored.

```bash
$H download --model qwen3-5-9b             # about 20 GB; gated models need HF_TOKEN (see below)
```

Gated model downloads require access approval on Hugging Face and a token supplied to the container:
`docker run ... -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" ...`.

## 4. Model serving and photo requests

```bash
$H serve-cmd --model qwen3-5-9b --gpus 1 --format compose --out /out   # writes containers/.env and engine.sh
docker compose -f containers/compose.yaml up --abort-on-container-exit
```

The engine starts, the harness waits until it answers, sends every photo, and writes
`$SNAPME_DATA/runs/snapme_v1/direct_v2/qwen3-5-9b/`:

| File | Holds |
|---|---|
| `predictions.jsonl` | one line per photo: the model's raw reply, parsed items, timing and token counts |
| `run_*.json` | model, revision, engine image, sampling, code commit, start and end time |
| `scores.json` | available measures; identification and line-match means include participant bootstrap intervals |
| `per_photo.parquet` | per-photo predicted and ASA24 values |

`SNAPME_SYSTEM` selects the method in the Compose command. `consumed` uses companion photos; `geometry`
requires precomputed point maps. `combine` and `check` also need source runs. The Compose file forwards
`SNAPME_COMBINE_MEMBERS` and `SNAPME_CHECK_SOURCE` to the harness container; set them in the shell or Compose
environment file before starting the run. An empty `SNAPME_CHECK_SOURCE` uses the method's default source.
Inputs and full-module-path execution are described in [pipelines.md](pipelines.md).

`SNAPME_FOLDS` restricts SNAPMe requests to participant groups numbered 0 through 4, assigned by the fixed
hash in `src/snapme_eval/folds.py`. For example, `SNAPME_FOLDS=0,2` selects two groups; all photos of a
participant belong to the same group. The filter is applied after `--split` and before `--limit`, and the
run record stores the selection. Leave it unset for all groups. This setting requires `subject_id` in the
reference table and does not apply to Nutrition5k's dish table.

For a separate harness invocation, add `-e SNAPME_FOLDS=0,2` to the Docker command that defines `$H`;
the supplied Compose file does not forward this variable. Use a separate output directory for a selected
subset, since existing predictions in a reused directory remain there. The public SNAPMe report includes
only runs covering the full reference photo set.

Resubmitting with the same output folder skips completed photos and retries request failures. An unreadable
model reply is retained and scored as an empty list.

The model's `[serve.a100]` and `[serve.l40s]` blocks record GPU allocations. Allocation also depends on the
engine and request limits; model weight file size alone is insufficient.

## 5. Food coding and scoring

Token F1 compares word sets and can be scored without food coding. Food-code and What We Eat in America
(WWEIA) category F1, matched-line shares, preparation and gram errors require the name-to-code cache.
Qwen3-Embedding-8B retrieves FNDDS candidates; Qwen3.8-27B selects a code. The example below uses separate
GPUs for these two servers; each GPU must have sufficient memory for its assigned model.

```bash
$H download --model qwen3-embedding-8b --model qwen3-8-27b
# two engines on two GPUs; paths are the snapshot folders under $SNAPME_DATA/models/hf/hub
docker run -d --gpus '"device=0"' -p 8001:8001 -v $SNAPME_DATA:/data vllm/vllm-openai:v0.32.0 \
  /data/models/hf/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/<revision> \
  --served-model-name qwen3-embedding-8b --runner pooling --port 8001
docker run -d --gpus '"device=1"' -p 8002:8002 -v $SNAPME_DATA:/data vllm/vllm-openai:v0.32.0 \
  /data/models/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/<revision> \
  --served-model-name qwen3-8-27b --max-model-len 16384 --port 8002
$H map --embed-url http://host.docker.internal:8001/v1 --judge-url http://host.docker.internal:8002/v1
$H score --run /data/runs/snapme_v1/direct_v2/qwen3-5-9b
```

Revisions are in `configs/aux_models/qwen3-embedding-8b.toml` and `configs/models/qwen3-8-27b.toml`. On Linux
add `--add-host host.docker.internal:host-gateway` to `$H`, or use the machine's address. The mapping cache (`data/mapping/name_to_fndds_v1.parquet`) is shared by all runs, so names mapped once are
not mapped again.

## 6. Nutrition5k comparison

The additional Nutrition5k comparison uses overhead color photos and weighed ingredients from cafeteria
plates. It scores the 507 dishes in the authors' overhead test list. The download is about 1.7 GB: metadata,
split lists, overhead color images and depth images. The published runs use the color photos. Sources and
the CC BY 4.0 license are recorded in [Datasets for food identification and amount estimation](ground_truth_datasets.md).

After the FNDDS setup above, build the Nutrition5k reference and send its test photos to a registered model
server. The example uses the endpoint described in the next section; `direct` selects the overhead prompt
from the reference's photo type.

```bash
$H nutrition5k fetch
$H nutrition5k build                       # reference tables in /data/gt/nutrition5k_v1
$H run --system direct --model qwen3-5-9b --base-url http://gpu-host:8000/v1 \
   --gt nutrition5k_v1 --split test --out /data/runs/nutrition5k_v1/direct_v2/qwen3-5-9b
$H map --embed-url http://host.docker.internal:8001/v1 --judge-url http://host.docker.internal:8002/v1
$H score --run /data/runs/nutrition5k_v1/direct_v2/qwen3-5-9b --gt nutrition5k_v1
```

The food-coding step uses the servers and shared name-to-code cache from section 5, including the reference
ingredient names. To keep collected results on the host, add
`-v $PWD/results:/app/results -v $PWD/site/public:/app/site/public` to the Docker command that defines `$H`,
then collect the scored runs:

```bash
$H report
```

On Atlas, pass `--gt nutrition5k_v1 --split test` to `hpc/atlas/submit.py`, with `--system direct` and the
model ID, to select this reference and test split.

## A separate model server

The harness can send image-and-text requests to an OpenAI-compatible chat endpoint. The endpoint must
support the request fields used by the registered model.

```bash
$H run --system direct --model qwen3-5-9b --base-url http://gpu-host:8000/v1 \
   --out /data/runs/snapme_v1/direct_v2/qwen3-5-9b
$H score --run /data/runs/snapme_v1/direct_v2/qwen3-5-9b
```

For a model that is not in the registry, write its file with `tools/add_model.py` (see
[add_model.md](add_model.md)) into `containers/models_extra/`; the harness reads that folder too.

## Without Docker

```bash
pixi install                               # Python environment from pixi.lock
export SNAPME_ROOT=$SNAPME_DATA
pixi run snapme-eval fetch                 # and the other commands above, without $H
```
