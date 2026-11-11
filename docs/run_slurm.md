# Running on a Slurm cluster

The jobs in `hpc/atlas/` implement the benchmark workflow for SCINet Atlas (USDA ARS). They use Slurm,
Apptainer and shared storage. Site settings cover accounts, paths and GPU partitions. The scripts also contain
Atlas module versions and default Slurm directives that need review on another cluster.

Model runs used GPU nodes of Atlas, the SCINet high-performance computing cluster hosted at Mississippi
State University under an agreement with USDA ARS, with NVIDIA A100 80 GB and L40S 48 GB graphics processors.

Each model runs in one job. An engine serves the model on the allocated GPUs; the harness sends photos and
scores the replies. Model weights, containers, caches and results are stored under the configured data root or
`$TMPDIR`, including a separate home directory for container execution.

## 1. Settings

```bash
git clone https://github.com/RichardStoker-USDA/snapme_food_vision_eval.git repo
cd repo
cp hpc/atlas/site.env.example hpc/atlas/site.env      # not tracked; edit it
```

| Setting | Meaning | Atlas value |
|---|---|---|
| `SNAPME_ROOT` | data, model weights, images and runs; large and shared, not home | `/90daydata/<project>/snapme_food_vision_eval` |
| `SBATCH_ACCOUNT` | Slurm account, read by `sbatch` | the project account |
| `SNAPME_GROUP` | group that owns shared files (optional) | `proj-<project>` |
| `SNAPME_PARTITIONS` | GPU type to partition, smaller GPUs first | `l40s=gpu-l40s,a100=gpu-a100` |
| `SNAPME_EXCLUDE` | nodes to avoid | a node where CUDA failed to start |

The GPU type names are the Slurm GRES names (`--gres=gpu:<type>:N`) and the `[serve.<type>]` keys in the model
files. On a cluster with H100s, add `h100=<partition>` and give models a `[serve.h100]` block
(`tools/add_model.py --gpu-types h100` proposes one).

`hpc/atlas/env.sh` sets the configured caches (Hugging Face, pip, Triton, vLLM, Apptainer) under `SNAPME_ROOT` or
`$TMPDIR`. Source it in each shell and job.

## 2. Environment, images, data

Run these on compute nodes (`srun` or `sbatch`), not on a login node.

```bash
source hpc/atlas/env.sh
srun -p <cpu partition> -c 8 --mem 32G -t 1:00:00 pixi install        # harness environment in .pixi/
# engine images: Docker images pinned by digest in configs/engines/engines.toml, converted to SIF
read -r engine_uri engine_sif < <(pixi run python -c \
  'import tomllib; e=tomllib.load(open("configs/engines/engines.toml","rb"))["vllm_0_32_0"]; print(e["image"], e["sif"])')
sbatch hpc/atlas/build_container.sbatch "$engine_uri" "$engine_sif"
srun -p <cpu partition> -c 4 --mem 16G -t 1:00:00 pixi run snapme-eval fetch
srun -p <cpu partition> -c 4 --mem 16G -t 0:30:00 bash -c \
  "pixi run snapme-eval fndds --src $SNAPME_ROOT/data/fndds && pixi run snapme-eval gt --version snapme_v1"
```

Build one SIF per engine the chosen models use (the `engine` field of each model file).

## 3. Model weights

Downloads go on a data-transfer node if the cluster has one (Atlas: `atlas-dtn`), not through Slurm:

```bash
source hpc/atlas/env.sh
pixi run snapme-eval download --model gemma-4-31b-it --model qwen3-5-9b
```

Each download writes a receipt to `$SNAPME_ROOT/models/receipts/`; `submit.py` skips models without one. On a
filesystem that purges by access time (Atlas `/90daydata`), new downloads refresh access times. Downloads skipped because a matching receipt exists do not refresh them.
Refresh access times before long gaps (`find $SNAPME_ROOT -type f -print0 | xargs -0 touch -a -c`).

## 4. Submit

```bash
source hpc/atlas/env.sh
P=.pixi/envs/default/bin/python
$P hpc/atlas/submit.py --system direct --split dev --limit 8 --time 00:45:00 --out-suffix _smoke gemma-4-31b-it
$P hpc/atlas/submit.py --system direct gemma-4-31b-it qwen3-5-9b      # full runs
$P hpc/atlas/submit.py --system direct --all                          # every food-record model
$P hpc/atlas/submit.py --system direct --all --retry                  # finish runs with photos that failed
```

The eight-photo smoke run checks engine startup and readable replies. Image use also requires inspecting
the replies against the photos. `submit.py` chooses a partition from configured GPU allocations and reported
free capacity, then appends submitted jobs to `$SNAPME_ROOT/control/jobs.tsv`. Only jobs in this project's
ledger belong to this workflow.

Job logs are in `$SNAPME_ROOT/logs/`; each run folder also keeps the engine log (`engine_*.log`).

The current `minimax-m3-1-flash` entry requests eight A100 GPUs, a whole Atlas node. Its configuration records
that loading on four A100s ran out of memory while repacking the model weights. Model weight file size alone does not
describe that loading requirement.

For a run limited to selected participant groups, export `SNAPME_FOLDS` before submission and use a separate
`--out-suffix`. The job inherits this variable. Group assignments and the order of filtering are described
in [run_local.md](run_local.md); leave the variable unset for a full run.

## 5. Food-list grading, food coding and results

`hpc/atlas/align_and_score.sbatch` serves gpt-oss-120b to grade food lists against ASA24 descriptions, then
rescores current runs. It uses one grader server per allocated GPU. `snapme-eval align --url URL` runs the
grading step against an existing server; repeat `--url` for several servers. `--run DIR` selects a run folder
and can be repeated. Without `--run`, `--pipeline NAME` adds a method folder to the current set and can also
be repeated. In the Slurm jobs, put additional method folder names in the space-separated
`SNAPME_ALIGN_PIPELINES` variable.

To divide grading across jobs, pass `--shard I/N`, where I starts at 0 and N is the number of non-overlapping
shares. Set `SNAPME_ALIGN_ARGS="--shard I/N"` for each job. A job with `SNAPME_ALIGN_ARGS` set grades its share
or pilot and skips rescoring. After every share finishes successfully, submit `hpc/atlas/score_all.sbatch`
on a CPU partition with an `afterok` dependency on all grading jobs. It rescores current runs and any extra
methods named in `SNAPME_ALIGN_PIPELINES`. Keep that variable the same for grading and rescoring.

`snapme-eval grader-sheet --out DIR` writes a sheet for a dietitian to grade, its answer key and the grading
rules. `snapme-eval grader-agreement --sheet FILE --key FILE` compares the completed sheet with the grader.
These commands prepare and assess a human review; they do not establish that one has been completed.

Food coding supplies the supplemental measures. Run it and collect the results with:

```bash
sbatch -p gpu-a100 --gres=gpu:a100:2 --cpus-per-task=16 --mem=200G \
  --export=ALL,SNAPME_REPO=$PWD hpc/atlas/map_and_score.sbatch     # maps new names, rescores every run
srun -p <cpu partition> -c 8 --mem 32G -t 0:30:00 pixi run snapme-eval report    # results/ and site data
srun -p <cpu partition> -c 4 --mem 16G -t 0:30:00 pixi run snapme-eval release --out $SNAPME_ROOT/releases/$(date +%F)
```

The mapping job uses two A100s in this example; its L40S example requests three GPUs.
`report` rewrites `results/snapme_v1/` and the site's data files; commit them to update the published site.

## Other methods

`--system consumed`, `combine` (with `--members a,b,c`), `geometry` (after `hpc/atlas/depth_moge.sbatch`) and
`check` (with `--source <pipeline>/<model>`): see [pipelines.md](pipelines.md).

`snapme-eval marker-table --run PIPELINE/MODEL --run PIPELINE/MODEL` reads the geometry runs' scale-card boxes,
checks them against the image pixels and writes the card-location table. Repeat `--run` for each geometry
run; `--out FILE` selects the output path. The scale-card methods use that table to select photos and paint
over the card or a same-size control patch.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| health check gets HTTP 503 | the HTTP proxy caught loopback calls; `env.sh` lists `127.0.0.1,localhost` in `no_proxy` |
| `cuInit` fails on one node | CUDA initialization failed; exclude the affected node with `SNAPME_EXCLUDE` while the cause is investigated |
| packages from `~/.local` load inside the container | `PYTHONNOUSERSITE=1` and `--home $TMPDIR/home` (set by the jobs) |
| FP8 model fails or returns unreadable text on A100 | review the model's engine settings; registered A100 examples use Marlin where supported |
| NVFP4 model on A100 or L40S | review the registered Marlin settings; support depends on model, engine and model weight format |
| `module` not found over plain ssh | use `bash -lc` |
| `max_num_seqs (256) exceeds available Mamba cache blocks` | hybrid linear-attention models on a small GPU; set `max_num_seqs` lower in `[engine_args]` |
| `in features size is not multiple of 16` with FP4 model weights | tensor parallel split a layer too narrow for the Marlin kernel; use `pipeline_parallel_size`, or `data_parallel_size` with `enable_expert_parallel` for MoE models |
