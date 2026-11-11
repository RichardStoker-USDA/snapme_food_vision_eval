# Registering a model

Each model has a TOML file in `configs/models/`. It identifies the Hugging Face repository and fixed revision,
license, parameter count, weight format, engine and GPU allocations. Files in folders listed by
`SNAPME_MODEL_DIRS` override registered files with the same ID.

## Creating the entry

```bash
pixi run python tools/add_model.py google/gemma-4-31B-it \
  --developer Google --country "United States" --architecture "dense transformer with vision encoder"
```

The tool reads repository metadata and writes an entry at its current revision. It estimates a GPU allocation
from weight size, allowing weights to occupy at most 72% of listed GPU memory and keeping the model on one
node. This estimate does not verify engine support or memory needed during requests.

| Option | Meaning |
|---|---|
| `--id` | model ID; defaults to a normalized repository name |
| `--active-b` | active language-model parameters per text token, in billions; supply a verified count |
| `--engine` | key in `configs/engines/engines.toml`; defaults to `vllm_0_32_0` |
| `--reasoning` | `none`, `optional` or `always`; optional reasoning is disabled, and always-on reasoning receives a larger output limit |
| `--role` | `generalist` writes food lists; `decision` answers yes/no and multiple-choice questions; `perception` locates foods |
| `--precision-of` | ID of the reference version when registering a lower-precision version of the same model |
| `--gpu-types` | GPU types for proposed allocations, such as `h100,l40s` |
| `--out` | output folder, including `containers/models_extra/` for local additions |

Weights are the model's numerical parameters. BF16 stores values in 16-bit floating-point format, FP8 in
8-bit floating-point format, INT4 in 4-bit integers, and NVFP4 in a 4-bit floating-point format with scaling.
A smaller weight file does not by itself establish whether a model can run on a given GPU.

## Parameter counts

`params_total_b` and `params_active_b` are counts in billions. A mixture-of-experts model selects groups of
parameters, called experts, for each text token (a word or part of a word). Use the developer's stated active
count when available. Otherwise, calculate it from the downloaded weight files:

```bash
pixi run python tools/param_count.py gemma-4-31b-it
```

The command requires `SNAPME_ROOT` and the downloaded weights. It reads tensor shapes from safetensors headers
and routing settings from `config.json` or `params.json`, then prints the counts and any reason the active count
could not be determined. It does not edit the registry.

The calculated active count includes non-routed language-model parameters and shared experts in full. Each
bank of routed experts contributes its parameter count multiplied by experts selected per token divided by
experts in that bank. Vision and audio encoders, multi-token prediction layers and per-layer lookup tables
such as n-gram embeddings count toward the total but not this active figure. Quantization scales are excluded
from both counts. Where two 4-bit weights are packed into one byte, the tool counts both parameters.

`add_model.py` takes its initial total from Hugging Face metadata, which can undercount packed 4-bit weights.
Check that total against the downloaded files. For a mixture-of-experts model, the tool leaves the active
field unset unless `--active-b` is supplied. Review the count output before updating the two registry fields.
If no developer-stated active count is available and the files cannot determine one, omit `params_active_b`.
The counter leaves it unavailable when the experts-per-token setting is missing, expert banks are shared
across layers, or expert counts differ by layer. Do not substitute the total for a missing active count.

## Reviewing the entry

The model card and the chosen engine's requirements determine the serving options. `[engine_args]` entries
become command-line flags: `max_num_seqs = 8` becomes `--max-num-seqs 8`. The `{snapshot}` placeholder expands
to the local weight directory. Required tokenizer formats and weight-file selections belong in the model entry.

Hardware-specific examples are recorded in existing configurations. For example, some A100 allocations use
vLLM's Marlin kernels for lower-precision weights. These settings are model- and engine-specific, rather than
a general guarantee of support for a numerical format.

`extra_pip` adds packages when the engine starts. `python_shim` selects a compatibility module under
`hpc/atlas/shims/`. Packages and model code should be fixed to reviewed versions. A `status_note` explains why
a registered model has no result in the public tables.

## Verifying a run

```bash
pixi run snapme-eval download --model gemma-4-31b-it
python hpc/atlas/submit.py --system direct --split dev --limit 8 --time 00:45:00 --out-suffix _smoke gemma-4-31b-it
```

This short run checks whether the engine starts, receives images and returns readable item lists. Inspection
of `predictions.jsonl` and the engine log is needed before the full run. A full run omits `--split`, `--limit`
and `--out-suffix`, with `SNAPME_FOLDS` unset. Docker execution is described in [run_local.md](run_local.md).

The main table reports developer-published models that write food lists. Lower-precision versions are linked
to reference versions through `precision_of`. Decision models are used by the `check` method and are listed
with that method's results.
