# Method implementations

A method, called a `pipeline` in result files, produces a food list for each photo. Each item can contain a
name, preparation, ingredients and grams. The shared output format allows the same scoring code to read
all methods. A full run is identified by method and version, such as `direct_v2`.

| Input and procedure | `--system` | Additional input |
|---|---|---|
| A model estimates foods and amounts from a before photo or package front | `direct` | none |
| The model also receives the after photo or ingredient label and estimates intake | `consumed` | companion photo, when available |
| A model receives the photo and several earlier food lists, then writes one list | `combine` | `SNAPME_COMBINE_MEMBERS`, or `submit.py --members a,b,c` |
| A model locates foods; estimated volumes accompany a second request for grams | `geometry` | MoGe-2 point maps from `hpc/atlas/depth_moge.sbatch` |
| A second model checks each listed food, selects its code and estimates a portion | `check` | `SNAPME_CHECK_SOURCE`, or `submit.py --source <pipeline>/<model>` |

Prompts are in `src/snapme_eval/prompts/`. Runs record a prompt hash. The `combine` prompt hash also includes
member IDs; the `check` hash includes its module text and source run. A comment edit to `check.py` therefore
changes its recorded hash even if the questions stay identical.

## Photo and list inputs

`direct` selects the prompt by photo type. The overhead-photo prompt is used for Nutrition5k.
`consumed` uses the first photo alone when the companion is absent, with the corresponding `direct` prompt.
For packages, the paired input is the front and ingredient label.

`combine` reads earlier `direct_v2` predictions. A member with no readable list for a photo is omitted for
that photo. The combining model also sees the photo. Member IDs are configurable; `combine_v2_top3` is a
run-folder name for a three-member comparison, not a requirement of the method.

`geometry` uses model-drawn rectangles and a MoGe-2 estimate of three-dimensional positions. The calculation
fits a table plane, estimates a local base near each rectangle's edge and integrates height above that base.
A usable checkerboard estimate rescales lengths and volumes; otherwise the MoGe-2 scale is retained.
Package fronts, missing point maps and unsuccessful geometry steps use `direct` requests. These are estimated
volumes, not independent physical measurements.

## Checking foods and portions

`check` reads items from an earlier run. If `SNAPME_CHECK_SOURCE` is unset or empty, its source is
`combine_v2_top3/gemma-4-31b-it`.
Each item's name supplies up to 15 FNDDS 2017-2018 candidates from the Qwen3-Embedding-8B name-search cache.
The photo and item are passed to a decision model, which returns probabilities for specified answers.

1. The presence question asks whether the food is visible. Items with probability below 0.5 are omitted from
   the final list, but retained in the run's `dropped` field.
2. The food question offers the FNDDS candidates and "none of these foods". If "none" receives the highest
   probability, the nearest retrieved candidate is retained. If there are no candidates, the earlier name and
   grams are retained without a selected food code.
3. The portion question uses multiples of up to five FNDDS household measures, selected near the earlier
   estimate of grams. The final amount is the probability-weighted mean of the options. Fewer than two options
   leaves the earlier amount unchanged.

Presence and food questions are answered before the portion question. Each item retains its original name
and grams (`name_reader`, `grams_reader`), candidate codes and returned probabilities where available.
These probabilities have not been validated as calibrated confidence estimates on SNAPMe.

| Model | Developer | Serving interface | `decision_api` |
|---|---|---|---|
| Clef, Clef Flash | Cloudflare | SGLang reads `joint_head_config.json` | `systemone` |
| GLiDE 27B | Fastino | SGLang reads `decision_config.json` | `systemone` |
| JEV 27B VL | AutoTrust | vLLM serves the supplied decision adapter; the client applies its bias and temperature | `jev_vllm` |
| Intern-Decision 4B | Shanghai AI Laboratory | `systemone_server.py` calls the supplied `inference.py` | `systemone` |

`src/snapme_eval/decision.py` provides the shared `ask(state, image, questions)` interface.

```bash
python hpc/atlas/submit.py --system check --source combine_v2_top3/gemma-4-31b-it clef glide-27b jev-27b-vl
```

Two variants omit a step's effect using existing predictions, without further model requests:

```bash
snapme-eval ablate --run $SNAPME_ROOT/runs/snapme_v1/check_v2/clef --variant keep_all
snapme-eval ablate --run $SNAPME_ROOT/runs/snapme_v1/check_v2/clef --variant reader_grams
```

`keep_all` restores foods omitted by the presence threshold. `reader_grams` restores the earlier gram
estimates while retaining the selected food codes. Both create separate run folders for scoring.

## Adding a method

A method module defines `NAME`, `VERSION`, `prompt_hash()` and:

```python
async def predict(row: dict, chat: Chat, release_dir: pathlib.Path) -> dict:
    ...
    return {"items": [{"name": ..., "preparation": ..., "ingredients": [...], "grams": ...}],
            "parse_error": None, ...}
```

The runner passes only `image_id`, `filename`, `photo_type`, `path`, `companion_path` and `companion_exists`.
ASA24 values and codes are excluded by `runner.PIPELINE_FIELDS`. `chat` sends requests to the registered model.
An item with `fndds_code` uses that code during scoring; other items use the shared name-to-code cache.
Additional returned fields are retained in `predictions.jsonl`.

`--system direct` loads `snapme_eval.systems.direct`. A full import path, such as
`--system snapme_eval.systems.direct`, is also accepted. Other modules must be importable in the harness
environment, for example through an installed package or `PYTHONPATH`. The run metadata and Atlas output
folder use the module's `NAME` and `VERSION`, not its import path.

The site-results collector still looks up methods under `snapme_eval.systems`. For collection, `NAME` must
identify a module there and contain no underscore: the collector splits the run identifier at its first
underscore to find the module and version. Loading an external module does not by itself add its results to
the site. The supplied Compose command also uses the short method name to construct its output path.
