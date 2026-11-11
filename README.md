# SNAPMe Food Vision Eval

This benchmark compares food lists estimated from SNAPMe photos with participants' ASA24 food records.
The reference contains 1,675 before photos or package fronts from 95 adults. Vision-language models receive
a photo and a written request for foods, drinks, preparation, ingredients of mixed dishes and grams.
The models are open-weight: their numerical parameters are available to download under their developers' licenses.

[Results, methods and photo explorer](https://richardstoker-usda.github.io/snapme_food_vision_eval/)

## Measures

The main measures use a text model, gpt-oss-120b, to pair each food list with the ASA24 descriptions. It grades
pairs as the same food, the same kind of food, or parts of a mixed dish, without seeing photos or food codes.
F1 combines the share of recorded lines found and the share of listed foods in the record. Parts count at both
grading levels. Corrections count entries to add, remove or rename. Amount measures use same-food pairs with
at least 10 g recorded and a positive estimate, separating foods and drinks. The grader has not yet been
compared with a dietitian's grades.

For supplemental measures, predicted food names are coded to FNDDS 2017-2018 foods, then compared with the ASA24
lines linked to each photo. Food identification is assessed at the exact food-code level and at broader What
We Eat in America (WWEIA) categories and FNDDS groups:

- Food identification F1 combines precision (the share of predicted items matching a recorded line) with
  recall (the share of recorded dishes with at least one matching item). Each photo has an F1 from 0 to 1;
  results report the mean across photos.
- ASA24 lines matched is the share of recorded lines paired one-to-one with predictions, averaged across photos.
  Matching uses exact food code first, then WWEIA category.
- Preparation agreement is the share of eligible matched lines with at least one preparation word in common.
- Item-gram error is the median absolute difference between estimated and recorded grams, divided by recorded
  grams. It uses matched pairs with a positive estimate and at least 10 g recorded, and is reported as a percentage.
  The share of those eligible matched foods with an amount is reported alongside the error.
- Token F1 compares sets of words. The 2023 comparison uses the paper's published reference sets on 1,463
  photos, including ingredient lists for mixed dishes. Inverse Cooking and Im2Recipe are rescored from their
  published outputs.

Zero or a missing value counts as no amount given. Photo totals sum positive estimates and are compared with
the full recorded total only when at least one food has an amount; percentage errors require at least 10 g
recorded. The share of photos with an amount is also reported. A total can omit foods with no amount, and a
small error can describe only a small share of a run. Nutrition5k amount measures compare summed positive
estimates with the full weighed dish mass and report the share of dishes with an amount.

The SNAPMe 95% bootstrap intervals resample participants with replacement, keeping each person's photos together.
Nutrition5k compares ingredient sets and resamples dishes.
They describe sampling uncertainty, not uncertainty in food records or repeated model replies. Tables include
completed runs available so far; they do not establish a final model ranking.

ASA24 grams describe food eaten. Before photos show food served, so leftovers can contribute to differences.
Calories and nutrients are neither estimated nor scored. Food identity and amount can be linked separately to
FNDDS or FoodData Central nutrient values.

## Methods compared

The benchmark evaluates estimates from a before photo or package front, then methods that add a second photo,
combine several models' lists, supply estimated food volumes, or ask a second model to check each food and
portion. [Method implementations](docs/pipelines.md) describe the inputs and fallback rules.

Developer-published versions with fewer bits per model weight are compared with their reference versions:
FP8 uses 8-bit floating-point values, INT4 uses 4-bit integers, and NVFP4 uses a 4-bit floating-point format with
scaling.

For SNAPMe, the benchmark compares food lists estimated by models from participants' phone photos of daily
meals with the ASA24 lines those participants entered. The additional Nutrition5k evaluation compares food
lists estimated by models from overhead cafeteria photos with the dishes' weighed ingredients. A camera on
a fixed rig looked down from directly above each plate. The scores use different references and matching rules and are not directly comparable.

## Reproduction

The harness sends requests to a model server and scores the replies. The following Docker example downloads
the data and a registered model, then starts the model server and harness. Food-code and matched-item scores
require the name-to-code cache described in the local guide.

Model runs used GPU nodes of Atlas, the SCINet high-performance computing cluster hosted at Mississippi
State University under an agreement with USDA ARS, with NVIDIA A100 80 GB and L40S 48 GB graphics processors.

```bash
git clone https://github.com/RichardStoker-USDA/snapme_food_vision_eval.git && cd snapme_food_vision_eval
docker pull ghcr.io/richardstoker-usda/snapme_food_vision_eval:latest   # or: docker build -f containers/harness.Dockerfile .
export SNAPME_DATA=/path/with/space
H="docker run --rm --user $(id -u):$(id -g) -v $SNAPME_DATA:/data -v $PWD/containers:/out \
   ghcr.io/richardstoker-usda/snapme_food_vision_eval:latest"
$H fetch && $H fndds --src /data/data/fndds && $H gt --version snapme_v1
$H download --model qwen3-5-9b
$H serve-cmd --model qwen3-5-9b --gpus 1 --format compose --out /out
docker compose -f containers/compose.yaml up --abort-on-container-exit
```

`snapme-eval align --url URL` grades current runs. Repeat `--url` to use several grader servers, `--run DIR`
to select run folders, or `--pipeline NAME` to include additional method folders. `--shard I/N` divides the
lists into N non-overlapping shares numbered from 0; run each share once when splitting grading across jobs.
On Atlas, `hpc/atlas/align_and_score.sbatch` serves the grader and grades the lists. When
`SNAPME_ALIGN_ARGS` selects a share or pilot, it skips rescoring. After all grading jobs finish, run
`hpc/atlas/score_all.sbatch` on a CPU node to rescore the runs, then `snapme-eval report` to collect results.

`snapme-eval grader-sheet --out DIR` prepares a sheet for a dietitian to grade. After the sheet is filled in,
`snapme-eval grader-agreement --sheet FILE --key FILE` compares those grades with the model grader's grades.
`snapme-eval marker-table --run PIPELINE/MODEL --run PIPELINE/MODEL` checks scale-card locations from geometry
runs against the image pixels and writes the table used by the scale-card comparisons.

- [Local run guide](docs/run_local.md): Docker, data, model serving, food coding and scoring.
- [Slurm guide](docs/run_slurm.md): Apptainer images and model jobs, as used on SCINet Atlas.
- [Kubernetes guide](docs/kubernetes.md): generating a model-serving manifest and reviewing its settings.
- [Model registration](docs/add_model.md): revisions, precision, engine options and GPU allocations.
- [Dataset comparison](docs/ground_truth_datasets.md): reported intake, weighed foods and three-dimensional scans.

## Repository contents

| Path | Contents |
|---|---|
| `src/snapme_eval/` | downloads, reference builders, model requests, food coding and scoring |
| `configs/models/` | one file per model, with its revision, license and run settings |
| `configs/engines/engines.toml` | engine images fixed by digest |
| `hpc/atlas/` | Slurm and Apptainer jobs |
| `results/` | collected scores and reference build records |
| `analysis/` | R figure code |
| `site/` | Astro results site for GitHub Pages |
| `containers/` | harness image and Docker Compose configuration |
| `tools/` | model-registration, Kubernetes-manifest and project-renaming tools |

## Sources and licenses

SNAPMe DB v1.0: Chin, Larke, Bouzid, Nguyen, Vainberg, Smilowitz and Lemay, Ag Data Commons,
[doi:10.15482/USDA.ADC/1528346](https://doi.org/10.15482/USDA.ADC/1528346), CC BY-SA 4.0.
The study is described by Larke et al. (2023), *Nutrients* 15:4972,
[doi:10.3390/nu15234972](https://doi.org/10.3390/nu15234972).

Nutrition5k, released by Google Research: Quin Thames, Arjun Karpur, Wade Norris, Fangting Xia, Liviu Panait,
Tobias Weyand and Jack Sim. "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food."
*2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021.
[doi:10.1109/CVPR46437.2021.00879](https://doi.org/10.1109/CVPR46437.2021.00879).
The [source repository](https://github.com/google-research-datasets/Nutrition5k) is archived and read-only;
data files are downloaded from the `nutrition5k_dataset` Google Cloud Storage bucket.
Nutrition5k is used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Results derived from it
are released under CC BY 4.0 with credit to Nutrition5k (Thames et al. 2021).

Computing: This research used resources provided by the SCINet project and/or the AI Center of Excellence of the
USDA Agricultural Research Service, ARS project numbers 0201-88888-003-000D and 0201-88888-002-000D
([SCINet](https://scinet.usda.gov/guides/resources/citation)).

The SNAPMe paper's [word sets and baseline outputs](https://github.com/JulesLarke-USDA/SNAPMe) are read at a fixed
revision. FNDDS 2017-2018 comes from the USDA ARS Food Surveys Research Group. Code, configurations and
documentation are CC0 1.0; results and reference tables derived from SNAPMe are CC BY-SA 4.0.
Model licenses are recorded in `configs/models/`.

USDA Agricultural Research Service, Western Human Nutrition Research Center, Davis, California.
