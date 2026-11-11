# Datasets for food identification and amount estimation

For SNAPMe, the benchmark compares food lists estimated by models from participants' phone photos of daily
meals with the ASA24 lines those participants entered. The additional Nutrition5k evaluation compares food
lists estimated by models from overhead cafeteria photos with the dishes' weighed ingredients. A camera on
a fixed rig looked down from directly above each plate. The two comparisons
answer different questions. Their scores use different references and matching rules and are not directly
comparable.

SNAPMe and Nutrition5k have reference builders and scoring paths in this repository. The remaining datasets
below are candidates for separate evaluations, not inputs to the current results. Counts and stated licenses
were reviewed against the linked sources on November 9, 2026. The Nutrition5k entry was checked again on
November 11, 2026.

| Dataset | Images and reference | Status in this benchmark |
|---|---|---|
| [SNAPMe DB v1.0](https://doi.org/10.15482/USDA.ADC/1528346) | Current reference: 1,675 before photos or package fronts from 95 adults, linked to ASA24 lines; companion photos are retained where available | Reported foods and grams; CC BY-SA 4.0 |
| [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) | Source contains 5,006 cafeteria dishes with weighed ingredients; overhead color and depth images are available for a subset | Separate comparison on the 507 dishes in the authors' overhead test list; CC BY 4.0 |
| [Food Portion Benchmark](https://huggingface.co/datasets/issai/Food_Portion_Benchmark) | 14,083 images in 138 food classes, with item rectangles and measured weights from a controlled setting | Candidate for amount estimation; dataset card states CC BY-NC 4.0 and hidden test labels |
| [NutritionVerse-Real](https://arxiv.org/abs/2401.08598) | 889 images of 251 dishes with weighed ingredients | Candidate for amount estimation; dataset license not confirmed here |
| [MetaFood3D](https://arxiv.org/abs/2409.01966) | 743 scanned food objects in 131 categories, with weights and food codes linked to a nutrition database | Candidate for volume estimation; dataset license not confirmed here |
| [January Food Benchmark](https://arxiv.org/abs/2508.09966) | 1,000 food images with human-validated annotations | Candidate for identification; dataset license and amount-reference procedure not confirmed here |
| [VLM Food Benchmark](https://github.com/Metabolic-Intelligence-Lab/vlm-food-benchmark) | 200 university-canteen meal photos with item weights measured on a kitchen scale | Candidate for identification and amounts; [data license](https://github.com/Metabolic-Intelligence-Lab/vlm-food-benchmark/blob/main/LICENSE-DATA) is CC BY 4.0 |

SNAPMe's count above is the number of evaluation photos in `results/snapme_v1/gt_build.json`, not the total
number of images in the source dataset. The Nutrition5k reference contains ingredient names, ingredient grams
and total dish mass. This benchmark assigns the ingredient names FNDDS codes with the same mapper used for
predicted names. Its identification reference excludes ingredients below
5 g. Amount scoring sums positive gram estimates and compares that sum with the full weighed dish mass.
Dishes with no positive estimate are omitted from amount errors; the share with an amount is reported separately.

Nutrition5k was released by Google Research and described by Quin Thames, Arjun Karpur, Wade Norris,
Fangting Xia, Liviu Panait, Tobias Weyand and Jack Sim. "Nutrition5k: Towards Automatic Nutritional
Understanding of Generic Food." *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*,
2021. [doi:10.1109/CVPR46437.2021.00879](https://doi.org/10.1109/CVPR46437.2021.00879).
The [source repository](https://github.com/google-research-datasets/Nutrition5k) is archived and read-only;
data files are downloaded from the dataset's Google Cloud Storage bucket, `nutrition5k_dataset`.
The data are used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and results derived from them
are released under CC BY 4.0 with credit to Nutrition5k (Thames et al. 2021).

## Adding a dataset

Each dataset needs a versioned reference under `gt/`, a builder in `src/snapme_eval/gt/` and a scorer suited
to its reference. `images.parquet` identifies photos and evaluation units; `items.parquet` holds foods and
amounts. Column meanings and matching rules must be defined for that dataset. Writing tables with the same
filenames alone does not make the SNAPMe scorer applicable.

Each evaluation reports agreement with its own reference: food records, weighed portions or scanned geometry.
A new evaluation also needs confirmed data-use terms and documented split and exclusion rules.

## Related studies

- [DietAI24](https://www.nature.com/articles/s43856-025-01159-0), *Communications Medicine* (2025).
- [Geometry-Enhanced Portion Estimation for Multimodal LLMs](https://arxiv.org/abs/2607.16514) (2026).
- [DietDelta: A Vision-Language Approach for Dietary Assessment via Before-and-After Images](https://arxiv.org/abs/2604.06352) (2026).

Dataset access: [NutritionVerse-Real](https://www.kaggle.com/datasets/nutritionverse/nutritionverse-real) and
[MetaFood3D](https://lorenz.ecn.purdue.edu/~food3d/). A paper's license does not establish the license of its
associated data files.
