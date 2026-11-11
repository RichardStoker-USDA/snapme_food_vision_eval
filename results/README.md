# Result files

`snapme-eval report` collects scored runs here for the site and release packaging. Each benchmark has a
separate directory. `leaderboard.json` includes intervals and grouped measures; `leaderboard.csv` contains
scalar columns. `gt_build.json` records the reference source, counts and exclusions.

Per-run detail files under `<benchmark>/runs/` are not tracked in Git. Release packaging reads them to produce
per-photo tables and copies the corresponding raw predictions. A checkout alone therefore does not contain
all inputs needed to rebuild a results release.
