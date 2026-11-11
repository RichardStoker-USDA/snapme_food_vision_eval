"""Download registry models at their pinned commits (Atlas: run on atlas-dtn in the hub environment).

    source hpc/atlas/env.sh
    $SNAPME_ROOT/env/hub/bin/python hpc/atlas/download_models.py configs/models/*.toml

The logic lives in src/snapme_eval/download.py (`snapme-eval download --model ID` does the same).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from snapme_eval.download import download  # noqa: E402

if __name__ == "__main__":
    download(sys.argv[1:])
