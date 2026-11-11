"""Rename the project in one step.

    python tools/rename_project.py --name "New Name" --slug new_slug [--apply]

The site and reports read the display name and slug from project.toml. This tool rewrites that file
and the README title and repository links, then prints commands for completing a rename
outside this folder (GitHub repository, local folder, workspace catalog). Internal names do not change:
the Python package stays `snapme_eval` and the data root (SNAPME_ROOT) stays where it is, so no data moves.
Without --apply it only shows what would change.
"""
from __future__ import annotations

import argparse
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", a.slug):
        raise SystemExit("slug: lowercase letters, digits, _ or -")

    toml = ROOT / "project.toml"
    t = toml.read_text()
    old_name = re.search(r'^name = "([^"]*)"', t, re.M).group(1)
    old_slug = re.search(r'^slug = "([^"]*)"', t, re.M).group(1)
    t2 = re.sub(r'^name = "[^"]*"', f'name = "{a.name}"', t, count=1, flags=re.M)
    t2 = re.sub(r'^slug = "[^"]*"', f'slug = "{a.slug}"', t2, count=1, flags=re.M)

    readme = ROOT / "README.md"
    r = readme.read_text()
    r2 = re.sub(r"^# .*$", f"# {a.name}", r, count=1, flags=re.M).replace(f"/{old_slug}", f"/{a.slug}")

    print(f"name: {old_name} -> {a.name}\nslug: {old_slug} -> {a.slug}")
    if a.apply:
        toml.write_text(t2)
        readme.write_text(r2)
        print("updated project.toml and README.md; rebuild the site (cd site && npm run build)")
    print("\nThen, when ready:")
    print(f"  gh repo rename {a.slug}            # if the GitHub repository already exists")
    print(f"  mv {ROOT} {ROOT.parent / a.slug}   # local folder (update catalog/workstreams.json path)")
    print("  git commit -am 'Rename project'")


if __name__ == "__main__":
    main()
