"""Stage the public data inputs used by the calibration and scenario code.

The model reads INRB/UMIE processed data from ``data/BDBV2026-Data`` and a
Flowminder HDX origin-destination matrix from ``data/flowminder_hdx``. This
script fetches both and rebuilds the March 2026 square HDX outflow matrix used
by ``model.data.load_mobility_outflow(source="hdx")``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BDBV_REPO = "https://github.com/INRB-UMIE/BDBV2026-Data.git"
BDBV_COMMIT = "60f67015625a9f1cf54d9da5339926121d90b72d"
# Upstream CI tags each build that passes its QA gate (`tools.qa`) as
# `build-<date>-<sha>`; `--bdbv-commit ""` checks out the newest such tag.
BDBV_RELEASE_TAGS = "refs/tags/build-*"
HDX_PACKAGES = (
    "bd5781f3-9c6a-427a-955b-ce2b59def8c3",
    "f3660479-fb8d-4a74-baa7-09d2d24c8e0d",
)
HDX_PACKAGE_SHOW = "https://data.humdata.org/api/3/action/package_show?id={package_id}"

# HDX republishes in place, so the package IDs above are not by themselves a
# pin. These are the relocation file and the derived matrix the manuscript was
# built from; `--allow-hdx-drift` downgrades a mismatch to a warning.
HDX_RELOCATIONS_FILE = "drc-estimated-relocations-2020_03-2026_04-v2.0-external.csv"
HDX_RELOCATIONS_SHA256 = "0ebd8344f1c1fe981b3908f92a2e313bc31c0615dd92a21a899cc3595a32240a"
HDX_MATRIX_SHA256 = "0e5c17fc3e55820d9d27e9a83222e8f7863babc3dec7834d5b8b1028d69b13b5"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str, label: str, allow_drift: bool) -> None:
    actual = sha256(path)
    if actual == expected:
        print(f"verified {label}: sha256 {actual}")
        return
    message = (
        f"{label} does not match the version used for the manuscript.\n"
        f"    file:     {path}\n"
        f"    expected: {expected}\n"
        f"    actual:   {actual}\n"
        "    The upstream HDX resource has changed since publication. Results\n"
        "    built from this file will not reproduce the manuscript figures."
    )
    if not allow_drift:
        raise SystemExit(
            f"ERROR: {message}\n"
            "    Re-run with --allow-hdx-drift to stage the current data anyway."
        )
    print(f"WARNING: {message}\n    Continuing because --allow-hdx-drift was passed.")


def latest_release(repo: Path) -> str:
    """Return the commit of the newest QA-verified upstream release tag."""
    print(f"+ resolving newest {BDBV_RELEASE_TAGS} tag")
    tag = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--sort=-committerdate",
            "--count=1",
            "--format=%(refname:short)",
            BDBV_RELEASE_TAGS,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not tag:
        raise SystemExit(f"ERROR: no {BDBV_RELEASE_TAGS} tags found in {repo}")
    commit = subprocess.run(
        ["git", "rev-parse", f"{tag}^{{commit}}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"newest verified release: {tag} ({commit})")
    return commit


def stage_inrb(commit: str = BDBV_COMMIT, force: bool = False) -> None:
    dest = DATA / "BDBV2026-Data"
    if force and dest.exists():
        shutil.rmtree(dest)
    if not dest.exists():
        run(["git", "clone", BDBV_REPO, str(dest)])
    run(["git", "fetch", "--tags", "origin"], cwd=dest)
    if not commit:
        commit = latest_release(dest)
    run(["git", "checkout", commit], cwd=dest)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "bdbv/0.1"})
    print(f"download {url}\n    -> {dest}")
    with urllib.request.urlopen(request) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def stage_hdx(force: bool = False, allow_drift: bool = False) -> None:
    raw = DATA / "flowminder_hdx" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for package_id in HDX_PACKAGES:
        with urllib.request.urlopen(HDX_PACKAGE_SHOW.format(package_id=package_id)) as r:
            package = json.load(r)["result"]
        for resource in package["resources"]:
            url = resource.get("url") or ""
            if not url.lower().endswith(".csv"):
                continue
            dest = raw / url.rsplit("/", 1)[-1]
            if force or not dest.exists():
                download(url, dest)
    verify(
        find_relocation_file(raw, allow_drift=allow_drift),
        HDX_RELOCATIONS_SHA256,
        "HDX relocation file",
        allow_drift,
    )
    process_hdx(allow_drift=allow_drift)


def clean_health_zone_name(value: object) -> str:
    name = str(value).strip()
    if " " in name and len(name.split(" ", 1)[0]) == 2:
        name = name.split(" ", 1)[1]
    return name.replace(" Zone de Sante", "").replace(" Zone de Santé", "").strip()


def find_relocation_file(raw: Path, allow_drift: bool = False) -> Path:
    pinned = raw / HDX_RELOCATIONS_FILE
    if pinned.exists():
        return pinned
    if not allow_drift:
        raise SystemExit(
            f"ERROR: expected HDX relocation file {HDX_RELOCATIONS_FILE} not found in {raw}.\n"
            "    The upstream HDX package no longer publishes the file used for the\n"
            "    manuscript. Re-run with --allow-hdx-drift to fall back to the first\n"
            "    CSV carrying from_hz_name/to_hz_name columns."
        )
    for path in sorted(raw.glob("*.csv")):
        header = path.open(encoding="utf-8-sig").readline()
        if "from_hz_name" in header and "to_hz_name" in header:
            print(f"WARNING: falling back to {path.name} instead of {HDX_RELOCATIONS_FILE}")
            return path
    raise FileNotFoundError(f"No HDX relocation CSV found in {raw}")


def process_hdx(month: str = "2026_03", allow_drift: bool = False) -> None:
    root = DATA / "flowminder_hdx"
    raw = root / "raw"
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    source = find_relocation_file(raw, allow_drift=allow_drift)
    relocations = pd.read_csv(source)
    flow_col = f"est_flows_{month}"
    if flow_col not in relocations:
        raise KeyError(f"{flow_col} not present in {source}")

    table = relocations[["from_hz_name", "to_hz_name", flow_col]].copy()
    table["from_hz_name"] = table["from_hz_name"].map(clean_health_zone_name)
    table["to_hz_name"] = table["to_hz_name"].map(clean_health_zone_name)
    table[flow_col] = pd.to_numeric(table[flow_col], errors="coerce").fillna(0.0)
    matrix = table.pivot_table(
        index="from_hz_name",
        columns="to_hz_name",
        values=flow_col,
        aggfunc="sum",
        fill_value=0.0,
    )
    zones = sorted(set(matrix.index) | set(matrix.columns))
    matrix = matrix.reindex(index=zones, columns=zones, fill_value=0.0)
    matrix_path = processed / "flowminder_hdx__outflow_2026_03.matrix.csv"
    matrix.to_csv(matrix_path, index_label="nom")
    verify(matrix_path, HDX_MATRIX_SHA256, "processed HDX outflow matrix", allow_drift)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--bdbv-commit",
        default=BDBV_COMMIT,
        help=(
            "INRB-UMIE/BDBV2026-Data commit to check out (default: %(default)s). "
            'Pass "" to check out the newest QA-verified release tag (build-*).'
        ),
    )
    parser.add_argument(
        "--skip-hdx",
        action="store_true",
        help="Only clone/check out INRB-UMIE/BDBV2026-Data.",
    )
    parser.add_argument(
        "--allow-hdx-drift",
        action="store_true",
        help=(
            "Continue when the HDX inputs no longer match the checksums used for "
            "the manuscript. Staged data will not reproduce the published figures."
        ),
    )
    args = parser.parse_args()

    stage_inrb(commit=args.bdbv_commit, force=args.force)
    if not args.skip_hdx:
        stage_hdx(force=args.force, allow_drift=args.allow_hdx_drift)


if __name__ == "__main__":
    main()
