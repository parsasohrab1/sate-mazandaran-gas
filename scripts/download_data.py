#!/usr/bin/env python3
"""Download real radiograph datasets catalogued in data/raw/sources.yaml.

Real-data sources (verified open / academic datasets — see data/raw/README.md):

    1. RIAWELC (Totino et al., 2023) — 24,407 radiographic weld images,
       classes: lack of penetration, cracks, porosity, etc. Academic dataset.
    2. Roboflow Universe "weld defect detection" by Xray — CC BY 4.0,
       object-detection format, downloadable via Roboflow API.
    3. Mendeley Data "Weld defect identification" — small X-ray image set.
    4. HuggingFace Datasets hub — community weld-defect image datasets.

Internal source (NOT auto-downloaded — agreement-gated):
    Sate Mazandaran Gas NDT archive — supplied by the Licensee under the
    project data-sharing protocol (docs/DataDictionary.md §3). Place exported
    DICOM/PNG images under data/raw/sate_mazandaran/ and run
    scripts/prepare_data.py to ingest them.

Usage:
    python scripts/download_data.py --sources all          # try every source
    python scripts/download_data.py --sources roboflow     # only one
    python scripts/download_data.py --list                 # just print catalog
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "data" / "raw" / "sources.yaml"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict:
    if not path.exists():
        print(f"[warn] catalog not found at {path}; nothing to download.", file=sys.stderr)
        return {"sources": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"sources": {}}


def _http_get(url: str, dst: Path, token: str | None = None) -> None:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp, dst.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)


def _extract(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.suffix in (".tar", ".gz", ".tgz"):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        # treat as raw file
        dest.mkdir(parents=True, exist_ok=True)
        (dest / archive.name).write_bytes(archive.read_bytes())


def fetch_hf(dataset_id: str, out: Path, token: str | None) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [skip] huggingface_hub not installed; pip install huggingface_hub", file=sys.stderr)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=dataset_id, repo_type="dataset",
                             local_dir=str(out), token=token)
    n = sum(1 for _ in Path(path).rglob("*") if _.is_file())
    return n


def fetch_roboflow(workspace: str, project: str, version: int, out: Path, key: str | None) -> int:
    if not key:
        print(f"  [skip] Roboflow needs ROBOFLOW_API_KEY env var ({workspace}/{project})", file=sys.stderr)
        return 0
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  [skip] roboflow SDK not installed; pip install roboflow", file=sys.stderr)
        return 0
    rf = Roboflow(api_key=key)
    ds = rf.workspace(workspace).project(project).version(version).download("folder", location=str(out))
    n = sum(1 for _ in Path(out).rglob("*") if _.is_file())
    return n


def fetch_url(url: str, out: Path, token: str | None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "download.archive"
    print(f"  fetching {url}")
    _http_get(url, archive, token=token)
    _extract(archive, out)
    archive.unlink(missing_ok=True)
    return sum(1 for _ in out.rglob("*") if _.is_file())


def fetch_one(name: str, spec: dict, out_root: Path) -> dict:
    kind = spec.get("kind")
    dest = out_root / name
    token = os.environ.get("DATA_DOWNLOAD_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        if kind == "huggingface":
            n = fetch_hf(spec["dataset_id"], dest, token)
        elif kind == "roboflow":
            n = fetch_roboflow(spec["workspace"], spec["project"], int(spec["version"]), dest,
                               os.environ.get("ROBOFLOW_API_KEY"))
        elif kind == "url":
            n = fetch_url(spec["url"], dest, token)
        elif kind == "manual":
            print(f"  [manual] {name}: {spec.get('note','')}", file=sys.stderr)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "PLACE_FILES_HERE.txt").write_text(
                f"Place {name} files here.\n{spec.get('note','')}\n", encoding="utf-8")
            return {"name": name, "ok": True, "files": 0, "note": "manual acquisition"}
        else:
            return {"name": name, "ok": False, "error": f"unknown kind: {kind}"}
        return {"name": name, "ok": True, "files": n}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


def main():
    p = argparse.ArgumentParser(description="Download catalogued real datasets.")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sources", default="all", help="comma-separated source names or 'all'")
    p.add_argument("--list", action="store_true", help="print catalog and exit")
    args = p.parse_args()

    catalog = load_catalog(args.catalog)
    sources = catalog.get("sources", {})
    if args.list:
        print(json.dumps(sources, indent=2, ensure_ascii=False))
        return

    names = list(sources.keys()) if args.sources == "all" else [s.strip() for s in args.sources.split(",")]
    results = []
    for name in names:
        spec = sources.get(name)
        if not spec:
            print(f"[warn] unknown source: {name}", file=sys.stderr)
            continue
        print(f"[fetch] {name} ({spec.get('kind')})")
        results.append(fetch_one(name, spec, args.out))

    print(json.dumps({"results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
