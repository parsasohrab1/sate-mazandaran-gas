#!/usr/bin/env python3
"""Download weld-defect radiograph datasets from the catalog.

Usage:
  python scripts/download_data.py --list
  python scripts/download_data.py --sources roboflow,mendeley_weld_identification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weld_defect.config import load_yaml
from weld_defect.utils.io import ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download weld-defect datasets")
    parser.add_argument("--catalog", default="data/raw/sources.yaml", help="Catalog YAML")
    parser.add_argument("--list", action="store_true", help="List catalog and exit")
    parser.add_argument("--sources", default="", help="Comma-separated source names to download")
    parser.add_argument("--out-root", default="data/raw", help="Output root directory")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"Error: catalog not found: {catalog_path}", file=sys.stderr)
        return 1

    catalog = load_yaml(catalog_path)
    sources = catalog.get("sources", {})

    if args.list or not args.sources:
        print(f"\nWeld-defect dataset catalog ({catalog_path}):\n")
        for name, info in sources.items():
            print(f"  {name}")
            print(f"    kind: {info.get('kind', '?')}")
            print(f"    license: {info.get('license', '?')}")
            if "url" in info:
                print(f"    url: {info['url']}")
            if "paper_url" in info:
                print(f"    paper: {info['paper_url']}")
            print(f"    note: {info.get('note', '').strip()[:120]}...")
            print()
        if not args.sources:
            print("Use --sources name1,name2 to download specific sources.")
        return 0

    requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    out_root = ensure_dir(args.out_root)

    for name in requested:
        if name not in sources:
            print(f"Warning: unknown source '{name}', skipping.", file=sys.stderr)
            continue
        info = sources[name]
        out_dir = out_root / name
        ensure_dir(out_dir)
        kind = info.get("kind", "manual")
        print(f"\n→ Downloading '{name}' (kind={kind}) → {out_dir}")

        if kind == "roboflow":
            _download_roboflow(info, out_dir)
        elif kind == "huggingface":
            _download_huggingface(info, out_dir)
        elif kind == "manual":
            print(f"  Manual source. License: {info.get('license')}")
            if "url" in info:
                print(f"  Download from: {info['url']}")
            if "paper_url" in info:
                print(f"  Paper: {info['paper_url']}")
            print(f"  Place extracted files under: {out_dir}/")
            print(f"  Note: {info.get('note', '').strip()[:200]}")
        else:
            print(f"  Unknown kind '{kind}', skipping.")

    return 0


def _download_roboflow(info: dict, out_dir: Path) -> None:
    import os
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("  ROBOFLOW_API_KEY env var not set. Get a free key at https://roboflow.com")
        return
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        workspace = info.get("workspace", "")
        project = info.get("project", "")
        version = info.get("version", 1)
        proj = rf.workspace(workspace).project(project)
        proj.version(version).download("folder", location=str(out_dir))
        print(f"  Downloaded to {out_dir}")
    except ImportError:
        print("  roboflow package not installed. Install with: pip install roboflow")
    except Exception as e:
        print(f"  Download failed: {e}")


def _download_huggingface(info: dict, out_dir: Path) -> None:
    import os
    dataset_id = info.get("dataset_id", "")
    if not dataset_id:
        print("  No dataset_id specified in catalog.")
        return
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=str(out_dir), token=token)
        print(f"  Downloaded to {out_dir}")
    except ImportError:
        print("  huggingface_hub not installed. Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"  Download failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
