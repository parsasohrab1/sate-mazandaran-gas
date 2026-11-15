# Real Data Sources Catalog — Weld Defect Radiographs

This directory holds **real** radiograph datasets for the project, alongside the
internal Sate Mazandaran Gas archive. The machine-readable catalog is
[`sources.yaml`](./sources.yaml); it is consumed by
[`scripts/download_data.py`](../../scripts/download_data.py).

> ⚠️ **Never commit real images to git.** The repository `.gitignore` excludes
> `data/raw/*`, `data/processed/*`, and `data/synthetic/*.png`. Only this
> README, `sources.yaml`, and `.gitkeep` are tracked.

## 1. Verified open / academic datasets

| # | Source | Kind | Size | License | URL |
|---|--------|------|------|---------|-----|
| 1 | **RIAWELC** (Totino et al., 2023) | radiographs | 24,407 images | CC BY 4.0 | https://doi.org/10.1016/j.dib.2023.109760 |
| 2 | **Roboflow Universe — "weld defect detection"** | object-detection | CC BY 4.0 | https://universe.roboflow.com/xray-dui1c/weld-defect-detection-yf4kg |
| 3 | **Mendeley — "Weld defect identification"** | X-ray images | CC BY | https://www.mendeley.com/catalogue/f920550a-65ca-36fe-82e5-28dc4b93f821 |
| 4 | **HuggingFace — `admin1523/Weld-defect-detection-datasets`** | radiographs | see card | https://huggingface.co/datasets/admin1523/Weld-defect-detection-datasets |

### RIAWELC (primary external benchmark)
Totino, B. et al. (2023). *RIAWELC: A Novel Dataset of Radiographic Images for
Welding Defect Classification.* **Data in Brief**, doi:10.1016/j.dib.2023.109760.
- 24,407 radiographic images; classes include lack of penetration, cracks,
  porosity — directly comparable to the project's class catalogue.
- Used as the **external benchmark** for generalization claims (BRS §6).

### Roboflow Universe — "weld defect detection"
- CC BY 4.0; downloadable via the Roboflow API (`ROBOFLOW_API_KEY` env var).
- Object-detection format; we use the per-class folder export for the
  classification pipeline.

## 2. Internal source (agreement-gated)

### Sate Mazandaran Gas NDT archive
- **Proprietary.** Supplied by the Licensee under the project data-sharing
  protocol (see [`docs/DataDictionary.md`](../../docs/DataDictionary.md) §3
  and [`docs/LabelingProtocol.md`](../../docs/LabelingProtocol.md)).
- Covers the Licensee's own pipeline radiographs (DICOM/PNG) and NDT reports
  for the segments in scope.
- **NOT auto-downloaded.** Export images + a `manifest.csv`
  (`image,label,group,segment,pipeline_id,date`) into
  `data/raw/sate_mazandaran_gas/`, then run
  `python scripts/prepare_data.py --raw data/raw/sate_mazandaran_gas --processed data/processed`.

## 3. Acquisition workflow (CMMI ML5 — OPF/OPP traceable)

```
sources.yaml ──► scripts/download_data.py ──► data/raw/<source>/
                                                       │
                                                       ▼
                              scripts/prepare_data.py  (preprocess + split)
                                                       │
                                                       ▼
                                   data/processed/{images,train,val,test}.csv
                                                       │
                                                       ▼
                                   weld_defect.train  (MLflow-tracked)
```

1. `python scripts/download_data.py --list` — print the catalog.
2. `python scripts/download_data.py --sources roboflow` — fetch one source.
   (`--sources all` tries every source; manual sources create a placeholder dir.)
3. For the internal archive, place files under `data/raw/sate_mazandaran_gas/`.
4. `python scripts/prepare_data.py --raw data/raw/<src> --processed data/processed`
   → produces preprocessed PNGs + stratified-group train/val/test splits +
   a `data_profile.json` (imbalance ratio, missing-image count) for OPP/QPM.

## 4. Data governance

- All external datasets are used under their stated CC BY / academic licenses;
  the source paper is cited in any derivative report.
- The internal archive is treated as **confidential**; access is logged and
  role-based (see [`docs/SRS.md`](../../docs/SRS.md) §5.6 — security).
- Real images are excluded from git and from MLflow artifacts by default;
  only metrics, manifests (without pixel data), and model weights are tracked.
