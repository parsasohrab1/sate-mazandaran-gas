# تشخیص عیوب جوش خطوط لوله مبتنی بر هوش مصنوعی · Sate Mazandaran Gas

# Weld Defect Detection in Pipelines — AI-Based (Sate Mazandaran Gas)

[![CI](https://github.com/parsasohrab1/sate-mazandaran-gas/actions/workflows/ci.yml/badge.svg)](https://github.com/parsasohrab1/sate-mazandaran-gas/actions/workflows/ci.yml)
[![ML Pipeline](https://github.com/parsasohrab1/sate-mazandaran-gas/actions/workflows/ml.yml/badge.svg)](https://github.com/parsasohrab1/sate-mazandaran-gas/actions/workflows/ml.yml)
[![Security](https://github.com/parsasohrab1/sate-mazandaran-gas/actions/workflows/security.yml/badge.svg)](https://github.com/parsasohrab1/sate-mazandaran-gas/actions/workflows/security.yml)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen)](https://github.com/parsasohrab1/sate-mazandaran-gas)
[![CMMI](https://img.shields.io/badge/CMMI-ML5%20Optimizing-blue)](docs/CMMI_ML5/00_overview.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Proprietary-orange)](LICENSE)

> **⚠️ ایمنی / Safety:** خروجی مدل صرفاً **پشتیبانی از کارشناس** (decision-support) است و
> جایگزین تصمیم ایمنی نیست. هر پیش‌بینی باید توسط کارشناس واجد صلاحیت NDT تأیید شود
> (API 1104، ASME Section V/VIII، ASME B31.3، SRS §5.4).
>
> Model outputs are decision-support only; they must be reviewed by a certified NDT
> specialist before any safety-critical decision.

---

## 📋 فهرست / Table of Contents
- [معرفی / Overview](#معرفی)
- [ویژگی‌ها / Features](#ویژگی‌ها)
- [ساختار مخزن / Repository Structure](#ساختار-مخزن)
- [نصب سریع / Quickstart](#نصب-سریع)
- [اعتبارسنجی با داده سینتتیک / Synthetic Validation](#اعتبارسنجی-سینتتیک)
- [داده واقعی / Real Data](#داده-واقعی)
- [CMMI ML5](#cmmi-ml5)
- [مستندات / Docs](#مستندات)
- [مجریان / Credits](#مجریان)

---

## معرفی

سیستمی مبتنی بر هوش مصنوعی برای **تشخیص و طبقه‌بندی عیوب جوش در خطوط لوله گاز**،
توسعه‌یافته برای **شرکت گاز استان مازندران** (کارفرما) توسط **شرکت پترو پالا توس**
(مجری)، با مجری علمی **دکتر نصرت‌علی اشرفی‌پیامن**. محصول بر اساس مدل **CMMI V2.0 —
سطح بلوغ ۵ (Optimizing)** ساختاردهی شده است.

An AI-based system for **detecting and classifying weld defects in gas pipelines**,
developed for **Sate Mazandaran Gas** (client) by **Petro Pala Toos** (executor),
with scientific lead **Dr. Nosratollah Ashrafi-Payaman**. The product is structured
to **CMMI V2.0 Maturity Level 5 (Optimizing)**.

## ویژگی‌ها

- **الگوریتم ML:** CustomCNN، ResNet50، EfficientNet-B0 و **Ensemble** (softmax avg) با
  focal loss برای عدم توازن کلاس.
- **داده واقعی:** کاتالوگ منابع باز/آکادمیک تأییدشده (RIAWELC، Roboflow، Mendeley،
  HuggingFace) + پروتکل intake داده محرمانه کارفرما.
- **داده سینتتیک:** تولیدگر رادیوگرافی سینتتیک برای اعتبارسنجی اولیه خط لوله.
- **MLOps:** MLflow (ردیابی+رجیستری)، پایپ‌لاین‌های train/eval/deploy، تشخیص انحراف
  (PSI/Evidently)، DAG Airflow برای بازآموزی، canary/rollback.
- **DevOps:** Docker (multi-stage، non-root)، docker-compose، Kubernetes (Deployment/HPA/
  Ingress)، Terraform (EKS+S3+RDS)، Helm chart.
- **CI/CD:** GitHub Actions (ci/cd/security/ml/release/docs)، GitLab CI، Jenkinsfile.
- **API:** FastAPI با OpenAPI خودکار، نقاط `/health`، `/predict`، `/predict/batch`.
- **داشبورد:** HTML راست‌چین با نمودارها (ECharts) + حالت بازبینی کارشناس.
- **CMMI ML5:** دارایی‌های فرآیندی OPF/OPP/OID/CAR/QPM/PAL/DAR + ارزیابی بلوغ.
- **تست:** pytest با پوشش ≥ ۸۰٪ (فعلی ۸۲٫۶۹٪).

## ساختار مخزن

```
sate-mazandaran-gas/
├── src/weld_defect/          # پکیج اصلی (data, models, train, eval, predict, api, cli)
├── configs/                  # train.yaml, eval.yaml, model_catalog.yaml
├── scripts/                  # generate_synthetic_data, prepare_data, download_data
├── data/                     # raw (sources.yaml + README), processed, synthetic
├── docs/                     # BRS, SRS, Architecture, API, Deployment, UserGuide, ...
│   └── CMMI_ML5/             # OPF, OPP, OID, CAR, QPM, PAL, DAR, Maturity, Metrics
├── mlops/                    # mlflow_config, pipelines, drift, airflow, feature_store
├── devops/                   # docker, docker-compose, k8s, terraform, helm
├── cicd/                     # .gitlab-ci.yml, Jenkinsfile, README
├── .github/workflows/        # ci, cd, security, ml, release, docs
├── dashboard/                # index.html (self-contained)
├── tests/                    # pytest suite (config, models, api, e2e, ...)
├── pyproject.toml            # پیکربندی بیلد + ابزارها
├── Makefile                  # اهداف رایج
├── requirements*.txt         # وابستگی‌ها
└── LICENSE                   # اختصاصی (Petro Pala Toos / Sate Mazandaran Gas)
```

## نصب سریع

```bash
git clone https://github.com/parsasohrab1/sate-mazandaran-gas.git
cd sate-mazandaran-gas
make dev                     # venv + نصب وابستگی‌های runtime + dev

# تولید داده سینتتیک + اعتبارسنجی خط لوله
make data-synthetic
make train                   #configs/train.yaml
make evaluate                #metrics/latest/

# خدمت‌رسانی
make serve                   #API روی :8000
make mlflow                  #MLflow UI روی :5000
# داشبورد: باز کردن dashboard/index.html در مرورگر
```

**اجزای کلیدی خط فرمان:**
```bash
python -m weld_defect.cli train    --config configs/train.yaml
python -m weld_defect.cli evaluate --config configs/eval.yaml
python -m weld_defect.cli predict  --image weld.png --model models/ensemble_best.pt
python -m weld_defect.cli serve    --host 0.0.0.0 --port 8000
```

## اعتبارسنجی سینتتیک

برای کاهش ریسک پیش از دسترسی به داده واقعی، کل خط لوله روی داده سینتتیک اعتبارسنجی
می‌شود (`scripts/generate_synthetic_data.py`). این مرحله در CI خودکار اجرا می‌شود
(`.github/workflows/ml.yml`) و تست e2e (`tests/test_e2e.py`) آن را پوشش می‌دهد.

```bash
python scripts/generate_synthetic_data.py --n-samples 1200 --out data/synthetic
python scripts/prepare_data.py --raw data/synthetic --processed data/processed
```

## داده واقعی

منابع داده واقعی در [`data/raw/README.md`](data/raw/README.md) و
[`data/raw/sources.yaml`](data/raw/sources.yaml) کاتالوگ شده‌اند:

| منبع | نوع | لایسنس |
|------|------|--------|
| RIAWELC (Totino et al., 2023) | ۲۴٬۴۰۷ رادیوگرافی | CC BY 4.0 |
| Roboflow "weld defect detection" | object-detection | CC BY 4.0 |
| Mendeley "Weld defect identification" | X-ray | CC BY |
| HuggingFace `admin1523/...` | رادیوگرافی | متغیر |
| Sate Mazandaran Gas (داخلی) | رادیوگرافی اختصاصی | محرمانه (توافق‌نامه) |

```bash
python scripts/download_data.py --list              # نمایش کاتالوگ
python scripts/download_data.py --sources roboflow  # دانلود یک منبع
python scripts/prepare_data.py --raw data/raw/<src> --processed data/processed
```

## CMMI ML5

دارایی‌های فرآیندی بلوغ ۵ در [`docs/CMMI_ML5/`](docs/CMMI_ML5/) مستند شده‌اند:

| Practice Area | سند |
|---------------|-----|
| OPF — Organizational Process Focus | [01_OPF.md](docs/CMMI_ML5/01_OPF.md) |
| OPP — Organizational Process Performance | [02_OPP.md](docs/CMMI_ML5/02_OPP.md) |
| OID — Innovation & Deployment | [03_OID.md](docs/CMMI_ML5/03_OID.md) |
| CAR — Causal Analysis & Resolution | [04_CAR.md](docs/CMMI_ML5/04_CAR.md) |
| QPM — Quantitative Process Management | [05_QPM.md](docs/CMMI_ML5/05_QPM.md) |
| PAL — Process Asset Library | [06_PAL.md](docs/CMMI_ML5/06_PAL.md) |
| DAR — Decision Analysis & Resolution | [07_DAR.md](docs/CMMI_ML5/07_DAR.md) |
| ارزیابی بلوغ | [08_MaturityAssessment.md](docs/CMMI_ML5/08_MaturityAssessment.md) |
| تعاریف معیارها | [09_MetricsDefinitions.md](docs/CMMI_ML5/09_MetricsDefinitions.md) |

## مستندات

| سند | شرح |
|-----|-----|
| [BRS](docs/BRS.md) | نیازمندی‌های کسب‌وکار |
| [SRS](docs/SRS.md) | نیازمندی‌های نرم‌افزار (IEEE-830) |
| [Architecture](docs/Architecture.md) | معماری سیستم |
| [API](docs/API.md) | راهنمای REST API |
| [Deployment](docs/Deployment.md) | راهنمای استقرار |
| [UserGuide](docs/UserGuide.md) | راهنمای کاربری |
| [DataDictionary](docs/DataDictionary.md) | فرهنگ داده |
| [LabelingProtocol](docs/LabelingProtocol.md) | پروتکل برچسب‌گذاری |

## مجریان

| نقش | نام |
|-----|-----|
| کارفرما | شرکت گاز استان مازندران (شرکت ملی گاز ایران) |
| مجری | شرکت پترو پالا توس (شناسه ملی 10320833145) — r&d@petropalatoos.ir |
| مجری علمی | دکتر نصرت‌علی اشرفی‌پیامن — ashrafi@khu.ac.ir |
| اعتبار | ۷٬۰۰۰٬۰۰۰٬۰۰۰ ریال · مدت ۱۴ ماه · تاریخ پایه ۱۴۰۵/۰۷/۲۹ |

---

**استانداردها:** API 1104 · ASME Section V · ASME Section VIII · ASME B31.3 · IEEE 830 · CMMI V2.0
