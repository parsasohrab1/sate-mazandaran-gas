# PAL — Process Asset Library (کتابخانه دارایی‌های فرآیند)

## ۱. هدف
نگهداری متمرکز و نسخه‌شده همه دارایی‌های فرآیند برای استفاده مجدد، آموزش و
بهبود. PAL منبع حقیقت سازمان برای فرآیندها است.

## ۲. ساختار PAL

PAL در این مخزن به دو بخش تقسیم می‌شود:

### ۲.۱ دارایی‌های اسنادی (`docs/` و `docs/CMMI_ML5/`)
| دارایی | مسیر | نسخه |
|--------|------|------|
| BRS | `docs/BRS.md` | 1.0 |
| SRS | `docs/SRS.md` | 1.0 |
| معماری | `docs/Architecture.md` | 1.0 |
| API | `docs/API.md` | 1.0 |
| استقرار | `docs/Deployment.md` | 1.0 |
| راهنمای کاربری | `docs/UserGuide.md` | 1.0 |
| فرهنگ داده | `docs/DataDictionary.md` | 1.0 |
| پروتکل برچسب | `docs/LabelingProtocol.md` | 1.0 |
| OPF/OPP/OID/CAR/QPM/DAR | `docs/CMMI_ML5/*.md` | 1.0 |

### ۲.۲ دارایی‌های اجرایی (کد و پیکربندی)
| دارایی | مسیر |
|--------|------|
| پایپ‌لاین پیش‌پردازش | `src/weld_defect/data/preprocess.py` |
| پایپ‌لاین آموزش | `src/weld_defect/train.py` |
| پایپ‌لاین ارزیابی | `src/weld_defect/evaluate.py` |
| پیکربندی آموزش | `configs/train.yaml` |
| پیکربندی ارزیابی | `configs/eval.yaml` |
| کاتالوگ مدل | `configs/model_catalog.yaml` |
| کاتالوگ داده | `data/raw/sources.yaml` |
| پایپ‌لاین‌های MLOps | `mlops/pipelines/*.py` |
| DAG Airflow | `mlops/airflow/dags/retrain.py` |
| CI/CD | `cicd/*.yml`, `.github/workflows/` |
| Dockerfile/Compose/Helm | `devops/` |
| اسکریپت‌های کمکی | `scripts/` |

## ۳. نسخه‌گذاری PAL
- همه دارایی‌ها در git نسخه‌گذاری می‌شوند.
- تغییر فرآیند با PR و ارجاع به CAR/OID-id.
- هر نسخه انتشار (release) یک عکس‌برداری (snapshot) از PAL است (`git tag v*`).

## ۴. دسته‌بندی دارایی‌ها (Asset Categories)
- **Process descriptions:** سندهای `docs/` و `CMMI_ML5/`.
- **Process assets:** پیکربندی‌های `configs/`.
- **Tailoring guides:** `Makefile` و `docs/Deployment.md` برای اقتباس محیطی.
- **Lessons learned:** `metrics/car_log.yaml` و `metrics/innovation_backlog.yaml`.
- **Templates:** `data/raw/sources.yaml` (قالب کاتالوگ)، `configs/*.yaml` (قالب پیکربندی).

## ۵. اقتباس (Tailoring)
هر پروژه/پایلوت می‌تواند PAL را با محدودیت‌های محیط اقتباس کند:
- کاهش `batch_size` برای CPU.
- خاموش‌کردن augmentation قوی برای ارزیابی نهایی.
- تغییر `class_weights` بر اساس پروفایل داده.
اقتباس باید در `metrics/tailoring_log.yaml` ثبت شود.

## ۶. مالکیت و دسترسی
- **مالک PAL:** مدیر فنی پترو پالا توس.
- **دسترسی:** خواندن برای همه تیم؛ نوشتن با PR و تأیید Process Owner.
- **حذفیات:** دارایی منسوخ با برچسب `deprecated` نگه‌داری می‌شود (نه حذف) برای ردیابی تاریخی.
