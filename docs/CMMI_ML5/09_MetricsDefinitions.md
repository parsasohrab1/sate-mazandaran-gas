# تعاریف معیارها (Metrics Definitions)

مرجع واحد برای همه معیارهای کمی استفاده‌شده در OPP/QPM/CAR/DAR. هر معیار
با تعریف، فرمول، واحد، حد مطلوب و منبع داده مستند شده است.

## ۱. معیارهای طبقه‌بندی

| معیار | تعریف | فرمول | واحد | حد مطلوب | منبع |
|-------|-------|-------|------|----------|------|
| accuracy | سهم نمونه‌های درست طبقه‌بندی‌شده | (TP+TN)/N | [0,1] | ≥ 0.85 | `metrics.json` |
| balanced_accuracy | میانگین فراخوانی کلاس‌ها | mean(recall_i) | [0,1] | ≥ 0.85 | `metrics.json` |
| precision_macro | میانگین دقت کلاس‌ها | mean(TP/(TP+FP)) | [0,1] | ≥ 0.85 | `metrics.json` |
| recall_macro | میانگین فراخوانی کلاس‌ها | mean(TP/(TP+FN)) | [0,1] | ≥ 0.85 | `metrics.json` |
| f1_macro | میانگین هم‌سانی کلاس‌ها | mean(2·P·R/(P+R)) | [0,1] | ≥ 0.85 (دروازه) | `metrics.json` |
| recall_crack | فراخوانی کلاس ترک (ایمنی‌بحرانی) | TP_crack/(TP_crack+FN_crack) | [0,1] | ≥ 0.95 (دروازه) | `metrics.json → per_class` |
| precision_no_defect | دقت کلاس بدون عیب | TP_no/(TP_no+FP_no) | [0,1] | ≥ 0.90 (دروازه) | `metrics.json → per_class` |
| cohen_kappa | توافق تصحیح‌شده با تصادفی | (p_o−p_e)/(1−p_e) | [−1,1] | ≥ 0.80 | `labeling_agreement.json` |
| matthews_corrcoef | ضریب همبستگی متیوز | see sklearn | [−1,1] | ≥ 0.80 | `metrics.json` |
| roc_auc_ovr | مساحت زیر ROC One-vs-Rest | — | [0,1] | ≥ 0.90 | `metrics.json` |

## ۲. معیارهای عملکردی

| معیار | تعریف | واحد | حد | منبع |
|-------|-------|------|-----|------|
| inference_ms | زمان استنتاج یک تصویر | ms | ≤ 120 (دروازه) | `/predict` log, predict.py |
| throughput | تصویر/ثانیه (batch, GPU) | img/s | ≥ 10 | load test |
| cold_start_s | زمان راه‌اندازی سرویس | s | ≤ 30 | K8s probe |

## ۳. معیارهای داده

| معیار | تعریف | واحد | حد | منبع |
|-------|-------|------|-----|------|
| n_samples | تعداد کل نمونه‌ها | count | — | `data_profile.json` |
| imbalance_ratio | max_class/min_class | ratio | ≤ 5 | `data_profile.json` |
| missing_images | تصاویر گمشده در manifest | count | 0 | `data_profile.json` |
| drift_psi | Population Stability Index (ورودی) | — | ≤ 0.20 | drift report |
| labeling_kappa | κ بین دو labeler | [−1,1] | ≥ 0.80 | `labeling_agreement.json` |

## ۴. معیارهای فرآیند/CI

| معیار | تعریف | واحد | حد | منبع |
|-------|-------|------|-----|------|
| test_pass_rate | نسبت تست‌های موفق CI | [0,1] | = 1.0 | GitHub Actions |
| coverage | پوشش خط کد | % | ≥ 80 | `pytest --cov` |
| lead_time_for_change | زمان از commit تا استقرار | min | ≤ 30 | CI/CD |
| defect_rate | باگ/هفته پس از استقرار | count | ≤ 2 | issue tracker |

## ۵. تعریف دروازه‌های پذیرش (Acceptance Gates)
دروازه‌ها در `configs/eval.yaml → acceptance` و `utils/metrics.py → check_acceptance`
پیاده شده‌اند. هر ارزیابی pass/fail هر دروازه را در `metrics/latest/error_analysis.json`
ثبت می‌کند:
- f1_macro ≥ 0.85
- recall_crack ≥ 0.95
- precision_no_defect ≥ 0.90
- inference_ms ≤ 120

## ۶. قرارداد نام‌گذاری
- پیشوند `val_` / `test_`: مجموعه ارزیابی.
- پیشوند `p95_` / `p99_`: درصد زمان‌ها.
- پسوند `_macro` / `_micro`: نوع میانگین.
