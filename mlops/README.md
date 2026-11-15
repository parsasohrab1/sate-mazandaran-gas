# MLOps — Machine Learning Operations

این پوشه زیرساخت عملیاتی‌سازی ML را نگهداری می‌کند: ردیابی آزمایش، رجیستری
مدل، پایپ‌لاین‌های آموزش/ارزیابی/استقرار، تشخیص انحراف، بازآموزی دوره‌ای و
پایش.

## ساختار

```
mlops/
├── mlflow_config.yaml            # پیکربندی ردیابی + رجیستری
├── serving_policy.yaml           # مسیریابی مدل، canary، rollback، human review
├── pipelines/
│   ├── train_pipeline.py         # آموزش + ارتقا به Staging
│   ├── eval_pipeline.py          # ارزیابی + دروازه‌های پذیرش + seed CAR
│   └── deploy_pipeline.py        # ارتقا به Production با canary
├── drift/
│   ├── drift_report.py           # PSI / Evidently → OPP feed
│   └── label_feedback.py         # بازخورد کارشناس → صف بازآموزی
├── airflow/dags/
│   └── retrain.py                # DAG هفتگی بازآموزی
├── feature_store/
│   └── schema.yaml               # طرح ویژگی‌های مشتق
└── monitoring/
    └── prometheus_alerts.yml     # هشدارهای QPM/SLO
```

## گردش کار کامل (CMMI ML5)

```
data (P-01/02) → train_pipeline (P-04) → eval_pipeline (P-05, acceptance gates)
   → drift_report (OPP) → deploy_decision (DAR) → deploy_pipeline (P-06, canary)
   → serving (P-07, serving_policy) → monitoring (QPM alerts)
   → human_review → label_feedback → next retrain (Airflow DAG)
```

## اجرا

```bash
# آموزش کامل + ثبت در رجیستری
python mlops/pipelines/train_pipeline.py --config configs/train.yaml

# ارزیابی + دروازه پذیرش
python mlops/pipelines/eval_pipeline.py --config configs/eval.yaml

# استقرار با canary (فقط اگر eval پاس شده باشد)
python mlops/pipelines/deploy_pipeline.py --canary

# تشخیص انحراف
python mlops/drift/drift_report.py \
  --reference metrics/baselines/reference_profile.json \
  --current   metrics/latest/current_profile.json
```

## رجیستری مدل

مدل‌ها در MLflow Model Registry با مراحل زیر مدیریت می‌شوند:
`None → Staging → Production → Archived`. ارتقا تنها پس از عبور از دروازه‌های
پذیرش BRS §6 انجام می‌شود (`eval_pipeline.py`).

## پایش و rollback

هشدارهای Prometheus در `monitoring/prometheus_alerts.yml` معیارهای SLO و ایمنی
را پوشش می‌دهند (زمان استنتاج، recall کلاس ترک، drift، 5xx). در صورت تخطی،
rollback خودکار به نسخه قبلی Production اجرا می‌شود (`serving_policy.yaml → rollback`).

## ایمنی (SRS §5.4)
مدل **هرگز** تصمیم ایمنی نهایی نمی‌گیرد؛ هر پیش‌بینی با `decision_support_only=True`
برچسب‌گذاری شده و نمونه‌های کم‌اطمینان یا ایمنی‌بحرانی به صف بازبینی کارشناس
routing می‌شوند.
