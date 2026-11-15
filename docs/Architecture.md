# معماری سیستم (System Architecture)

## ۱. نمای کلی

معماری محصول چندلایه و ماژولار است و بر سه محور استوار است: **کد منبع
(`src/weld_defect`)**، **زیرساخت ML/MLOps**، و **لایه خدمت‌رسانی (API +
داشبورد)**. طراحی به‌گونه‌ای است که هر لایه بدون وابستگی سخت به لایه دیگر
قابل آزمایش و جایگزینی باشد.

```
┌─────────────────────────────────────────────────────────────────┐
│  لایه ارائه (Presentation)                                      │
│   ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│   │ داشبورد HTML │   │ REST API     │   │ OpenAPI /docs       │  │
│   │ (ECharts)    │   │ (FastAPI)    │   │                     │  │
│   └──────┬───────┘   └──────┬───────┘   └────────────────────┘  │
└──────────┼─────────────────┼────────────────────────────────────┘
           │                 │
┌──────────┼─────────────────┼────────────────────────────────────┐
│  لایه منطق (Application)  │                                     │
│   ┌──────▼───────┐   ┌─────▼────────┐                          │
│   │ weld_defect. │   │ weld_defect. │                          │
│   │ predict      │   │ api          │                          │
│   └──────┬───────┘   └──────────────┘                          │
│          │                                                      │
│  لایه مدل (Model)                                               │
│   ┌──────▼──────────────────────────────────────┐              │
│   │ models: CustomCNN | ResNet50 | EfficientNet │              │
│   │        | Ensemble (softmax_avg)             │              │
│   └──────┬──────────────────────────────────────┘              │
│          │                                                      │
│  لایه داده (Data)                                               │
│   ┌──────▼──────────────────────────────────────┐              │
│   │ data: loader | preprocess | augment | splits│              │
│   └──────┬──────────────────────────────────────┘              │
└──────────┼─────────────────────────────────────────────────────┘
           │
┌──────────▼─────────────────────────────────────────────────────┐
│  لایه زیرساخت (Infrastructure)                                 │
│   MLflow Tracking+Registry │ Airflow │ Drift (Evidently)        │
│   Docker │ Kubernetes │ Helm │ Terraform │ CI/CD                │
└────────────────────────────────────────────────────────────────┘
```

## ۲. اجزا و مسئولیت‌ها

| لایه | مؤلفه | فایل/ماژول | مسئولیت |
|------|-------|-----------|---------|
| داده | PreprocessPipeline | `data/preprocess.py` | denoise+CLAHE+resize+normalize |
| داده | Augmenter | `data/augment.py` | Albumentations/numpy |
| داده | WeldDataset | `data/loader.py` | بارگذاری + split stratified-group |
| مدل | CustomCNN/ResNet/EfficientNet/Ensemble | `models/*.py` | معماری‌های طبقه‌بندی |
| مدل | factory | `models/factory.py` | ساخت/ذخیره/بارگذاری |
| منطق | train | `train.py` | حلقه آموزش + MLflow |
| منطق | evaluate | `evaluate.py` | معیارها + دروازه پذیرش |
| منطق | predict | `predict.py` | استنتاج تک‌تصویر |
| ارائه | FastAPI app | `api/app.py` | نقاط انتهایی REST |
| ارائه | Dashboard | `dashboard/index.html` | UI راست‌چین |
| MLOps | MLflow | `mlops/mlflow_config.yaml` | ردیابی + رجیستری |
| MLOps | Airflow DAG | `mlops/airflow/dags/retrain.py` | بازآموزی دوره‌ای |
| MLOps | Drift | `mlops/drift/drift_report.py` | تشخیص انحراف |

## ۳. جریان داده (Data Flow)

### ۳.۱ آموزش
1. `download_data.py` → `data/raw/<source>/`
2. `prepare_data.py` → `data/processed/{images,train,val,test}.csv`
3. `weld_defect.train` → `models/ensemble_best.pt` + MLflow run
4. `weld_defect.evaluate` → `metrics/latest/`

### ۳.۲ استنتاج
1. کاربر تصویر را در داشبورد/API بارگذاری می‌کند.
2. `api.app` → `predict.predict` → مدل → پاسخ JSON با `decision_support_only=True`.
3. کارشناس نتیجه را در داشبورد تأیید/رد می‌کند (لاگ برای CAR).

## ۴. تصمیمات کلیدی معماری (ADR)

### ADR-1 — Ensemble به جای تک‌مدل
**زمینه:** تنوع عیوب (بافت vs هندسی) باعث خطاهای همبسته در یک مدل می‌شود.
**تصمیم:** ensemble از سه مدل با بایاس متنوع (ResNet بافت، EfficientNet هندسی،
CustomCNN سبک) با میانگین softmax.
**پیامدها:** افزایش ~۲-۴ نقطه F1-macro، افزایش کم تأخیر.

### ADR-2 — Stratified-group split
**تصمیم:** تقسیم بر اساس گروه «قطعه جوش» برای جلوگیری از نشت داده.
**پیامدها:** برآورد واقع‌بینانه‌تر تعمیم‌پذیری.

### ADR-3 — Decision-support، نه خودکار
**تصمیم:** خروجی مدل هرگز به‌تنهایی تصمیم ایمنی نیست؛ پرچم
`decision_support_only` و الزام تأیید کارشناس در همه لایه‌ها.

### ADR-4 — MLflow برای ردیابی و رجیستری
**تصمیم:** یک منبع حقیقت برای آزمایش‌ها، مدل‌ها و نسخه‌ها.

## ۵. ملاحظات مقیاس‌پذیری

- سرویس API به‌صورت stateless deploy می‌شود؛ چند نمونه پشت یک Service.
- استنتاج batch از یک صف (در استقرار K8s با HPA) بهره می‌برد.
- مدل سنگین ensemble روی GPU؛ مدل سبک CustomCNN روی CPU ایستگاه بازرسی.

## ۶. ملاحظات امنیتی

- داده واقعی هرگز در git/MLflow artifacts ذخیره نمی‌شود.
- احراز هویت API Key + RBAC؛ TLS داخلی.
- لاگ دسترسی برای حسابرسی (OPP/QPM).

## ۷. رابط با دارایی‌های فرآیندی CMMI ML5

هر جزء معماری به دارایی فرآیندی مرتبط است:
- `train.py/evaluate.py` ↔ QPM (کنترل کمی زیرپروسه آموزش)
- `drift` ↔ OPP (پایه عملکرد فرآیند)
- `error_analysis` ↔ CAR (تحلیل علل خطا)
- `configs/` ↔ PAL (کتابخانه دارایی فرآیند)
