# راهنمای کاربری (User Guide)

این راهنما برای کاربر نهایی (کارشناس NDT، سرپرست بازرسی) و کاربر فنی
(مهندس ML/IT) نوشته شده است.

## ۱. نصب

### ۱.۱ پیش‌نیازها
- پایتون ≥ ۳.۱۰
- (پیشنهادی) GPU با CUDA برای آموزش سریع‌تر

### ۱.۲ نصب محیط
```bash
git clone https://github.com/parsasohrab1/sate-mazandaran-gas.git
cd sate-mazandaran-gas
make dev    # venv + نصب وابستگی‌های runtime + dev
```

## ۲. گردش کار پایه

### ۲.۱ تولید داده سینتتیک (برای اعتبارسنجی اولیه)
```bash
make data-synthetic    # ۱۲۰۰ تصویر در data/synthetic/
```
این مرحله خط لوله کامل را بدون نیاز به داده واقعی تأیید می‌کند.

### ۲.۲ دانلود داده واقعی
```bash
python scripts/download_data.py --list              # نمایش کاتالوگ
python scripts/download_data.py --sources roboflow  # دانلود یک منبع
```
برای داده داخلی شرکت گاز استان مازندران، فایل‌ها را در
`data/raw/sate_mazandaran_gas/` قرار دهید (مطابق `data/raw/README.md`).

### ۲.۳ آماده‌سازی داده
```bash
python scripts/prepare_data.py \
  --raw data/raw/sate_mazandaran_gas \
  --processed data/processed
```

### ۲.۴ آموزش
```bash
make train     #configs/train.yaml
```
بهترین چک‌پوینت در `models/ensemble_best.pt` و اجرا در MLflow ثبت می‌شود.

### ۲.۵ ارزیابی
```bash
make evaluate  #metrics/latest/metrics.json + confusion_matrix.png
```
دروازه‌های پذیرش BRS در `metrics/latest/error_analysis.json` گزارش می‌شوند.

### ۲.۶ خدمت‌رسانی
```bash
make serve     # uvicorn روی :8000
make mlflow    # MLflow UI روی :5000
```

## ۳. داشبورد کارشناس

داشبورد (`dashboard/index.html`) را در مرورگر باز کنید. کاربر می‌تواند:
1. یک تصویر جوش بارگذاری کند.
2. پیش‌بینی مدل + top-k احتمالات را ببیند.
3. با دکمه «تأیید کارشناس» یا «رد» تصمیم نهایی را ثبت کند.
4. معیارهای لحظه‌ای و ماتریس درهم‌ریختگی را مشاهده کند.

> **توجه ایمنی:** پیش‌بینی مدل یک پیشنهاد است. تصمیم نهایی ایمنی تنها پس
> از تأیید کارشناس واجد صلاحیت NDT معتبر است (SRS §5.4).

## ۴. خط فرمان

```bash
python -m weld_defect.cli predict --image weld.png --model models/ensemble_best.pt
python -m weld_defect.cli train --config configs/train.yaml
python -m weld_defect.cli evaluate --config configs/eval.yaml
python -m weld_defect.cli serve --host 0.0.0.0 --port 8000
```

## ۵. عیب‌یابی

| مشکل | راه‌حل |
|------|--------|
| `torch not installed` | `pip install -r requirements.txt` |
| مدل بارگذاری نشد (`/health` degraded) | مسیر `MODEL_CHECKPOINT` را بررسی کنید |
| خطای OCR فونت فارسی در نمودار | فونت Noto Sans SC نصب است (`/usr/share/fonts/`) |
| کندی استنتاج روی CPU | از `custom_cnn` به‌جای ensemble استفاده کنید |

## ۶. پرسش‌های رایج

**آیا خروجی مدل معتبر است؟** خیر — خروجی **پشتیبانی از کارشناس** است و نیاز
به تأیید NDT دارد.

**آیا داده واقعی من commit می‌شود؟** خیر — `.gitignore` داده‌ها را مستثنی می‌کند.

**چگونه مدل را بازگردانی کنم؟** از MLflow registry نسخه قبلی Production را
فعال کنید.
