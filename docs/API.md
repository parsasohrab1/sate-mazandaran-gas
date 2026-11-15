# راهنمای API (REST)

سرویس پیش‌بینی روی FastAPI پیاده‌سازی شده و مستندات OpenAPI خودکار در
`/docs` در دسترس است. پایه کد: `src/weld_defect/api/app.py`.

## ۱. آدرس پایه و احراز هویت

| فیلد | مقدار |
|------|-------|
| آدرس پایه (پیش‌فرض) | `http://<host>:8000` |
| احراز هویت | API Key در هدر `X-API-Key` |
| قالب | JSON (multipart برای آپلود تصویر) |
| محدودیت آپلود | `API_MAX_UPLOAD_MB` (پیش‌فرض ۱۰ مگابایت) |

> **امنیت:** نقاط انتهایی نوشتنی نیاز به API Key دارند. کلید را از متغیر
> محیطی `API_KEY` تنظیم کنید و هرگز در git ذخیره نکنید.

## ۲. نقاط انتهایی

### ۲.۱ `GET /`
اطلاعات سرویس.
```bash
curl -s http://localhost:8000/
```
```json
{"service":"sate-mazandaran-gas-weld-defect-detection","version":"1.0.0",
 "docs":"/docs","decision_support_only":true}
```

### ۲.۲ `GET /health`
بررسی زنده‌بودن و وضعیت بارگذاری مدل.
```json
{"status":"ok","model_loaded":true,"device":"cpu","version":"1.0.0"}
```

### ۲.۳ `POST /predict`
پیش‌بینی روی یک تصویر (multipart).
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "X-API-Key: $API_KEY" \
  -F "file=@weld.png"
```
```json
{
  "image": "weld.png",
  "predicted_class": "porosity",
  "confidence": 0.872,
  "top_k": [
    {"class_name":"porosity","probability":0.872},
    {"class_name":"slag_inclusion","probability":0.081},
    {"class_name":"no_defect","probability":0.034}
  ],
  "inference_ms": 41.2,
  "model": "ensemble",
  "decision_support_only": true
}
```

### ۲.۴ `POST /predict/batch`
پیش‌بینی دسته‌ای (حداکثر ۶۴ فایل).
```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: $API_KEY" \
  -F "files=@a.png" -F "files=@b.png"
```
```json
{"results":[...],"count":2,"avg_inference_ms":38.7}
```

## ۳. کدهای خطا

| کد | معنی |
|----|------|
| 200 | موفق |
| 413 | آپلود بیش از حد بزرگ |
| 422 | قالب ورودی نامعتبر |
| 503 | مدل بارگذاری نشده |

## ۴. یکپارچه‌سازی پایتون

```python
import httpx
r = httpx.post("http://localhost:8000/predict",
               headers={"X-API-Key": API_KEY},
               files={"file": open("weld.png","rb")})
print(r.json()["predicted_class"])
```

## ۵. ملاحظات ایمنی

هر پاسخ حامل `decision_support_only: true` است. این یک الزام SRS §5.4 است:
خروجی باید توسط کارشناس NDT تأیید شود. در داشبورد، دکمه «تأیید/رد کارشناس»
این مرحله را ثبت می‌کند.
