# Tests

مجموعه تست‌های pytest برای پوشش کامل ماژول‌ها. تست‌ها طوری طراحی شده‌اند که
بدون torch هم قابل اجرا باشند (بخش‌های ML با `importorskip` نادیده گرفته
می‌شوند) تا CI سریع اجرا شود.

## اجرا

```bash
# نصب وابستگی‌های dev
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# تست سریع (بدون slow/gpu/integration)
pytest -m "not slow and not gpu and not integration"

# تست کامل با پوشش
pytest --cov=weld_defect --cov-report=html
```

## ساختار

| فایل | پوشش | نیاز به torch؟ |
|------|------|----------------|
| `conftest.py` | fixtures + path setup | خیر |
| `test_config.py` | بارگذاری YAML + env substitution | خیر |
| `test_preprocess.py` | پایپ‌لاین پیش‌پردازش | خیر (بجز to_tensor) |
| `test_augment.py` | augmenter (numpy fallback) | خیر |
| `test_data_loader.py` | stratified-group split | خیر |
| `test_metrics.py` | معیارها + دروازه پذیرش | خیر |
| `test_synthetic.py` | تولید داده سینتتیک | خیر |
| `test_models.py` | معماری‌ها + save/load | بله |
| `test_api.py` | نقاط انتهایی REST | بله (پیش‌بینی) |
| `test_predict.py` | استنتاج تک‌تصویر | بله |
| `test_e2e.py` | خط لوله کامل (integration/slow) | بله |

## نشانه‌ها (Markers)
- `slow` — طولانی (مثلاً آموزش کامل)
- `gpu` — نیاز به GPU
- `integration` — نیاز به خط لوله کامل

## پوشش
هدف پوشش ≥ ۸۰٪ (تنظیم در `pyproject.toml → [tool.coverage.report] fail_under`).
