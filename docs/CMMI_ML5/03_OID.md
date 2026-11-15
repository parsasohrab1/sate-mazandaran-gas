# OID — Organizational Innovation & Deployment (نوآوری و استقرار سازمانی)

## ۱. هدف
شناسایی نوآوری‌های داده‌محور که عملکرد فرآیندها را بهبود می‌بخشند، و استقرار
کنترل‌شده آن‌ها در سازمان.

## ۲. خط لوله نوآوری

```
جمع‌آوری ایده (CAR/داشبورد/ادبیات) → غربالگری (DAR) → پایلوت (QPM)
  → ارزیابی کمی (OPP) → استقرار (OPF/PAL) → اندازه‌گیری اثر
```

## ۳. ثبت ایده‌های نوآوری

ثبت‌نام در `metrics/innovation_backlog.yaml` با ساختار:
```yaml
- id: OID-001
  title: "افزودن Test-Time Augmentation به پیش‌بینی"
  source: CAR-003 (اختلاط کلاس crack با slag)
  expected_gain: "+1.5 pp recall_crack"
  status: pilot
  owner: ML engineer
  pilot_metric: recall_crack
  pilot_baseline: 0.91
  pilot_target: 0.93
```

## ۴. نمونه‌های نوآوری فعال پروژه

| شناسه | نوآوری | وضعیت | اثر اندازه‌گیری‌شده |
|-------|--------|-------|---------------------|
| OID-001 | Ensemble سه‌مدلی به جای تک‌مدل | استقرار‌شده | +۲-۴ pp F1-macro |
| OID-002 | CLAHE در پیش‌پردازش | استقرار‌شده | +۱٫۸ pp F1 روی سینتتیک |
| OID-003 | Focal loss به جای CE | استقرار‌شده | +۰٫۹ pp recall اقلیت |
| OID-004 | Test-Time Augmentation | پایلوت | در حال ارزیابی |
| OID-005 | Active learning روی تصاویر مرزی | پیشنهاد | — |

## ۵. معیار پذیرش استقرار نوآوری
- بهبود قابل اندازه‌گیری در یک PPB (OPP §2) با سطح اطمینان آماری معنادار.
- بدون تخطی از دروازه‌های ایمنی (SRS §5.4): recall_crack ≥ ۰٫۹۵.
- پذیرفته‌شده توسط DAR (§07) و ثبت در PAL (§06).

## ۶. ریسک استقرار و کنترل
- ریسک: کاهش پایداری پس از استقرار. کنترل: canary deployment در Helm
  (`devops/helm/weld-defect/values.yaml → canary.enabled`).
- ریسک: اثر منفی بر کلاس ایمنی‌بحرانی. کنترل: دروازه پذیرش در CI قبل از promote.
