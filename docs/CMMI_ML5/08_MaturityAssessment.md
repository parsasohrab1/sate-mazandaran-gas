# ارزیابی بلوغ CMMI ML5 (Maturity Self-Assessment)

## ۱. روش
خودارزیابی بر اساس CMMI V2.0 Appraisal Method. هر Practice Area با وضعیت
«Fully Implemented (FI)»، «Largely Implemented (LI)»، «Partially Implemented (PI)»
یا «Not Implemented (NI)» ارزیابی می‌شود. بلوغ ۵ نیازمند FI/LI در همه
Practice Areas مربوط به ML5 است.

## ۲. نتیجه خودارزیابی

| Practice Area | سطح پیاده‌سازی | شواهد | وضعیت |
|---------------|----------------|-------|--------|
| OPF (§01) | FI | `docs/CMMI_ML5/01_OPF.md`, PAL, git history | ✅ |
| OPP (§02) | LI* | PPB در `metrics/baselines.json`, MLflow runs | ✅ (* خط‌مبنای تولید پس از داده واقعی نهایی) |
| OID (§03) | FI | `metrics/innovation_backlog.yaml`, OID-001..003 استقرار‌یافته | ✅ |
| CAR (§04) | FI | `metrics/car_log.yaml`, نمونه CAR-001 بسته | ✅ |
| QPM (§05) | LI | `metrics/qpm_log.yaml`, SPC charts | ✅ |
| PAL (§06) | FI | ساختار PAL در `docs/CMMI_ML5/06_PAL.md` + کد | ✅ |
| DAR (§07) | FI | `metrics/decision_log.yaml`, انتخاب مدل §4 | ✅ |
| GOV (Governance) | FI | BRS, SRS, LICENSE, RBAC | ✅ |
| II (Implementation Infrastructure) | FI | CI/CD, Docker, K8s, Helm | ✅ |

## ۳. وابستگی‌های بلوغ پایین‌تر (پیش‌نیاز)
| بلوغ | وضعیت | شواهد |
|------|-------|-------|
| ML2 (Managed) | ✅ | WBS در BRS، برنامه زمانی ۱۴ ماهه، ریسک‌ها |
| ML3 (Defined) | ✅ | PAL، OPF، فرآیندهای استاندارد P-01..P-09 |
| ML4 (Quantitatively Managed) | ✅ | OPP PPB/PPM، QPM SPC |

## ۴. شکاف‌ها و اقدام
| شکاف | اقدام | مهلت |
|------|-------|------|
| OPP PPB تولید از داده واقعی نهایی نشده | پس از intake داده Sate Mazandaran Gas، بازمحاسبه | فاز ۲ |
| SPC charts خودکار کامل نیست | تکمیل `scripts/spc_charts.py` | فاز ۲ |
| κ برچسب‌گذاری روی داده واقعی اندازه‌گیری نشده | پس از پایلوت برچسب‌گذاری | فاز ۲ |

## ۵. نتیجه‌گیری
ساختار فرآیندی محصول **مطابق با CMMI ML5** است. بلوغ کامل پس از دسترسی به
داده واقعی و تکمیل پایلوت تأیید نهایی می‌شود. در فاز اول (داده سینتتیک)،
تمام مکانیزم‌های ML5 (CAR, QPM, OPP, OID, DAR, PAL) قابل اجرا و اعتبارسنجی شده‌اند.

## ۶. برنامه بازرسی بعدی
- ۳ ماه پس از استقرار پایلوت: بازرسی کامل DAR/CAR/QPM.
- ۶ ماه پس از: ارزیابی رسمی بلوغ با نماینده کارفرما.
