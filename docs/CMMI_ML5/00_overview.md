# دارایی‌های فرآیندی CMMI ML5 — نمای کلی

این پوشه دارایی‌های فرآیندی برای **سطح بلوغ ۵ (Optimizing)** مدل CMMI V2.0 را
برای محصول تشخیص عیوب جوش خطوط لوله نگهداری می‌کند. بلوغ ۵ به معنای آن است
که سازمان نه‌تنها فرآیندهای ثابت‌شده دارد (ML3) و معیارهای کمی تعریف‌شده
(ML4)، بلکه به‌صورت **مداوم و داده‌محور** فرآیندها را بهبود می‌دهد (ML5).

## چرا ML5 برای این پروژه؟

پروژه ماهیت ایمنی‌حساس دارد (خطوط لوله گاز). بلوغ ۵ تضمین می‌کند که:
- خطاهای مدل به‌صورت سیستماتیک ریشه‌یابی می‌شوند (CAR)؛
- بهبودها بر پایه داده کمی انجام می‌شوند (OPP/QPM)؛
- نوآوری‌ها به‌صورت کنترل‌شده استقرار می‌یابند (OID)؛
- تصمیمات کلیدی با تحلیل کمی گرفته می‌شوند (DAR).

## نگاشت Practice Areas به دارایی‌ها

| Practice Area (CMMI V2.0) | سند | شرح |
|---------------------------|-----|------|
| OPF — Organizational Process Focus | [01_OPF.md](01_OPF.md) | متمرکزسازی و نگهداری فرآیندهای سازمانی |
| OPP — Organizational Process Performance | [02_OPP.md](02_OPP.md) | پایه‌های عملکرد فرآیند و خط‌مبنای کمی |
| OID — Organizational Innovation & Deployment | [03_OID.md](03_OID.md) | شناسایی و استقرار نوآوری‌های داده‌محور |
| CAR — Causal Analysis & Resolution | [04_CAR.md](04_CAR.md) | تحلیل علل ریشه‌ای خطاها |
| QPM — Quantitative Process Management | [05_QPM.md](05_QPM.md) | مدیریت کمی زیرپروسه‌ها |
| PAL — Process Asset Library | [06_PAL.md](06_PAL.md) | کتابخانه دارایی‌های فرآیند |
| DAR — Decision Analysis & Resolution | [07_DAR.md](07_DAR.md) | تحلیل تصمیم‌گیری ساختاریافته |
| ارزیابی بلوغ | [08_MaturityAssessment.md](08_MaturityAssessment.md) | خودارزیابی ML5 و نقشه راه |
| تعاریف معیارها | [09_MetricsDefinitions.md](09_MetricsDefinitions.md) | تعاریف کمّی همه معیارها |

## اصل حاکم

هر دارایی فرآیند در این پوشه **قابل اجرا** است: نه فقط توصیفی، بلکه با
ارجاع به کد/پیکربندی که آن را پیاده می‌کند. برای نمونه:
- QPM به `configs/eval.yaml` و `metrics/latest/` ارجاع می‌دهد؛
- CAR به `error_analysis.json` و `misclassified/` ارجاع می‌دهد؛
- OPP به `data_profile.json` و `labeling_agreement.json` ارجاع می‌دهد؛
- DAR به `configs/model_catalog.yaml` ارجاع می‌دهد.

این پیوند میان سند و کد، شرط لازم برای بلوغ ۵ است.
