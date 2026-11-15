# CI/CD — Continuous Integration & Delivery

این پوشه و همچنین `.github/workflows/` خطوط لوله یکپارچه‌سازی و تحویل مداوم
را برای سه سیستم فراهم می‌کند: GitHub Actions (اصلی)، GitLab CI و Jenkins.

## ۱. GitHub Actions (`.github/workflows/`)

| Workflow | محرک | شرح |
|----------|------|------|
| `ci.yml` | push/PR | lint (ruff/black/mypy) + pytest + coverage + pre-commit |
| `cd.yml` | push main / tag | build ایمیج → push GHCR → helm upgrade → smoke test |
| `security.yml` | push/PR + هفتگی | bandit SAST + trivy (fs+image) + gitleaks |
| `ml.yml` | push به src/configs + شبانه | اعتبارسنجی کامل خط لوله ML روی داده سینتتیک |
| `release.yml` | tag `v*` | ساخت GitHub Release + changelog |
| `docs.yml` | push به docs/ | build MkDocs و deploy به GitHub Pages |

## ۲. GitLab CI (`cicd/.gitlab-ci.yml`)
همان مراحل: lint → test → build → scan → deploy. برای استفاده، این فایل را
در ریشه مخزن GitLab به‌عنوان `.gitlab-ci.yml` قرار دهید.

## ۳. Jenkins (`cicd/Jenkinsfile`)
خط لوله اعلانی (declarative) با همان مراحل. نیاز به agent با Docker/Python/Helm
و credentials برای رجیستری و kubeconfig.

## جریان کلی
```
push/PR → CI (lint+test) → (main/tag) → build+push → scan → deploy (Helm) → smoke test
                                            ↑
                   ML validation (synthetic) + nightly drift/retrain trigger
```

## محیط‌های استقرار
- `dev` — Docker Compose محلی
- `staging` — خوشه K8s با Helm (canary)
- `production` — خوشه EKS/GKE با Terraform + Helm، TLS و HPA

## رازها (Secrets)
GitHub: `KUBECONFIG_B64`, `API_KEY`, `JWT_SECRET`, `INGRESS_HOST`.
GitLab/Jenkins: متغیرهای هم‌نام در CI/CD settings.
هرگز راز را در کد/commit ذخیره نکنید.
