# راهنمای استقرار (Deployment Guide)

این سند سه مسیر استقرار را پوشش می‌دهد: **Docker Compose** (توسعه/پایلوت)،
**Kubernetes + Helm** (تولید)، و **Terraform** (زیرساخت ابری). همه مانیفست‌ها
در `devops/` نگهداری می‌شوند.

## ۱. پیش‌نیازها

- Docker ≥ ۲۴، Docker Compose v2
- (تولید) kubectl ≥ ۱.۲۸، Helm ۳، یک خوشه Kubernetes
- (اختیاری) Terraform ≥ ۱.۶ برای زیرساخت

## ۲. مسیر ۱ — Docker Compose (سریع‌ترین)

```bash
cp .env.example .env   # ویرایش مقادیر
make docker-build
make docker-up         # API:8000 + MLflow:5000 + Dashboard:8080
make docker-down
```

سرویس‌ها (مشاهده `devops/docker-compose.yml`):
- `api` — FastAPI روی پورت ۸۰۰۰
- `mlflow` — ردیابی و رجیستری روی پورت ۵۰۰۰
- `dashboard` — داشبورد ایستا روی پورت ۸۰۸۰

## ۳. مسیر ۲ — Kubernetes + Helm (تولید)

### ۳.۱ بیلد و push ایمیج
```bash
docker build -f devops/docker/Dockerfile.api -t registry/weld-defect-api:1.0.0 .
docker push registry/weld-defect-api:1.0.0
```

### ۳.۲ استقرار با Helm
```bash
helm install weld-defect devops/helm/weld-defect \
  --set image.repository=registry/weld-defect-api \
  --set image.tag=1.0.0 \
  --set ingress.enabled=true
```

Chart در `devops/helm/weld-defect/` شامل: Deployment، Service، HPA،
Ingress، ConfigMap، Secret، و ServiceAccount. مقادیر قابل تنظیم در `values.yaml`.

### ۳.۳ مانیفست‌های خام (بدون Helm)
در `devops/k8s/` موجود است: `namespace.yaml`، `deployment.yaml`،
`service.yaml`، `hpa.yaml`، `ingress.yaml`، `configmap.yaml`، `secret.yaml`.

## ۴. مسیر ۳ — Terraform (زیرساخت)

```bash
cd devops/terraform
terraform init
terraform plan -var-file=prod.tfvars
terraform apply -var-file=prod.tfvars
```

ماژول‌ها: VPC، EKS/GKE، S3 (آرشیو)، RDS (MLflow backend)، IAM. متغیرها در
`variables.tf` و `terraform.tfvars.example`.

## ۵. متغیرهای محیطی کلیدی

| متغیر | پیش‌فرض | شرح |
|-------|---------|-----|
| `MODEL_CHECKPOINT` | `models/ensemble_best.pt` | مسیر چک‌پوینت |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | URI ردیابی |
| `API_KEY` | — | کلید احراز هویت (الزامی برای تولید) |
| `API_MAX_UPLOAD_MB` | ۱۰ | محدودیت آپلود |
| `APP_LOG_LEVEL` | INFO | سطح لاگ |

## ۶. CI/CD

استقرار تولید از GitHub Actions `cicd/cd.yml` اجرا می‌شود: build → scan →
push → helm upgrade. جزئیات در [`cicd/`](../cicd/) و راهنمای CI/CD.

## ۷. پایش و بازیابی

- Healthcheck: `GET /health` (liveness/readiness probe در K8s).
- لاگ‌ها به stdout؛ جمع‌آوری با Loki/ELK (پیکربندی در Helm values).
- بازگردانی مدل: rollback از MLflow registry به نسخه قبلی Production.

## ۸. امنیت استقرار

- ایمیج non-root (دایرکتوری `devops/docker/`).
- Secret‌ها از K8s Secret یا Vault؛ هرگز در ConfigMap.
- TLS با cert-manager برای Ingress.
