# DevOps — Containerization, Orchestration, IaC

این پوشه زیرساخت استقرار محصول را نگهداری می‌کند: کانتینرها، compose،
مانیفست‌های Kubernetes، Terraform و Helm chart.

## ساختار

```
devops/
├── docker/
│   ├── Dockerfile.api          # سرویس FastAPI (multi-stage، non-root)
│   ├── Dockerfile.mlflow       # سرور MLflow
│   ├── Dockerfile.dashboard    # داشبورد ایستا (nginx-unprivileged)
│   └── .dockerignore
├── docker-compose.yml          # استک محلی/dev (api+mlflow+dashboard)
├── k8s/                        # مانیفست‌های خام Kubernetes
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   └── secret.yaml
├── terraform/                  # زیرساخت به‌عنوان کد (VPC/EKS/S3/RDS)
│   ├── main.tf
│   ├── variables.tf
│   └── terraform.tfvars.example
└── helm/weld-defect/           # Helm chart تولید
    ├── Chart.yaml
    ├── values.yaml
    └── templates/
        ├── _helpers.tpl
        ├── serviceaccount.yaml
        ├── configmap.yaml
        ├── secret.yaml
        ├── deployment.yaml
        ├── service.yaml
        ├── hpa.yaml
        └── ingress.yaml
```

## مسیرهای استقرار

### ۱. توسعه / پایلوت (Docker Compose)
```bash
cp .env.example .env
make docker-build && make docker-up
# API: localhost:8000 | MLflow: localhost:5000 | Dashboard: localhost:8080
```

### ۲. تولید (Kubernetes + Helm)
```bash
docker build -f devops/docker/Dockerfile.api -t registry/weld-defect-api:1.0.0 .
docker push registry/weld-defect-api:1.0.0
helm install weld-defect devops/helm/weld-defect \
  --set image.repository=registry/weld-defect-api \
  --set image.tag=1.0.0 \
  --set ingress.enabled=true \
  --set ingress.host=weld-defect.satemazandaran.internal
```

### ۳. زیرساخت ابری (Terraform)
```bash
cd devops/terraform
cp terraform.tfvars.example terraform.tfvars  # ویرایش مقادیر
terraform init && terraform plan && terraform apply
```
Terraform یک خوشه EKS + S3 (آرتیفکت) + RDS (MLflow backend) می‌سازد و Helm
chart را deploy می‌کند.

## امنیت
- همه ایمیج‌ها **non-root** اجرا می‌شوند.
- Secret‌ها از K8s Secret/Vault؛ هرگز در ConfigMap یا git.
- TLS با cert-manager برای Ingress.
- `.dockerignore` داده/لاگ/mlruns را از کانتکست بیلد مستثنی می‌کند.

## rollback
از MLflow registry نسخه قبلی Production فعال می‌شود
(`mlops/pipelines/deploy_pipeline.py`). در صورت تخطی از SLO، rollback خودکار
با `serving_policy.yaml → rollback` اجرا می‌شود.
