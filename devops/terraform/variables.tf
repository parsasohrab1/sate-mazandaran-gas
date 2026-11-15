variable "region" {
  description = "Cloud region (AWS). For GCP/Azure, swap provider and this var."
  type        = string
  default     = "eu-central-1"
}

variable "environment" {
  description = "Deployment environment: dev | staging | production"
  type        = string
  default     = "production"
}

variable "gpu_enabled" {
  description = "Provision GPU nodes for inference (ensemble on GPU)."
  type        = bool
  default     = true
}

variable "image_repo" {
  description = "Container registry repository for the API image."
  type        = string
  default     = "registry/weld-defect-api"
}

variable "image_tag" {
  description = "Image tag to deploy."
  type        = string
  default     = "1.0.0"
}

variable "ingress_host" {
  description = "Ingress hostname for the API."
  type        = string
  default     = "weld-defect.satemazandaran.internal"
}

variable "mlflow_db_password" {
  description = "RDS password for the MLflow backend. Sensitive — set via tfvars or env, never commit."
  type        = string
  sensitive   = true
}

# Notes for GCP/Azure equivalents:
#   GCP:  replace aws provider with google; EKS → GKE; RDS → Cloud SQL; S3 → GCS.
#   Azure: replace with azurerm; EKS → AKS; RDS → Azure Database for PostgreSQL; S3 → Blob.
