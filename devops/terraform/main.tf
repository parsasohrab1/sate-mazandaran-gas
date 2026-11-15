# Terraform — production infrastructure for the weld-defect stack.
# Modules: VPC, EKS (or GKE), S3 artifact bucket, RDS (MLflow backend), IAM.
# This is a representative module skeleton; adapt provider/region to the
# Licensee's cloud (AWS shown; GCP/Azure equivalents noted in variables.tf).

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.23" }
    helm = { source = "hashicorp/helm", version = "~> 2.11" }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

# ---------- VPC ----------
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "weld-defect-vpc"
  cidr = "10.20.0.0/16"
  azs             = ["${var.region}a", "${var.region}b", "${var.region}c"]
  private_subnets = ["10.20.1.0/24", "10.20.2.0/24", "10.20.3.0/24"]
  public_subnets  = ["10.20.101.0/24", "10.20.102.0/24", "10.20.103.0/24"]
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_dns_hostnames = true
  tags = { Project = "sate-mazandaran-gas" }
}

# ---------- EKS ----------
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "weld-defect-${var.environment}"
  cluster_version = "1.29"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  eks_managed_node_groups = {
    inference = {
      min_size     = 2
      max_size     = 6
      desired_size = 2
      instance_types = var.gpu_enabled ? ["g5.xlarge"] : ["m6i.large"]
      labels = { role = "inference" }
    }
  }
  tags = { Project = "sate-mazandaran-gas" }
}

# ---------- S3 artifact bucket (MLflow artifacts + data archive) ----------
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "weld-defect-mlflow-${data.aws_caller_identity.current.account_id}-${var.environment}"
  tags   = { Project = "sate-mazandaran-gas", Purpose = "mlflow-artifacts" }
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

# ---------- RDS for MLflow backend store ----------
resource "aws_db_instance" "mlflow" {
  identifier        = "weld-defect-mlflow-${var.environment}"
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = "db.t3.medium"
  allocated_storage = 50
  db_name           = "mlflow"
  username          = "mlflow"
  password          = var.mlflow_db_password
  vpc_id            = module.vpc.vpc_id
  db_subnet_ids     = module.vpc.private_subnets
  skip_final_snapshot = var.environment != "production"
  storage_encrypted = true
  tags              = { Project = "sate-mazandaran-gas" }
}

# ---------- Helm release (deploys the app into EKS) ----------
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec {
      api_version = "client.authentication.api.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

resource "helm_release" "weld_defect" {
  name       = "weld-defect"
  repository = ""
  chart      = "${path.module}/../helm/weld-defect"
  namespace  = "weld-defect"
  create_namespace = true

  values = [
    file("${path.module}/../helm/weld-defect/values.yaml"),
    yamlencode({
      image = { repository = var.image_repo, tag = var.image_tag }
      ingress = { enabled = true, host = var.ingress_host }
      mlflow = { backendStoreUri = "postgresql://mlflow:${var.mlflow_db_password}@${aws_db_instance.mlflow.address}:5432/mlflow" }
    })
  ]
}

output "cluster_endpoint" { value = module.eks.cluster_endpoint }
output "mlflow_s3_bucket" { value = aws_s3_bucket.mlflow_artifacts.id }
output "mlflow_db_endpoint" { value = aws_db_instance.mlflow.address }
