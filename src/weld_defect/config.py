"""Configuration loader: YAML → dataclass with env-var substitution.

Reads ``configs/train.yaml`` and ``configs/eval.yaml`` (the single source of
truth for the pipeline). Supports the ``${ENV_VAR,default}`` placeholder
syntax used in those files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Match ${VAR} or ${VAR,default} (default may contain colons, e.g. file:./x).
_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?:,([^}]*))?\}")


def _substitute_env(value: Any) -> Any:
    """Recursively substitute ${VAR,default} placeholders in a parsed YAML object."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else m.group(0))
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file with env-var substitution."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _substitute_env(raw)


# --- dataclasses mirroring configs/train.yaml ---


@dataclass
class ExperimentConfig:
    name: str = "weld-defect-ensemble-v1"
    description: str = ""
    seed: int = 42
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "weld-defect-detection"
    mlflow_model_name: str = "weld-defect-ensemble"


@dataclass
class DataConfig:
    root: str = "./data"
    train_split: str = "data/processed/train.csv"
    val_split: str = "data/processed/val.csv"
    test_split: str = "data/processed/test.csv"
    image_dir: str = "data/processed/images"
    input_size: int = 224
    channels: int = 1
    num_classes: int = 6
    class_names: list = field(default_factory=lambda: [
        "no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion",
    ])
    class_weights: list = field(default_factory=lambda: [1.0, 1.4, 1.6, 1.5, 1.2, 1.3])


@dataclass
class AugmentationConfig:
    enabled: bool = True
    policy: str = "albumentations"
    ops: list = field(default_factory=list)


@dataclass
class ModelConfig:
    architecture: str = "ensemble"
    members: list = field(default_factory=lambda: ["resnet50", "efficientnet_b0", "custom_cnn"])
    pretrained: bool = True
    pooling: str = "avg"
    dropout: float = 0.3
    ensemble_strategy: str = "softmax_avg"


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 7
    metric: str = "val_f1_macro"
    mode: str = "max"


@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 32
    num_workers: int = 4
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine_warm"
    warmup_epochs: int = 3
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    mixed_precision: bool = True
    gradient_clip: float = 1.0


@dataclass
class LossConfig:
    name: str = "focal_loss"
    gamma: float = 2.0
    alpha: list = field(default_factory=lambda: [1.0, 1.4, 1.6, 1.5, 1.2, 1.3])


@dataclass
class MetricsConfig:
    primary: str = "f1_macro"
    also: list = field(default_factory=list)
    confusion_matrix: bool = True
    per_class_report: bool = True


@dataclass
class ArtifactsConfig:
    checkpoint_dir: str = "models/checkpoints"
    export_dir: str = "models"
    export_formats: list = field(default_factory=lambda: ["pt"])
    log_dir: str = "logs"


@dataclass
class TrainConfig:
    """Top-level container for configs/train.yaml."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    raw: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return self.data.num_classes

    @property
    def class_names(self) -> list:
        return self.data.class_names


def _build_train_config(raw: dict) -> TrainConfig:
    """Build TrainConfig from a parsed YAML dict, filling defaults."""
    cfg = TrainConfig(raw=raw)
    if "experiment" in raw:
        cfg.experiment = ExperimentConfig(**raw["experiment"])
    if "data" in raw:
        d = raw["data"]
        # Don't pass extra keys; filter to known fields.
        known = {k: v for k, v in d.items() if k in DataConfig.__dataclass_fields__}
        cfg.data = DataConfig(**known)
    if "augmentation" in raw:
        cfg.augmentation = AugmentationConfig(**raw["augmentation"])
    if "model" in raw:
        m = raw["model"]
        known = {k: v for k, v in m.items() if k in ModelConfig.__dataclass_fields__}
        cfg.model = ModelConfig(**known)
    if "training" in raw:
        t = raw["training"]
        es = t.pop("early_stopping", None)
        known = {k: v for k, v in t.items() if k in TrainingConfig.__dataclass_fields__}
        cfg.training = TrainingConfig(**known)
        if es:
            cfg.training.early_stopping = EarlyStoppingConfig(**es)
    if "loss" in raw:
        cfg.loss = LossConfig(**raw["loss"])
    if "metrics" in raw:
        cfg.metrics = MetricsConfig(**raw["metrics"])
    if "artifacts" in raw:
        a = raw["artifacts"]
        known = {k: v for k, v in a.items() if k in ArtifactsConfig.__dataclass_fields__}
        cfg.artifacts = ArtifactsConfig(**known)
    return cfg


def load_config(path: str | Path = "configs/train.yaml") -> TrainConfig:
    """Load and parse configs/train.yaml into a TrainConfig dataclass."""
    raw = load_yaml(path)
    return _build_train_config(raw)


# --- eval config ---


@dataclass
class EvalModelConfig:
    checkpoint: str = "models/ensemble_best.pt"
    architecture: str = "ensemble"
    input_size: int = 224
    num_classes: int = 6


@dataclass
class EvalDataConfig:
    test_split: str = "data/processed/test.csv"
    image_dir: str = "data/processed/images"
    batch_size: int = 64
    num_workers: int = 4
    class_names: list = field(default_factory=lambda: [
        "no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion",
    ])


@dataclass
class ThresholdsConfig:
    no_defect: float = 0.50
    porosity: float = 0.45
    crack: float = 0.55
    lack_of_fusion: float = 0.50
    undercut: float = 0.50
    slag_inclusion: float = 0.48


@dataclass
class EvalEvalConfig:
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    top_k: int = 1
    calibration: str = "temperature_scaling"


@dataclass
class EvalMetricsConfig:
    primary: str = "f1_macro"
    also: list = field(default_factory=list)
    confusion_matrix: bool = True
    per_class_report: bool = True
    roc_curves: bool = True


@dataclass
class AcceptanceConfig:
    min_f1_macro: float = 0.85
    min_recall_crack: float = 0.95
    min_precision_no_defect: float = 0.90
    max_inference_ms: float = 120
    require_expert_review: bool = True


@dataclass
class EvalArtifactsConfig:
    report_dir: str = "metrics/latest"
    mlflow_run: str = "link"
    save_misclassified: bool = True
    error_analysis_json: str = "metrics/latest/error_analysis.json"


@dataclass
class EvalConfig:
    """Top-level container for configs/eval.yaml."""
    model: EvalModelConfig = field(default_factory=EvalModelConfig)
    data: EvalDataConfig = field(default_factory=EvalDataConfig)
    eval: EvalEvalConfig = field(default_factory=EvalEvalConfig)
    metrics: EvalMetricsConfig = field(default_factory=EvalMetricsConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    artifacts: EvalArtifactsConfig = field(default_factory=EvalArtifactsConfig)
    raw: dict = field(default_factory=dict)

    @property
    def class_names(self) -> list:
        return self.data.class_names


def load_eval_config(path: str | Path = "configs/eval.yaml") -> EvalConfig:
    """Load and parse configs/eval.yaml into an EvalConfig dataclass."""
    raw = load_yaml(path)
    cfg = EvalConfig(raw=raw)
    if "model" in raw:
        known = {k: v for k, v in raw["model"].items() if k in EvalModelConfig.__dataclass_fields__}
        cfg.model = EvalModelConfig(**known)
    if "data" in raw:
        known = {k: v for k, v in raw["data"].items() if k in EvalDataConfig.__dataclass_fields__}
        cfg.data = EvalDataConfig(**known)
    if "eval" in raw:
        e = raw["eval"]
        th = e.pop("thresholds", None)
        known = {k: v for k, v in e.items() if k in EvalEvalConfig.__dataclass_fields__}
        cfg.eval = EvalEvalConfig(**known)
        if th:
            cfg.eval.thresholds = ThresholdsConfig(**th)
    if "metrics" in raw:
        known = {k: v for k, v in raw["metrics"].items() if k in EvalMetricsConfig.__dataclass_fields__}
        cfg.metrics = EvalMetricsConfig(**known)
    if "acceptance" in raw:
        known = {k: v for k, v in raw["acceptance"].items() if k in AcceptanceConfig.__dataclass_fields__}
        cfg.acceptance = AcceptanceConfig(**known)
    if "artifacts" in raw:
        known = {k: v for k, v in raw["artifacts"].items() if k in EvalArtifactsConfig.__dataclass_fields__}
        cfg.artifacts = EvalArtifactsConfig(**known)
    return cfg
