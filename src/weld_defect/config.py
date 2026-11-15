"""Typed configuration loader.

Reads the YAML files in ``configs/`` and resolves environment-variable
references of the form ``${VAR,default}``. The loader is deliberately
dependency-light (only ``yaml`` + ``os``) so it works in CI before heavy ML
deps are installed.

Typical usage::

    from weld_defect.config import load_config, TrainConfig
    cfg = load_config("configs/train.yaml", TrainConfig)
    print(cfg.data.class_names)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, TypeVar

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?:,([^}]*))?\}")

T = TypeVar("T")


# --------------------------------------------------------------------------- #
#  Dataclasses mirroring configs/train.yaml & configs/eval.yaml
# --------------------------------------------------------------------------- #
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
    class_names: tuple[str, ...] = (
        "no_defect",
        "porosity",
        "crack",
        "lack_of_fusion",
        "undercut",
        "slag_inclusion",
    )
    class_weights: tuple[float, ...] = (1.0, 1.4, 1.6, 1.5, 1.2, 1.3)

    @property
    def idx_to_class(self) -> dict[int, str]:
        return dict(enumerate(self.class_names))


@dataclass
class AugmentConfig:
    enabled: bool = True
    policy: str = "albumentations"
    ops: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelConfig:
    architecture: str = "ensemble"
    members: list[str] = field(default_factory=lambda: ["resnet50", "efficientnet_b0", "custom_cnn"])
    pretrained: bool = True
    pooling: str = "avg"
    dropout: float = 0.3
    ensemble_strategy: str = "softmax_avg"


@dataclass
class TrainingLoopConfig:
    epochs: int = 40
    batch_size: int = 32
    num_workers: int = 4
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine_warm"
    warmup_epochs: int = 3
    mixed_precision: bool = True
    gradient_clip: float = 1.0


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 7
    metric: str = "val_f1_macro"
    mode: str = "max"


@dataclass
class LossConfig:
    name: str = "focal_loss"
    gamma: float = 2.0
    alpha: tuple[float, ...] = (1.0, 1.4, 1.6, 1.5, 1.2, 1.3)


@dataclass
class MetricsConfig:
    primary: str = "f1_macro"
    also: list[str] = field(default_factory=list)
    confusion_matrix: bool = True
    per_class_report: bool = True


@dataclass
class ArtifactsConfig:
    checkpoint_dir: str = "models/checkpoints"
    export_dir: str = "models"
    log_dir: str = "logs"


@dataclass
class ExperimentConfig:
    name: str = "weld-defect-ensemble-v1"
    description: str = ""
    seed: int = 42
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "weld-defect-detection"
    mlflow_model_name: str = "weld-defect-ensemble"


@dataclass
class TrainConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)


# --------------------------------------------------------------------------- #
#  Loader
# --------------------------------------------------------------------------- #
def _resolve_env(value: Any) -> Any:
    """Recursively replace ``${VAR,default}`` tokens with environment values."""
    if isinstance(value, str):
        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else "")

        return _ENV_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _to_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    """Best-effort conversion of a nested dict into a dataclass instance.

    With ``from __future__ import annotations`` (used here), field ``.type``
    values are strings, so we resolve nested dataclasses by name via a registry
    built from the module's dataclass definitions.
    """
    if not hasattr(cls, "__dataclass_fields__"):
        return data  # type: ignore[return-value]
    # Registry of nested dataclass types by their class name.
    nested_registry = {
        name: obj for name, obj in globals().items()
        if isinstance(obj, type) and hasattr(obj, "__dataclass_fields__")
    }
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for name, val in data.items():
        if name not in field_types:
            continue
        ftype = field_types[name]
        # Resolve string annotations to the actual dataclass if it is one.
        resolved = nested_registry.get(ftype) if isinstance(ftype, str) else (
            ftype if isinstance(ftype, type) and hasattr(ftype, "__dataclass_fields__") else None
        )
        if resolved is not None and isinstance(val, dict):
            kwargs[name] = _to_dataclass(resolved, val or {})
        elif name in ("class_names", "class_weights", "alpha") and isinstance(val, list):
            kwargs[name] = tuple(val)
        else:
            kwargs[name] = val
    return cls(**kwargs)  # type: ignore[call-arg]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file with env-var substitution."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _resolve_env(raw)


def load_config(path: str | Path, cls: type[T] = TrainConfig) -> T:
    """Load ``path`` into a dataclass instance of ``cls``."""
    data = load_yaml(path)
    return _to_dataclass(cls, data)


def as_dict(cfg: Any) -> dict[str, Any]:
    """Recursively convert a config dataclass tree to a plain dict (for logging)."""
    return asdict(cfg)
