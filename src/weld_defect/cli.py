"""Command-line interface for weld defect detection.

Entry points (pyproject.toml [project.scripts]):
  weld-detect-train    — train a model
  weld-detect-eval     — evaluate a checkpoint
  weld-detect-predict  — single-image inference
  weld-detect-serve    — start the REST API

Usage:
  python -m weld_defect.cli train    --config configs/train.yaml
  python -m weld_defect.cli evaluate --config configs/eval.yaml
  python -m weld_defect.cli predict  --image weld.png --model models/ensemble_best.pt
  python -m weld_defect.cli serve    --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train
    metrics = train(args.config)
    print(f"\nTraining complete. Best val F1: {metrics.get('best_val_f1_macro', 0):.4f}")
    print(f"Checkpoint: {metrics.get('checkpoint')}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate
    result = evaluate(args.config)
    acceptance = result["acceptance"]
    print(f"\nAcceptance: {'PASS' if acceptance['pass'] else 'FAIL'}")
    for name, check in acceptance["checks"].items():
        status = "✓" if check["pass"] else "✗"
        print(f"  {status} {name}: {check['value']} (threshold: {check['threshold']})")
    return 0 if acceptance["pass"] else 1


def _cmd_predict(args: argparse.Namespace) -> int:
    from .predict import WeldPredictor
    predictor = WeldPredictor(
        checkpoint_path=args.model,
        architecture=args.architecture,
        device=args.device,
    )
    result = predictor.predict(args.image, image_name=Path(args.image).name)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import os
    os.environ.setdefault("API_HOST", args.host)
    os.environ.setdefault("API_PORT", str(args.port))
    if args.model:
        os.environ.setdefault("MODEL_CHECKPOINT", args.model)
    if args.architecture:
        os.environ.setdefault("MODEL_ARCHITECTURE", args.architecture)

    import uvicorn
    uvicorn.run(
        "weld_defect.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weld_defect",
        description="Sate Mazandaran Gas — AI Weld Defect Detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", default="configs/train.yaml")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a checkpoint")
    p_eval.add_argument("--config", default="configs/eval.yaml")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Single-image inference")
    p_pred.add_argument("--image", required=True, help="Path to image file")
    p_pred.add_argument("--model", default="models/ensemble_best.pt", help="Checkpoint path")
    p_pred.add_argument("--architecture", default="ensemble", help="Model architecture")
    p_pred.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    p_pred.set_defaults(func=_cmd_predict)

    p_serve = sub.add_parser("serve", help="Start the REST API")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--model", default=None, help="Checkpoint path")
    p_serve.add_argument("--architecture", default=None, help="Model architecture")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
