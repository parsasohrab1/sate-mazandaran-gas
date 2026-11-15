"""Command-line interface.

Subcommands (registered as console_scripts in pyproject.toml):
    weld-detect-train     [config]
    weld-detect-eval      [config]
    weld-detect-predict   --image IMG [--model PATH]
    weld-detect-serve     [--host H --port P]

Each subcommand is also importable (``from weld_defect.cli import train``) so it
can be invoked from notebooks or Airflow operators.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def train(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="weld-detect-train", description="Train the weld-defect ensemble.")
    p.add_argument("--config", default="configs/train.yaml")
    args = p.parse_args(argv)
    from weld_defect.train import train as _train
    out = _train(args.config)
    print(json.dumps({"best_val_f1_macro": out["best_val_f1_macro"], "checkpoint": out["checkpoint"]}, indent=2))
    return 0


def evaluate(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="weld-detect-eval", description="Evaluate a trained model on the test split.")
    p.add_argument("--config", default="configs/eval.yaml")
    args = p.parse_args(argv)
    from weld_defect.evaluate import evaluate as _eval
    out = _eval(args.config)
    print(json.dumps({"f1_macro": out["metrics"]["f1_macro"], "acceptance_passed": out["acceptance"]["passed"],
                      "inference_ms_mean": out["inference_ms_mean"]}, indent=2))
    return 0


def predict(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="weld-detect-predict", description="Predict on a single image.")
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="models/ensemble_best.pt")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args(argv)
    from weld_defect.predict import predict as _predict
    out = _predict(args.image, checkpoint=args.model, top_k=args.top_k)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def serve(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="weld-detect-serve", description="Serve the REST API with uvicorn.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)
    import uvicorn
    uvicorn.run("weld_defect.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Top-level dispatcher: ``python -m weld_defect.cli <subcommand> [...]``."""
    parser = argparse.ArgumentParser(prog="weld_defect")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(fn=train)
    sub.add_parser("evaluate").set_defaults(fn=evaluate)
    sub.add_parser("predict").set_defaults(fn=predict)
    sub.add_parser("serve").set_defaults(fn=serve)
    args, rest = parser.parse_known_args(argv)
    return args.fn(rest)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
