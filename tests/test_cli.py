"""Tests for the CLI."""

from __future__ import annotations

import pytest

from weld_defect.cli import build_parser, main


def test_parser_has_subcommands():
    """build_parser returns a parser with train/evaluate/predict/serve subcommands."""
    parser = build_parser()
    # The parser should parse known subcommands without error.
    args = parser.parse_args(["predict", "--image", "test.png", "--model", "m.pt"])
    assert args.command == "predict"
    assert args.image == "test.png"
    assert args.model == "m.pt"


def test_parser_train_defaults():
    """train subcommand has correct defaults."""
    parser = build_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"
    assert args.config == "configs/train.yaml"


def test_parser_evaluate_defaults():
    """evaluate subcommand has correct defaults."""
    parser = build_parser()
    args = parser.parse_args(["evaluate"])
    assert args.command == "evaluate"
    assert args.config == "configs/eval.yaml"


def test_parser_serve_defaults():
    """serve subcommand has correct defaults."""
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 8000


def test_main_predict_calls_function(monkeypatch, tiny_checkpoint, tmp_path, sample_image_uint8):
    """main() dispatches to predict."""
    import cv2
    img_path = tmp_path / "weld.png"
    cv2.imwrite(str(img_path), sample_image_uint8)

    monkeypatch.setattr("sys.argv", ["weld_defect", "predict", "--image", str(img_path), "--model", str(tiny_checkpoint), "--architecture", "custom_cnn"])
    ret = main()
    assert ret == 0


def test_main_train_dispatches(monkeypatch, tmp_path):
    """main() dispatches to train."""
    called = {"n": 0}

    def fake_train(config_path):
        called["n"] += 1
        return {"f1_macro": 0.5, "checkpoint": str(tmp_path / "m.pt")}

    monkeypatch.setattr("weld_defect.cli.train", fake_train, raising=False)
    # Actually patch at the import location.
    import weld_defect.cli as cli_mod
    monkeypatch.setattr(cli_mod, "_cmd_train", lambda args: (fake_train(args.config), 0)[1])
    monkeypatch.setattr("sys.argv", ["weld_defect", "train", "--config", "configs/train.yaml"])
    ret = main()
    assert ret == 0


def test_main_evaluate_dispatches(monkeypatch):
    """main() dispatches to evaluate."""
    import weld_defect.cli as cli_mod

    def fake_eval(args):
        return 0

    monkeypatch.setattr(cli_mod, "_cmd_evaluate", fake_eval)
    monkeypatch.setattr("sys.argv", ["weld_defect", "evaluate", "--config", "configs/eval.yaml"])
    ret = main()
    assert ret == 0
