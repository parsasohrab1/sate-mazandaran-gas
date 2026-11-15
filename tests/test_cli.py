"""CLI dispatcher tests — exercises argument parsing without heavy execution."""
from __future__ import annotations

import pytest

from weld_defect.cli import main, train, predict, serve, evaluate


def test_main_requires_subcommand(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_train_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        train(["--help"])
    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert "--config" in out


def test_predict_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        predict(["--help"])
    assert exc.value.code == 0


def test_serve_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        serve(["--help"])
    assert exc.value.code == 0


def test_evaluate_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        evaluate(["--help"])
    assert exc.value.code == 0


def test_main_unknown_subcommand_exits(capsys):
    with pytest.raises(SystemExit):
        main(["frobnicate"])

