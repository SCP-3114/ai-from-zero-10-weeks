import json

import pytest
import torch

from week06_evaluation.report import confusion_matrix
from week07_robustness.noise import add_noise
from week09_experiments.compare import collect_runs


def test_confusion_matrix_orientation():
    matrix = confusion_matrix(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1]), 2)
    assert matrix.tolist() == [[1, 1], [0, 1]]


def test_noise_zero_and_negative():
    image = torch.tensor([[-1.0, 0.0, 1.0]])
    assert torch.equal(add_noise(image, 0, torch.Generator().manual_seed(1)), image)
    with pytest.raises(ValueError):
        add_noise(image, -0.1, torch.Generator())


def test_collect_runs_skips_unrelated(tmp_path):
    valid = tmp_path / "one"
    valid.mkdir()
    (valid / "summary.json").write_text(json.dumps({"best_validation_accuracy": 0.8}), encoding="utf-8")
    (valid / "config.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
    unrelated = tmp_path / "two"
    unrelated.mkdir()
    (unrelated / "summary.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
    (unrelated / "config.json").write_text("{}", encoding="utf-8")
    assert [row["run"] for row in collect_runs(tmp_path)] == ["one"]
