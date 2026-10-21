import json

import pytest

torch = pytest.importorskip("torch")

from src.model import FashionMLP
from src.utils import count_trainable_parameters, load_config


def test_parameter_count() -> None:
    hidden_size = 16
    model = FashionMLP(hidden_size=hidden_size)
    expected = 784 * hidden_size + hidden_size + hidden_size * 10 + 10
    assert count_trainable_parameters(model) == expected


def test_load_config_reports_missing_fields(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"epochs": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing config fields"):
        load_config(path)
