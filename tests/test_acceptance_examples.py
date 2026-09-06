"""Run the same public examples and acceptance assertions that users execute."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("example", ["torus", "mixed_plate", "components", "periodic"])
def test_complete_space_to_field_example(example, capsys):
    path = Path(__file__).resolve().parents[1] / "examples" / f"acceptance_{example}.py"
    runpy.run_path(str(path), run_name="__main__")
    report = json.loads(capsys.readouterr().out)
    assert report["example"] == example
    assert report["dimension"] > 0
