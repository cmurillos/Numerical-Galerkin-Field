import json
import subprocess
import sys
from pathlib import Path


def test_field_benchmark_uses_the_operational_basis_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/field_evaluation.py",
            "--batch-sizes",
            "1",
            "--modes",
            "2",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["results"][0]["N"] == 2
    assert report["results"][0]["B"] == 1
    assert report["results"][0]["Q"] > 0
