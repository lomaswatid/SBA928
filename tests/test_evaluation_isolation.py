"""Evaluation primitives import without model libraries or filesystem access."""

import subprocess
import sys
from pathlib import Path


def test_primitives_import_without_model_dependencies():
    # A fresh process avoids replacing dataclass identities in other tests.
    program = """
import builtins, importlib, sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'datasets', 'huggingface_hub'}:
        raise AssertionError('Model dependency imported by pure scoring')
    return original_import(name, *args, **kwargs)
def forbidden_open(*args, **kwargs):
    raise AssertionError('Scoring import opened a data file')
builtins.__import__ = guarded_import
builtins.open = forbidden_open
for name in ('schema', 'numerical', 'repetition', 'dataset', 'scoring', 'reporting'):
    importlib.import_module('sba_928.evaluation.' + name)
"""
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(source)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
