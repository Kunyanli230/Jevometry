"""Verify development dependencies locally without making API calls."""

import importlib
import importlib.metadata
import sys

import numpy as np
from scipy.special import expit
from typesafe_sdk import Choice, Noul, Score

import jevometry

assert (3, 11) <= sys.version_info[:2] < (3, 14)
for module, package in (
    ("numpy", "numpy"), ("scipy", "scipy"), ("pydantic", "pydantic"),
    ("yaml", "PyYAML"), ("typer", "typer"), ("plotly", "plotly"),
    ("jinja2", "Jinja2"), ("typesafe_sdk", "typesafe-sdk"),
    ("pytest", "pytest"), ("hypothesis", "hypothesis"),
):
    importlib.import_module(module)
    print(f"{package}: {importlib.metadata.version(package)}")

assert expit(0.0) == 0.5
assert np.allclose(np.linalg.eigvalsh(np.eye(2)), [1.0, 1.0])
questions = {
    "action": Choice(instructions="Choose an action.", criteria={"keep": "Keep", "review": "Review"}),
    "applicable": Noul(instructions="Is the action applicable?"),
    "risk": Score(instructions="Assess risk.", criteria=["Low", "Medium", "High"]),
}
assert len(questions) == 3
print(f"Jevometry {jevometry.__version__}; Python {sys.version.split()[0]}")
print("PASS: imports, numerical runtime, and SDK question construction. No API requests made.")
