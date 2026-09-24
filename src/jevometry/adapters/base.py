"""Framework-independent adapter protocols and shared data structures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from jevometry.geometry.derivatives import NodeEvaluation, NodeEvaluator
from jevometry.schemas.experiment import CaseSpec
from jevometry.schemas.joint import JointDistribution
from jevometry.schemas.system import SystemSpec
from jevometry.schemas.trace import EvaluationTrace

FloatArray = NDArray[np.float64]


def request_fingerprint(
    *,
    provider: str,
    model: str | None,
    adapter_version: str | None,
    node_id: str,
    question_hash: str,
    rendered_fingerprint: str,
    theta: Mapping[str, float],
    history: Mapping[str, str] | None = None,
    stencil_role: str = "center",
) -> str:
    """Canonical request fingerprint used by caching and replay matching.

    The fingerprint deliberately excludes the repeat index: repeated requests
    share a semantic request hash but receive distinct capture ids.
    """
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "adapter_version": adapter_version,
            "node_id": node_id,
            "question_hash": question_hash,
            "rendered": rendered_fingerprint,
            "theta": {key: float(value) for key, value in sorted(theta.items())},
            "history": dict(sorted((history or {}).items())),
            "stencil_role": stencil_role,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentPoint:
    """One (case, theta, repeat) evaluation point."""

    case: CaseSpec
    theta: Mapping[str, float]
    point_id: str
    repeat: int = 0
    history: Mapping[str, str] = field(default_factory=dict)
    stencil_role: str = "center"


@runtime_checkable
class SystemAdapter(Protocol):
    """Anything that can describe itself and evaluate declared nodes."""

    def describe(self) -> SystemSpec: ...

    def evaluate_point(self, point: ExperimentPoint) -> list[EvaluationTrace]: ...


@runtime_checkable
class JointModel(Protocol):
    """A declared finite joint probability family over system outcomes."""

    node_order: list[str]

    def probabilities(self, theta: Mapping[str, float]) -> JointDistribution: ...

    def jacobian(self, theta: Mapping[str, float]) -> FloatArray | None: ...


@runtime_checkable
class LikelihoodModel(Protocol):
    """A declared observation model for synthetic inference."""

    support: tuple[str, ...]

    def log_prob(self, observations: Sequence[int], theta: Mapping[str, float]) -> float: ...

    def sample(
        self, theta: Mapping[str, float], n: int, rng: np.random.Generator
    ) -> NDArray[np.int64]: ...


__all__ = [
    "ExperimentPoint",
    "JointModel",
    "LikelihoodModel",
    "NodeEvaluation",
    "NodeEvaluator",
    "SystemAdapter",
]
