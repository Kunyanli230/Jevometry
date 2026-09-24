"""Experiment parameter and finite-difference stencil declarations."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import Field, field_validator, model_validator

from jevometry.schemas.common import ParameterRole, StrictModel

MAX_PARAMETERS = 8


class StencilKind(str, Enum):
    """Finite-difference stencil orientation."""

    CENTRAL = "central"
    FORWARD = "forward"
    BACKWARD = "backward"


class ParameterSpec(StrictModel):
    """One external experiment parameter.

    ``theta`` is always an *external* experiment coordinate rendered into an
    input; it is never a claim about inaccessible Jev internal weights.
    """

    name: str
    role: ParameterRole
    unit: str
    bounds: tuple[float, float]
    scale: float = 1.0
    step: float
    continuous: bool = True
    description: str | None = None
    center: float | None = None

    @field_validator("bounds")
    @classmethod
    def _validate_bounds(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError("bounds must be finite")
        if low >= high:
            raise ValueError("bounds must satisfy low < high")
        return value

    @field_validator("scale", "step")
    @classmethod
    def _validate_positive(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("scale and step must be finite and positive")
        return value

    @model_validator(mode="after")
    def _validate_center(self) -> ParameterSpec:
        if self.center is not None:
            low, high = self.bounds
            if not (low <= self.center <= high):
                raise ValueError("center must lie within bounds")
        return self

    def contains(self, value: float, *, margin: float = 0.0) -> bool:
        low, high = self.bounds
        return (low + margin) <= value <= (high - margin)

    def standardize(self, value: float) -> float:
        return value / self.scale


class StencilSpec(StrictModel):
    """Which finite-difference steps to evaluate for every coordinate."""

    kind: StencilKind = StencilKind.CENTRAL
    step_scales: list[float] = Field(default_factory=lambda: [1.0, 0.5])
    relative_stability_threshold: float = 0.10
    matrix_relative_floor: float = 1e-12

    @field_validator("step_scales")
    @classmethod
    def _validate_scales(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("step_scales must not be empty")
        for scale in value:
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("step scales must be finite and positive")
        if 1.0 not in value:
            raise ValueError("step_scales must include the nominal step 1.0")
        return value

    @field_validator("relative_stability_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("relative_stability_threshold must be positive")
        return value


class ParameterSet(StrictModel):
    """Ordered parameter declarations for one experiment."""

    parameters: list[ParameterSpec]

    @model_validator(mode="after")
    def _validate_unique(self) -> ParameterSet:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        if not names:
            raise ValueError("at least one parameter is required")
        if len(names) > MAX_PARAMETERS:
            raise ValueError(
                f"{len(names)} parameters requested but the v0.1 limit is "
                f"{MAX_PARAMETERS}; reduce the parameter set"
            )
        return self

    @property
    def names(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]

    def by_name(self, name: str) -> ParameterSpec:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    def scales(self) -> list[float]:
        return [parameter.scale for parameter in self.parameters]
