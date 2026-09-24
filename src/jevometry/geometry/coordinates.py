"""Raw versus standardized parameter coordinates.

Standardized coordinates ``z_a = theta_a / scale_a`` are used only to make
cross-parameter comparisons dimensionless.  Both raw and standardized results
are persisted; eigenvalue comparisons are only meaningful when parameters,
units, scales, experiment points and probability objects match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jevometry.geometry.fisher import transform_coordinates, transform_jacobian
from jevometry.schemas.parameters import ParameterSet, ParameterSpec

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoordinateTransform:
    """Linear coordinate change theta = theta0 + S z."""

    names: list[str]
    scales: FloatArray

    @classmethod
    def from_parameters(cls, parameters: ParameterSet | list[ParameterSpec]) -> CoordinateTransform:
        specs = parameters.parameters if isinstance(parameters, ParameterSet) else parameters
        return cls(names=[spec.name for spec in specs], scales=np.asarray([spec.scale for spec in specs]))

    def standardize(self, theta: dict[str, float]) -> dict[str, float]:
        return {
            name: theta[name] / float(scale)
            for name, scale in zip(self.names, self.scales, strict=True)
        }

    def fisher(self, raw_matrix: FloatArray) -> FloatArray:
        return transform_coordinates(raw_matrix, self.scales)

    def jacobian(self, raw_jacobian: FloatArray) -> FloatArray:
        return transform_jacobian(raw_jacobian, self.scales)
