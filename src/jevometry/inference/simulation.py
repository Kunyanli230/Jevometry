"""Synthetic inference validation: bias, variance, RMSE and coverage.

These simulations validate the *declared model*.  They are not evidence about
the real system, and the reported numbers carry Monte Carlo error that is
reported alongside them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jevometry.adapters.base import LikelihoodModel
from jevometry.inference.likelihood import mle, profile_interval
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.parameters import ParameterSpec
from jevometry.schemas.results import SimulationSummary


@dataclass
class SimulationConfig:
    """One declared synthetic experiment."""

    parameter: str
    true_value: float
    sample_size: int
    replications: int
    seed: int = 0
    confidence_level: float = 0.95
    other_parameters: dict[str, float] | None = None


def run_simulation(
    model: LikelihoodModel,
    config: SimulationConfig,
    *,
    parameter_specs: Sequence[ParameterSpec],
    crlb_variance: float | None = None,
) -> SimulationSummary:
    """Simulate MLE behaviour under the declared model."""
    names = [spec.name for spec in parameter_specs]
    if config.parameter not in names:
        return SimulationSummary(
            parameter=config.parameter,
            true_value=config.true_value,
            sample_size=config.sample_size,
            replications=config.replications,
            notes=[f"unknown parameter {config.parameter!r}"],
        )
    true_theta = dict(config.other_parameters or {})
    true_theta[config.parameter] = config.true_value
    rng = np.random.default_rng(config.seed)
    estimates: list[float] = []
    covered: list[bool] = []
    widths: list[float] = []
    failures = 0
    for _ in range(config.replications):
        observations = model.sample(true_theta, config.sample_size, rng)
        fit = mle(model, observations, parameter_specs=parameter_specs, initial=true_theta)
        if fit.theta is None:
            failures += 1
            continue
        estimates.append(fit.theta[config.parameter])
        interval = profile_interval(
            model,
            observations,
            config.parameter,
            parameter_specs=parameter_specs,
            level=config.confidence_level,
            initial=true_theta,
        )
        if interval.lower is None or interval.upper is None:
            failures += 1
            continue
        covered.append(interval.lower <= config.true_value <= interval.upper)
        widths.append(interval.upper - interval.lower)
    successes = len(estimates)
    if successes == 0:
        return SimulationSummary(
            parameter=config.parameter,
            true_value=config.true_value,
            sample_size=config.sample_size,
            replications=config.replications,
            failures=failures,
            crlb_variance=crlb_variance,
            notes=["no successful replications"],
        )
    estimate_array = np.asarray(estimates, dtype=np.float64)
    bias = float(estimate_array.mean() - config.true_value)
    variance = float(estimate_array.var(ddof=1)) if successes > 1 else 0.0
    rmse = float(np.sqrt(np.mean((estimate_array - config.true_value) ** 2)))
    coverage = float(np.mean(covered)) if covered else None
    mean_width = float(np.mean(widths)) if widths else None
    monte_carlo_error = (
        float(np.sqrt(coverage * (1.0 - coverage) / len(covered)))
        if coverage is not None and covered
        else None
    )
    notes = [
        "synthetic simulation of the declared observation model",
        f"successful replications: {successes} of {config.replications}",
        "coverage error is a binomial Monte Carlo standard error",
    ]
    if crlb_variance is not None:
        notes.append(
            "compare empirical variance with the analytic CRLB variance; "
            "a variance below the CRLB indicates a model or implementation problem"
        )
    return SimulationSummary(
        parameter=config.parameter,
        true_value=config.true_value,
        sample_size=config.sample_size,
        replications=config.replications,
        failures=failures,
        bias=bias,
        variance=variance,
        rmse=rmse,
        coverage=coverage,
        mean_interval_width=mean_width,
        crlb_variance=crlb_variance,
        monte_carlo_standard_error=monte_carlo_error,
        notes=notes,
    )


def bernoulli_crlb_variance(p: float) -> float:
    """Analytic single-observation Bernoulli CRLB variance p(1-p)."""
    return float(p * (1.0 - p))


def crlb_standard_error_from_information(information: float) -> float:
    """sqrt(1 / I_1) for a scalar parameter."""
    if information <= 0.0:
        raise ValueError("information must be positive")
    return float(1.0 / np.sqrt(information))


def simulation_status(summary: SimulationSummary) -> MetricStatus:
    """Status helper: a simulation with all failures is not a success."""
    if summary.failures >= summary.replications:
        return MetricStatus.FAILED
    if summary.failures > 0:
        return MetricStatus.CONDITIONAL
    return MetricStatus.OK
