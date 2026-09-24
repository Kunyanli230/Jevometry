"""Synthetic MLE validation against the analytic Bernoulli CRLB."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.adapters.analytic import bernoulli_likelihood
from jevometry.inference.likelihood import mle, profile_interval
from jevometry.inference.simulation import SimulationConfig, run_simulation
from jevometry.schemas.parameters import ParameterSpec


def p_spec() -> ParameterSpec:
    return ParameterSpec(
        name="p",
        role="task_relevant",
        unit="probability",
        bounds=(0.0, 1.0),
        step=1e-4,
        center=0.5,
    )


def test_mle_recovers_sample_frequency() -> None:
    model = bernoulli_likelihood("p")
    observations = [1] * 30 + [0] * 70
    fit = mle(model, observations, parameter_specs=[p_spec()])
    assert fit.theta is not None
    assert fit.theta["p"] == pytest.approx(0.3, abs=1e-3)


def test_profile_interval_contains_estimate_and_is_asymmetric_near_boundary() -> None:
    model = bernoulli_likelihood("p")
    observations = [1] * 95 + [0] * 5
    interval = profile_interval(model, observations, "p", parameter_specs=[p_spec()])
    assert interval.lower is not None and interval.upper is not None
    assert interval.lower < interval.estimate < interval.upper
    assert interval.width is not None


def test_simulation_variance_matches_crlb_within_monte_carlo_error() -> None:
    model = bernoulli_likelihood("p")
    p = 0.3
    n = 200
    config = SimulationConfig(
        parameter="p", true_value=p, sample_size=n, replications=200, seed=20260924
    )
    summary = run_simulation(model, config, parameter_specs=[p_spec()])
    assert summary.failures == 0
    assert summary.variance is not None
    crlb = p * (1 - p) / n
    assert summary.variance == pytest.approx(crlb, rel=0.35)
    assert summary.rmse is not None
    assert summary.rmse == pytest.approx(np.sqrt(crlb), rel=0.25)
    assert summary.coverage is not None
    assert summary.coverage >= 0.90
    assert summary.mean_interval_width is not None
    assert summary.mean_interval_width > 0.0
    assert summary.monte_carlo_standard_error is not None


def test_simulation_is_reproducible_with_fixed_seed() -> None:
    model = bernoulli_likelihood("p")
    config = SimulationConfig(
        parameter="p", true_value=0.45, sample_size=50, replications=20, seed=7
    )
    first = run_simulation(model, config, parameter_specs=[p_spec()])
    second = run_simulation(model, config, parameter_specs=[p_spec()])
    assert first.bias == second.bias
    assert first.variance == second.variance
    assert first.coverage == second.coverage


def test_simulation_reports_unknown_parameter() -> None:
    model = bernoulli_likelihood("p")
    config = SimulationConfig(
        parameter="q", true_value=0.5, sample_size=10, replications=2, seed=1
    )
    summary = run_simulation(model, config, parameter_specs=[p_spec()])
    assert summary.failures == 0
    assert summary.bias is None
    assert any("unknown parameter" in note for note in summary.notes)
