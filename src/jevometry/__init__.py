"""Information-geometric analysis of Jev agent systems.

Public API::

    from jevometry import Experiment, analyze, render_report
    from jevometry.adapters import TypeSafeAdapter

    experiment = Experiment.from_yaml("experiment.yaml")
    captures = experiment.run(adapter=TypeSafeAdapter.from_env(...), live=True)
    analysis = analyze(captures, adapter=adapter)
    render_report(analysis, output="analysis/report.html")

Offline users can work directly with
:class:`~jevometry.schemas.distribution.DistributionRecord` and declared joint
models without any provider.
"""

from jevometry.analysis import Analysis, analyze
from jevometry.experiments.acquisition import Captures
from jevometry.pipeline import (
    Experiment,
    load_captures,
    load_run_experiment,
    render_report,
    run_inference,
)
from jevometry.schemas.common import (
    AnalysisObject,
    MetricResult,
    MetricStatus,
    ParameterRole,
    RoutingSemantics,
)
from jevometry.schemas.distribution import DistributionRecord, DistributionSource
from jevometry.schemas.experiment import ExperimentSpec, SamplingContract
from jevometry.schemas.joint import JointDistribution
from jevometry.schemas.parameters import ParameterSpec, StencilSpec
from jevometry.schemas.questions import QuestionSpec
from jevometry.schemas.system import CompositionMode, SystemSpec

__version__ = "0.1.0"

__all__ = [
    "Analysis",
    "AnalysisObject",
    "Captures",
    "CompositionMode",
    "DistributionRecord",
    "DistributionSource",
    "Experiment",
    "ExperimentSpec",
    "JointDistribution",
    "MetricResult",
    "MetricStatus",
    "ParameterRole",
    "ParameterSpec",
    "QuestionSpec",
    "RoutingSemantics",
    "SamplingContract",
    "StencilSpec",
    "SystemSpec",
    "__version__",
    "analyze",
    "load_captures",
    "load_run_experiment",
    "render_report",
    "run_inference",
]
