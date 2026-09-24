"""Framework-agnostic adapters and providers.

Adapters implement the protocols in :mod:`jevometry.adapters.base`.  Nothing in
the geometry, systems or inference core depends on a specific agent framework.
"""

from jevometry.adapters.analytic import (
    AnalyticAdapter,
    AnalyticJointModel,
    AnalyticLikelihoodModel,
    AnalyticNode,
    bernoulli_likelihood,
    bernoulli_node,
    bernoulli_pair_joint_model,
    choice_question,
    conditional_tree_joint_model,
    logistic_node,
    noul_question,
    rank_deficient_softmax_node,
    sample_case,
    score_question,
    softmax_node,
)
from jevometry.adapters.base import (
    ExperimentPoint,
    JointModel,
    LikelihoodModel,
    NodeEvaluator,
    SystemAdapter,
    request_fingerprint,
)
from jevometry.adapters.capture import CaptureRecord, CaptureRecorder, capture
from jevometry.adapters.replay import ReplayAdapter, ReplayError
from jevometry.adapters.typesafe import (
    TypeSafeAdapter,
    TypeSafeProviderError,
    TypeSafeQuestion,
    typesafe_choice_question,
    typesafe_noul_question,
    typesafe_score_question,
)

__all__ = [
    "AnalyticAdapter",
    "AnalyticJointModel",
    "AnalyticLikelihoodModel",
    "AnalyticNode",
    "CaptureRecord",
    "CaptureRecorder",
    "ExperimentPoint",
    "JointModel",
    "LikelihoodModel",
    "NodeEvaluator",
    "ReplayAdapter",
    "ReplayError",
    "SystemAdapter",
    "TypeSafeAdapter",
    "TypeSafeProviderError",
    "TypeSafeQuestion",
    "bernoulli_likelihood",
    "bernoulli_node",
    "bernoulli_pair_joint_model",
    "capture",
    "choice_question",
    "conditional_tree_joint_model",
    "logistic_node",
    "noul_question",
    "rank_deficient_softmax_node",
    "request_fingerprint",
    "sample_case",
    "score_question",
    "softmax_node",
    "typesafe_choice_question",
    "typesafe_noul_question",
    "typesafe_score_question",
]
