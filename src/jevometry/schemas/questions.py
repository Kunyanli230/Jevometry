"""Question, outcome and support-set declarations.

Semantic identity of a question is captured by a stable hash over the
primitive, the ordered outcome identifiers, their descriptions, the rubric and
the legend.  Two probability vectors may only be compared positionally when
their semantic hashes match, or when the caller supplies an explicit mapping.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import Field, model_validator

from jevometry.schemas.common import Primitive, StrictModel

MAX_OUTCOMES_PER_NODE = 32


class OutcomeSpec(StrictModel):
    """One outcome (candidate, grade, or noul polarity)."""

    id: str
    description: str | None = None
    numeric: float | None = None

    @model_validator(mode="after")
    def _validate_id(self) -> OutcomeSpec:
        if not self.id:
            raise ValueError("outcome id must be non-empty")
        return self


class QuestionSpec(StrictModel):
    """A Jev question with a fixed outcome support."""

    id: str
    primitive: Primitive
    outcomes: list[OutcomeSpec]
    legend: dict[str, str] | None = None
    instructions: str | None = None
    rubric: str | None = None
    version: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> QuestionSpec:
        if not self.id:
            raise ValueError("question id must be non-empty")
        if not self.outcomes:
            raise ValueError("a question must declare at least one outcome")
        if len(self.outcomes) > MAX_OUTCOMES_PER_NODE:
            raise ValueError(
                f"{len(self.outcomes)} outcomes requested but the v0.1 limit is "
                f"{MAX_OUTCOMES_PER_NODE} per node"
            )
        ids = [outcome.id for outcome in self.outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("outcome ids must be unique")
        if self.primitive is Primitive.NOUL:
            if set(ids) != {"false", "true"}:
                raise ValueError("noul outcomes must be exactly false/true")
        if self.legend is not None:
            unknown = set(self.legend) - set(ids)
            if unknown:
                raise ValueError(f"legend references unknown outcomes: {sorted(unknown)}")
        return self

    @property
    def support(self) -> tuple[str, ...]:
        return tuple(outcome.id for outcome in self.outcomes)

    def numeric_encoding(self) -> dict[str, float] | None:
        encodings = {
            outcome.id: outcome.numeric
            for outcome in self.outcomes
            if outcome.numeric is not None
        }
        if not encodings:
            return None
        if len(encodings) != len(self.outcomes):
            return None
        return encodings

    def semantic_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "primitive": self.primitive.value,
            "outcomes": [
                {
                    "id": outcome.id,
                    "description": outcome.description,
                    "numeric": outcome.numeric,
                }
                for outcome in self.outcomes
            ],
            "legend": self.legend,
            "instructions": self.instructions,
            "rubric": self.rubric,
            "version": self.version,
        }

    def semantic_hash(self) -> str:
        payload = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SupportMapping(StrictModel):
    """Explicit one-to-one mapping between two question supports."""

    left_question: str
    right_question: str
    mapping: dict[str, str]
    confirmed: bool = False

    @model_validator(mode="after")
    def _validate(self) -> SupportMapping:
        if not self.confirmed:
            raise ValueError("explicit support mappings must be confirmed by the caller")
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError("support mapping must be injective")
        return self


class QuestionSet(StrictModel):
    """Ordered question declarations for a system."""

    questions: list[QuestionSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique(self) -> QuestionSet:
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return self

    def by_id(self, question_id: str) -> QuestionSpec:
        for question in self.questions:
            if question.id == question_id:
                return question
        raise KeyError(question_id)
