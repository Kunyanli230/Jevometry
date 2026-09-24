"""Deterministic parameter renderers.

A renderer maps external experiment coordinates ``theta`` and a fixed case to
the state that is actually sent to the system.  Renderers must be
deterministic; if two distinct theta values render to the same input, the
derivative engine reports ``resolution_limited`` instead of pretending the
parameter is observable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from jevometry.schemas.experiment import CaseSpec
from jevometry.schemas.trace import RenderedInput


def fingerprint_state(state: Any) -> str:
    """Stable content hash of a rendered state."""
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class ParameterRenderer(Protocol):
    """Protocol for deterministic theta -> state renderers."""

    version: str

    def render(self, theta: Mapping[str, float], case: CaseSpec) -> RenderedInput: ...


class TextTemplateRenderer:
    """Render a text state from a Python format template.

    The template may reference any declared parameter, e.g.
    ``"amount={amount:.2f}; wait={wait:.1f}"``.  Format specifiers control the
    effective resolution of the parameter in the rendered input.
    """

    def __init__(self, template: str, *, version: str = "1") -> None:
        if "{" not in template:
            raise ValueError("template must reference at least one parameter")
        self.template = template
        self.version = version

    def render(self, theta: Mapping[str, float], case: CaseSpec) -> RenderedInput:
        missing = [name for name in _template_fields(self.template) if name not in theta]
        if missing:
            raise KeyError(f"theta is missing template fields: {sorted(missing)}")
        state = self.template.format(**dict(theta))
        return RenderedInput(
            state=state,
            fingerprint=fingerprint_state({"text": state, "version": self.version}),
            theta=dict(theta),
            case_id=case.id,
            renderer_version=self.version,
        )


class StructuredRenderer:
    """Render a JSON object state from per-field format templates."""

    def __init__(
        self,
        templates: Mapping[str, str],
        *,
        static: Mapping[str, Any] | None = None,
        version: str = "1",
    ) -> None:
        if not templates:
            raise ValueError("at least one field template is required")
        self.templates = dict(templates)
        self.static = dict(static or {})
        self.version = version

    def render(self, theta: Mapping[str, float], case: CaseSpec) -> RenderedInput:
        state: dict[str, Any] = dict(self.static)
        for field, template in self.templates.items():
            missing = [name for name in _template_fields(template) if name not in theta]
            if missing:
                raise KeyError(f"theta is missing fields for {field!r}: {sorted(missing)}")
            state[field] = template.format(**dict(theta))
        if case.state is not None:
            state["case"] = case.state
        return RenderedInput(
            state=state,
            fingerprint=fingerprint_state({"json": state, "version": self.version}),
            theta=dict(theta),
            case_id=case.id,
            renderer_version=self.version,
        )


class CallableRenderer:
    """Wrap a user-supplied deterministic function as a renderer."""

    def __init__(
        self,
        function: Callable[[Mapping[str, float], CaseSpec], Any],
        *,
        version: str = "callable-1",
    ) -> None:
        self.function = function
        self.version = version

    def render(self, theta: Mapping[str, float], case: CaseSpec) -> RenderedInput:
        state = self.function(dict(theta), case)
        return RenderedInput(
            state=state,
            fingerprint=fingerprint_state({"state": state, "version": self.version}),
            theta=dict(theta),
            case_id=case.id,
            renderer_version=self.version,
        )


def _template_fields(template: str) -> list[str]:
    fields: list[str] = []
    for _, field_name, _, _ in _formatter().parse(template):
        if field_name is None:
            continue
        root = field_name.split(".")[0].split("[")[0]
        if root:
            fields.append(root)
    return fields


def _formatter() -> Any:
    import string

    return string.Formatter()


def check_renderer_resolution(
    renderer: ParameterRenderer,
    theta: Mapping[str, float],
    case: CaseSpec,
    parameters: Mapping[str, float],
) -> dict[str, bool]:
    """Check whether a parameter step changes the rendered input.

    Returns a mapping parameter name -> True when the step is *resolved* by the
    renderer (i.e. the fingerprint changes).
    """
    center = renderer.render(theta, case)
    resolved: dict[str, bool] = {}
    for name, step in parameters.items():
        shifted = dict(theta)
        shifted[name] = theta[name] + step
        moved = renderer.render(shifted, case)
        resolved[name] = moved.fingerprint != center.fingerprint
    return resolved
