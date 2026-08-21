"""
src/guardrails/__init__.py

Public API for the Asmodieus guardrail system.
Import: from src.guardrails import check_off_topic, check_input_safety, check_grounded
"""

from .checker import (
    check_off_topic,
    check_input_safety,
    check_grounded,
    run_all_guardrails,
    GuardrailResult,
)

__all__ = [
    "check_off_topic",
    "check_input_safety",
    "check_grounded",
    "run_all_guardrails",
    "GuardrailResult",
]
