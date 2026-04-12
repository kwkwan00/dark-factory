"""DeepEval metrics for evaluating generated specs, code, and tests.

Provides GEval-based metrics that assess whether generated artifacts are correct,
coherent, complete, and safe relative to requirements and acceptance criteria.
Uses an OpenAI model as the LLM-as-a-judge for evaluation.
"""

from __future__ import annotations

import os

import structlog
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

log = structlog.get_logger()

# DeepEval natively supports OpenAI models via the model string parameter.
# Set OPENAI_API_KEY in your environment.
#
# The eval model is mutable at runtime so the Settings tab can switch it
# without a process restart. ``get_eval_model()`` is called inside each
# metric builder (at metric construction time, not at import time) so
# the latest value is always picked up.
#
# ``set_eval_model`` is called from:
#   - ``load_settings`` at app startup (after env var + config.toml merge)
#   - ``PATCH /api/settings`` when the operator changes the dropdown
#
# NOTE: gpt-5.4 may not be available in all OpenAI accounts. If you see
# `403 model not found`, change the value via the Settings tab (or set
# ``EVAL_MODEL=gpt-4.1`` etc. in the environment at boot).
_eval_model: str = os.getenv("EVAL_MODEL", "gpt-5.4")


def get_eval_model() -> str:
    """Return the currently-configured evaluation model name."""
    return _eval_model


def set_eval_model(name: str) -> None:
    """Update the evaluation model at runtime.

    Subsequent metric-builder calls will use the new name. Metrics
    already built and handed to DeepEval are not retroactively updated
    — this only affects the NEXT pipeline run.
    """
    global _eval_model
    if not name or not isinstance(name, str):
        raise ValueError(f"eval model name must be a non-empty string, got {name!r}")
    _eval_model = name
    log.info("eval_model_updated", model=name)


# ── Metric builders ──────────────────────────────────────────────────


def build_correctness_metric(threshold: float = 0.5) -> GEval:
    """Test logic correctly validates the acceptance criteria."""
    return GEval(
        name="Test Correctness",
        criteria=(
            "Determine whether the generated test code correctly validates the "
            "acceptance criteria from the specification. The tests should assert "
            "the right conditions, use appropriate test patterns, and would pass "
            "when run against a correct implementation."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=get_eval_model(),
    )


def build_coherence_metric(threshold: float = 0.5) -> GEval:
    """Test code is well-structured, readable, and logically organised."""
    return GEval(
        name="Test Coherence",
        criteria=(
            "Assess whether the generated test code is well-structured, readable, "
            "and logically organised. Tests should have clear names, proper setup/"
            "teardown, no redundant logic, and follow standard testing conventions "
            "for the language."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
        model=get_eval_model(),
    )


def build_completeness_metric(threshold: float = 0.5) -> GEval:
    """Tests cover all acceptance criteria from the spec."""
    return GEval(
        name="Test Completeness",
        criteria=(
            "Evaluate whether the generated tests cover ALL acceptance criteria "
            "listed in the specification. Each criterion should have at least one "
            "corresponding test case. Missing coverage of any criterion should "
            "lower the score."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.INPUT,
        ],
        threshold=threshold,
        model=get_eval_model(),
    )


def build_code_quality_metric(threshold: float = 0.5) -> GEval:
    """Generated code follows best practices and handles edge cases."""
    return GEval(
        name="Code Quality",
        criteria=(
            "Assess whether the generated code follows language best practices, "
            "handles edge cases, includes proper error handling, and uses "
            "appropriate data structures and algorithms."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=get_eval_model(),
    )


# ── Spec evaluation metrics ───────────────────────────────────────────


def build_spec_correctness_metric(threshold: float = 0.5) -> GEval:
    """Spec accurately captures the requirement's intent."""
    return GEval(
        name="Spec Correctness",
        criteria=(
            "Determine whether the generated specification accurately captures "
            "the original requirement's intent. The spec should correctly translate "
            "the requirement into technical details without adding unsupported "
            "functionality or omitting key aspects of the requirement."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.INPUT,
        ],
        threshold=threshold,
        model=get_eval_model(),
    )


def build_spec_coherence_metric(threshold: float = 0.5) -> GEval:
    """Spec is well-structured, unambiguous, and internally consistent."""
    return GEval(
        name="Spec Coherence",
        criteria=(
            "Assess whether the specification is well-structured, unambiguous, "
            "and internally consistent. It should have a clear title, detailed "
            "description, concrete acceptance criteria with WHEN/THEN scenarios, "
            "and no contradictory statements."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
        model=get_eval_model(),
    )


def build_spec_instruction_following_metric(threshold: float = 0.5) -> GEval:
    """Spec follows the OpenSpec format and includes all required fields."""
    return GEval(
        name="Spec Instruction Following",
        criteria=(
            "Evaluate whether the specification follows the requested format: "
            "includes an id, title, description, requirement_ids, acceptance "
            "criteria, dependencies, capability (kebab-case), and WHEN/THEN "
            "scenarios. Each scenario must have a name, when condition, and "
            "then outcome. The spec should be actionable enough for a developer "
            "to implement without further clarification."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.INPUT,
        ],
        threshold=threshold,
        model=get_eval_model(),
    )


def build_spec_safety_metric(threshold: float = 0.5) -> GEval:
    """Spec does not introduce unsafe, unethical, or harmful functionality."""
    return GEval(
        name="Spec Safety & Ethics",
        criteria=(
            "Evaluate whether the specification avoids introducing unsafe, "
            "unethical, or harmful functionality. Check that it does not: "
            "specify collection of unnecessary personal data, bypass security "
            "controls, introduce discriminatory logic, enable surveillance "
            "without consent, or violate privacy regulations. The spec should "
            "include appropriate security considerations (authentication, "
            "authorization, input validation) where relevant."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.INPUT,
        ],
        threshold=threshold,
        model=get_eval_model(),
    )


def build_spec_test_case(
    *,
    requirement_title: str,
    requirement_description: str,
    spec_json: str,
) -> LLMTestCase:
    """Build a deepeval LLMTestCase for evaluating a generated spec.

    - input: the original requirement (what was asked for)
    - actual_output: the generated spec JSON (what was produced)
    - expected_output: the requirement description (ground truth)
    """
    return LLMTestCase(
        input=f"Requirement: {requirement_title}\n\n{requirement_description}",
        actual_output=spec_json,
        expected_output=requirement_description,
    )


def evaluate_generated_spec(
    *,
    requirement_title: str,
    requirement_description: str,
    spec_json: str,
    threshold: float = 0.5,
    target_spec_id: str | None = None,
    sub_spec_title: str | None = None,
) -> dict:
    """Run all spec evaluation metrics and return results.

    ``target_spec_id`` and ``sub_spec_title`` are passed through purely
    for log enrichment — with decomposition on, a single parent
    requirement fans out into many sub-specs, all of which would
    otherwise share the same ``requirement`` field in the
    ``evaluating_spec`` log event and become indistinguishable.

    Returns a dict with metric names as keys and dicts of score/passed/reason.
    """
    log.info(
        "evaluating_spec",
        requirement=requirement_title,
        target_spec_id=target_spec_id,
        sub_spec_title=sub_spec_title,
        threshold=threshold,
    )
    test_case = build_spec_test_case(
        requirement_title=requirement_title,
        requirement_description=requirement_description,
        spec_json=spec_json,
    )

    metrics = [
        build_spec_correctness_metric(threshold),
        build_spec_coherence_metric(threshold),
        build_spec_instruction_following_metric(threshold),
        build_spec_safety_metric(threshold),
    ]

    results: dict[str, dict] = {}
    for metric in metrics:
        metric.measure(test_case)
        results[metric.name] = {
            "score": metric.score,
            "passed": metric.is_successful(),
            "reason": getattr(metric, "reason", None),
        }

    return results


# ── Test case builder ────────────────────────────────────────────────


def build_test_case(
    *,
    spec_title: str,
    acceptance_criteria: list[str],
    source_code: str,
    test_code: str,
) -> LLMTestCase:
    """Build a deepeval LLMTestCase for evaluating generated tests.

    Maps dark-factory concepts to deepeval fields:
    - input: the spec title + acceptance criteria (what was asked for)
    - actual_output: the generated test code (what was produced)
    - expected_output: the acceptance criteria (what should be validated)
    - context: the source code being tested
    """
    criteria_text = "\n".join(f"- {c}" for c in acceptance_criteria)
    return LLMTestCase(
        input=f"Generate tests for: {spec_title}\n\nAcceptance Criteria:\n{criteria_text}",
        actual_output=test_code,
        expected_output=f"Tests that validate:\n{criteria_text}",
        context=[source_code],
    )


# ── Evaluation runner ────────────────────────────────────────────────


def evaluate_generated_tests(
    *,
    spec_title: str,
    acceptance_criteria: list[str],
    source_code: str,
    test_code: str,
    threshold: float = 0.5,
) -> dict:
    """Run all evaluation metrics on a generated test and return results.

    Returns a dict with metric names as keys and dicts of score/passed/reason as values.
    """
    log.info("evaluating_tests", spec=spec_title, threshold=threshold)
    test_case = build_test_case(
        spec_title=spec_title,
        acceptance_criteria=acceptance_criteria,
        source_code=source_code,
        test_code=test_code,
    )

    metrics = [
        build_correctness_metric(threshold),
        build_coherence_metric(threshold),
        build_completeness_metric(threshold),
    ]

    results: dict[str, dict] = {}
    for metric in metrics:
        metric.measure(test_case)
        results[metric.name] = {
            "score": metric.score,
            "passed": metric.is_successful(),
            "reason": getattr(metric, "reason", None),
        }

    return results
