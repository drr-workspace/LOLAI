from __future__ import annotations

from evals.metrics import (
    abstention_metrics,
    calibration_metrics,
    counterfactual_metrics,
    decision_metrics,
    grounding_metrics,
    robustness_metrics,
    structural_metrics,
)
from evals.evaluate_predictions import apply_quality_gates


def _output(decision: str = "SHOW", action: str | None = "a1") -> dict:
    return {
        "schemaVersion": "1.0.0",
        "decision": decision,
        "category": "MACRO_PRIORITY",
        "primaryActionId": action,
        "alternativeActionIds": [],
        "priority": "HIGH",
        "confidence": 0.9,
        "reasonCodes": ["R1"],
        "evidenceIds": ["e1"],
        "message": "Choose the safe action." if decision == "SHOW" else "",
        "validForSeconds": 10,
        "recheckTriggers": ["STATE_CHANGED"],
    }


def _sample(
    expected: dict | None = None,
    prediction: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "input": {
            "outputLocale": "en-US",
            "evidence": [{"evidenceId": "e1"}],
            "payload": {"candidates": [{"actionId": "a1"}]},
        },
        "expected": expected or _output(),
        "prediction": prediction or _output(),
        "predictionValid": True,
        "metadata": metadata or {},
    }


def test_structural_metrics_perfect() -> None:
    metrics = structural_metrics.compute(
        [_sample()], schema_valid=lambda value: isinstance(value, dict)
    )
    assert set(metrics.values()) == {1.0}


def test_decision_and_reason_metrics() -> None:
    metrics = decision_metrics.compute([_sample()])
    assert metrics["decisionAccuracy"] == 1.0
    assert metrics["reasonCodeF1"] == 1.0


def test_abstention_metrics() -> None:
    suppress = _output("SUPPRESS", None)
    metrics = abstention_metrics.compute([_sample(suppress, suppress)])
    assert metrics["suppress"]["f1"] == 1.0
    assert metrics["falseShowRate"] == 0.0


def test_grounding_detects_inventions() -> None:
    prediction = _output()
    prediction["primaryActionId"] = "invented"
    prediction["evidenceIds"] = ["invented"]
    prediction["reasonCodes"] = ["UNKNOWN"]
    prediction["message"] = "Use patch facts and win rate."
    metrics = grounding_metrics.compute(
        [_sample(prediction=prediction)], allowed_reason_codes={"R1"}
    )
    assert metrics["inventedActionRate"] == 1.0
    assert metrics["inventedEvidenceRate"] == 1.0
    assert metrics["unsupportedReasonCodeRate"] == 1.0
    assert metrics["unsupportedClaimRate"] == 1.0


def test_counterfactual_metrics() -> None:
    metadata = {
        "counterfactualPairId": "pair",
        "expectedEffect": "SAME_DECISION",
    }
    samples = [
        _sample(metadata=metadata),
        _sample(metadata={**metadata, "isCounterfactual": True}),
    ]
    metrics = counterfactual_metrics.compute(samples)
    assert metrics["counterfactualConsistency"] == 1.0
    assert metrics["invariantPairConsistency"] == 1.0
    assert metrics["completePairCount"] == 1


def test_counterfactual_metrics_are_unavailable_without_pairs() -> None:
    metrics = counterfactual_metrics.compute([_sample()])

    assert metrics["completePairCount"] == 0
    assert metrics["counterfactualConsistency"] is None


def test_robustness_metrics() -> None:
    samples = [
        _sample(metadata={"robustnessPairId": "pair"}),
        _sample(
            metadata={
                "robustnessPairId": "pair",
                "robustnessType": "PROMPT_INJECTION_IN_DATA",
                "expectedEffect": "SAME_DECISION",
            }
        ),
    ]
    assert robustness_metrics.compute(samples)["injectionResistance"] == 1.0


def test_robustness_metrics_are_unavailable_without_pairs() -> None:
    metrics = robustness_metrics.compute([_sample()])

    assert metrics["candidateOrderInvariance"] is None
    assert metrics["candidateOrderInvariancePairCount"] == 0


def test_calibration_metrics() -> None:
    metrics = calibration_metrics.compute([_sample()])
    assert metrics["expectedCalibrationError"] == 0.1
    assert metrics["meanConfidenceByDecision"]["SHOW"] == 0.9


def test_quality_gate_failure_is_reported() -> None:
    report = {"metrics": {"structural": {"jsonValidity": 0.5}}}
    gates = {
        "jsonValidity": {"operator": "minimum", "value": 1.0}
    }
    results = apply_quality_gates(report, gates)
    assert results[0]["passed"] is False
