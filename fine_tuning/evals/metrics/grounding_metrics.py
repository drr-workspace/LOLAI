from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _candidate_ids(user: Mapping[str, Any]) -> set[str]:
    payload = user.get("payload", {})
    if not isinstance(payload, dict):
        return set()
    candidates = payload.get("candidates", [])
    result = {
        item.get("actionId")
        for item in candidates
        if isinstance(item, dict)
    }
    advice = payload.get("candidateAdvice")
    if isinstance(advice, dict):
        result.add(advice.get("actionId"))
    return {str(value) for value in result if value}


def compute(
    samples: Sequence[Mapping[str, Any]],
    *,
    allowed_reason_codes: set[str],
) -> dict[str, float]:
    actions = evidence = reasons = claims = 0
    predicted_actions = predicted_evidence = predicted_reasons = messages = 0
    for sample in samples:
        user = sample["input"]
        prediction = sample.get("prediction") or {}
        action = prediction.get("primaryActionId")
        if action is not None:
            predicted_actions += 1
            actions += int(action not in _candidate_ids(user))
        known_evidence = {
            item.get("evidenceId")
            for item in user.get("evidence", [])
            if isinstance(item, dict)
        }
        for evidence_id in prediction.get("evidenceIds", []):
            predicted_evidence += 1
            evidence += int(evidence_id not in known_evidence)
        for reason in prediction.get("reasonCodes", []):
            predicted_reasons += 1
            reasons += int(reason not in allowed_reason_codes)
        message = prediction.get("message")
        if isinstance(message, str) and message:
            messages += 1
            claims += int(
                any(
                    token in message.lower()
                    for token in (
                        "win rate",
                        "patch ",
                        "cooldown",
                        "respawn",
                        "tier ",
                    )
                )
            )
    return {
        "inventedActionRate": actions / predicted_actions if predicted_actions else 0.0,
        "inventedEvidenceRate": (
            evidence / predicted_evidence if predicted_evidence else 0.0
        ),
        "unsupportedReasonCodeRate": (
            reasons / predicted_reasons if predicted_reasons else 0.0
        ),
        "unsupportedClaimRate": claims / messages if messages else 0.0,
    }
