from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def compute(
    samples: Sequence[Mapping[str, Any]],
    *,
    schema_valid: Callable[[object], bool],
    maximum_message_length: int = 180,
) -> dict[str, float]:
    total = len(samples)
    if not total:
        return {
            "jsonValidity": 0.0,
            "schemaConformance": 0.0,
            "localeCompliance": 0.0,
            "maximumMessageLengthCompliance": 0.0,
        }
    valid_json = schema = locale = length = 0
    for sample in samples:
        prediction = sample.get("prediction")
        if sample.get("predictionValid") and isinstance(prediction, dict):
            valid_json += 1
            schema += int(schema_valid(prediction))
            message = prediction.get("message")
            if isinstance(message, str):
                length += int(len(message) <= maximum_message_length)
                locale += int(
                    _locale_compliant(
                        message, sample.get("input", {}).get("outputLocale")
                    )
                )
    return {
        "jsonValidity": valid_json / total,
        "schemaConformance": schema / total,
        "localeCompliance": locale / total,
        "maximumMessageLengthCompliance": length / total,
    }


def _locale_compliant(message: str, locale: object) -> bool:
    if not message:
        return True
    lowered = message.lower()
    italian = {"scegli", "priorità", "minaccia", "squadra", "evidenza"}
    english = {"choose", "priority", "threat", "team", "evidence"}
    if locale == "en-US":
        return not any(word in lowered for word in italian)
    if locale == "it-IT":
        return not any(word in lowered for word in english)
    return False
