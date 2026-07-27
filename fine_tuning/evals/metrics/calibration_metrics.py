from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def compute(
    samples: Sequence[Mapping[str, Any]], bins: int = 10
) -> dict[str, Any]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    by_decision: defaultdict[str, list[float]] = defaultdict(list)
    for sample in samples:
        prediction = sample.get("prediction") or {}
        confidence = prediction.get("confidence")
        if not isinstance(confidence, (int, float)):
            continue
        value = min(1.0, max(0.0, float(confidence)))
        correct = prediction.get("decision") == sample["expected"].get("decision")
        buckets[min(bins - 1, int(value * bins))].append((value, correct))
        decision = prediction.get("decision")
        if isinstance(decision, str):
            by_decision[decision].append(value)
    total = sum(len(bucket) for bucket in buckets)
    rendered: list[dict[str, Any]] = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        confidence = (
            sum(item[0] for item in bucket) / len(bucket) if bucket else 0.0
        )
        accuracy = (
            sum(item[1] for item in bucket) / len(bucket) if bucket else 0.0
        )
        if total:
            ece += len(bucket) / total * abs(accuracy - confidence)
        rendered.append(
            {
                "minimum": index / bins,
                "maximum": (index + 1) / bins,
                "count": len(bucket),
                "meanConfidence": confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "confidenceBins": rendered,
        "expectedCalibrationError": round(ece, 12),
        "accuracyPerConfidenceBand": {
            f"{item['minimum']:.1f}-{item['maximum']:.1f}": item["accuracy"]
            for item in rendered
        },
        "meanConfidenceByDecision": {
            decision: sum(values) / len(values)
            for decision, values in sorted(by_decision.items())
        },
    }
