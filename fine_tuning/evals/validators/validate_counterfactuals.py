from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evals.validators.validate_canonical import load_jsonl


IGNORED_FIELDS = {
    "scenarioId",
    "parentScenarioId",
    "counterfactualPairId",
    "causalSignature",
    "sourceType",
    "seed",
}


def classify_effect(parent: dict[str, Any], child: dict[str, Any]) -> str:
    first = parent["expectedOutput"]
    second = child["expectedOutput"]
    if (
        first.get("decision") == second.get("decision")
        and first.get("primaryActionId") == second.get("primaryActionId")
    ):
        return "SAME_DECISION"
    if second.get("decision") == "REQUEST_REFRESH":
        return "CHANGE_TO_REFRESH"
    if first.get("decision") == "REQUEST_REFRESH":
        return "CHANGE_FROM_REFRESH"
    if second.get("decision") == "SUPPRESS":
        return "CHANGE_TO_SUPPRESS"
    return "CHANGE_ACTION"


def changed_dimensions(
    parent: dict[str, Any], child: dict[str, Any]
) -> set[str]:
    dimensions: set[str] = set()
    for key in set(parent) | set(child):
        if key in IGNORED_FIELDS or key == "expectedOutput":
            continue
        if parent.get(key) != child.get(key):
            dimensions.add(key)
    if parent.get("input") != child.get("input"):
        dimensions.add("input")
    return dimensions


def validate(
    scenarios_path: Path,
    pairs_path: Path,
    split_locations: dict[str, str] | None = None,
) -> list[str]:
    scenarios, errors = load_jsonl(scenarios_path)
    pairs, pair_errors = load_jsonl(pairs_path)
    errors.extend(pair_errors)
    by_id = {item.get("scenarioId"): item for item in scenarios}
    for index, pair in enumerate(pairs, start=1):
        location = f"counterfactual-pairs.jsonl:{index}"
        parent_id = pair.get("parentScenarioId")
        child_id = pair.get("counterfactualScenarioId")
        if parent_id not in by_id:
            errors.append(f"{location}: parent inesistente {parent_id!r}")
            continue
        if child_id not in by_id:
            errors.append(f"{location}: counterfactual inesistente {child_id!r}")
            continue
        parent, child = by_id[parent_id], by_id[child_id]
        if child.get("parentScenarioId") != parent_id:
            errors.append(f"{location}: parentScenarioId incoerente")
        if (
            parent.get("counterfactualPairId")
            != child.get("counterfactualPairId")
            or parent.get("counterfactualPairId")
            != pair.get("counterfactualPairId")
        ):
            errors.append(f"{location}: counterfactualPairId incoerente")
        dimensions = changed_dimensions(parent, child)
        if dimensions != {"input"}:
            errors.append(
                f"{location}: mutazione causale non singola: {sorted(dimensions)}"
            )
        observed = classify_effect(parent, child)
        if observed != pair.get("expectedEffect"):
            errors.append(
                f"{location}: expectedEffect={pair.get('expectedEffect')}, "
                f"osservato={observed}"
            )
        if pair.get("observedEffect") != observed:
            errors.append(f"{location}: observedEffect non verificato")
        if split_locations is not None and (
            split_locations.get(str(parent_id))
            != split_locations.get(str(child_id))
        ):
            errors.append(f"{location}: coppia spezzata fra split")
    return errors


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=root / "datasets/canonical/releases/2.0.0",
    )
    args = parser.parse_args(argv)
    errors = validate(
        args.canonical_dir / "scenarios.jsonl",
        args.canonical_dir / "counterfactual-pairs.jsonl",
    )
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Counterfactuals: {'FAIL' if errors else 'PASS'}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
