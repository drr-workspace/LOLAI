from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from evals.validators.validate_contracts import (
    build_registry,
    build_validators,
    format_validation_error,
    load_schemas,
    sorted_errors,
)
from evals.validators.validate_dataset import (
    FORBIDDEN_VOLATILE_KEYS,
)


SIGNATURE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_TYPES = {
    "SYNTHETIC",
    "COUNTERFACTUAL",
    "ADVERSARIAL",
    "TEMPORAL",
    "REALISTIC_ABSTRACTED",
    "HUMAN_REVIEWED",
}
REVIEW_STATUSES = {
    "GENERATED",
    "AUTO_VALIDATED",
    "NEEDS_REVIEW",
    "APPROVED",
    "REJECTED",
}


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return records, [f"{path}: file mancante"]
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{path}:{line_number}: JSON non valido: {error}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{path}:{line_number}: oggetto atteso")
                continue
            records.append(value)
    return records, errors


def volatile_paths(value: object, prefix: str = "$") -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{prefix}.{key}"
            if key in FORBIDDEN_VOLATILE_KEYS:
                yield location
            yield from volatile_paths(child, location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from volatile_paths(child, f"{prefix}[{index}]")


def validate_records(
    records: list[dict[str, Any]], schema_dir: Path
) -> list[str]:
    schemas = load_schemas(schema_dir)
    registry = build_registry(schemas)
    validator = build_validators(schemas, registry)[
        "canonical-scenario.schema.json"
    ]
    errors: list[str] = []
    seen_ids: set[str] = set()
    signature_outputs: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        location = f"scenarios.jsonl:{index}"
        for error in sorted_errors(validator, record):
            errors.append(format_validation_error(location, error))
        scenario_id = record.get("scenarioId")
        if scenario_id in seen_ids:
            errors.append(f"{location}: scenarioId duplicato {scenario_id!r}")
        elif isinstance(scenario_id, str):
            seen_ids.add(scenario_id)
        if record.get("sourceType") not in SOURCE_TYPES:
            errors.append(f"{location}: sourceType non valido")
        if record.get("reviewStatus") not in REVIEW_STATUSES:
            errors.append(f"{location}: reviewStatus non valido")
        signature = record.get("causalSignature")
        if not isinstance(signature, str) or not SIGNATURE.fullmatch(signature):
            errors.append(f"{location}: causalSignature non SHA-256")
        output = record.get("expectedOutput")
        if isinstance(signature, str) and isinstance(output, dict):
            semantic = json.dumps(
                _semantic_output(record),
                sort_keys=True,
            )
            previous = signature_outputs.setdefault(signature, semantic)
            if previous != semantic:
                errors.append(
                    f"{location}: stessa causalSignature con output incompatibile"
                )
            _validate_oracle_projection(record, errors, location)
        for volatile in volatile_paths(record):
            errors.append(f"{location}: campo volatile vietato {volatile}")
    return errors


def _semantic_output(record: Mapping[str, Any]) -> dict[str, Any]:
    user = record.get("input")
    output = record.get("expectedOutput")
    if not isinstance(user, dict) or not isinstance(output, dict):
        return {}
    payload = user.get("payload")
    candidates = (
        payload.get("candidates", [])
        if isinstance(payload, dict)
        else []
    )
    primary_id = output.get("primaryActionId")
    primary = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("actionId") == primary_id
        ),
        None,
    )
    primary_semantics = (
        {
            "type": primary.get("type"),
            "effects": sorted(primary.get("effects", [])),
            "supportsFunctions": sorted(
                primary.get("supportsFunctions", [])
            ),
            "feasibilityBand": _unit_bucket(
                primary.get("feasibility")
            ),
        }
        if isinstance(primary, dict)
        else None
    )
    evidence_by_id = {
        evidence.get("evidenceId"): evidence
        for evidence in user.get("evidence", [])
        if isinstance(evidence, dict)
    }
    evidence_types = sorted(
        {
            evidence_by_id[evidence_id].get("type")
            for evidence_id in output.get("evidenceIds", [])
            if evidence_id in evidence_by_id
        }
    )
    return {
        "decision": output.get("decision"),
        "primaryAction": primary_semantics,
        "reasonCodes": sorted(output.get("reasonCodes", [])),
        "evidenceTypes": evidence_types,
    }


def _unit_bucket(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return min(4, max(0, int(float(value) * 5)))


def _validate_oracle_projection(
    record: Mapping[str, Any], errors: list[str], location: str
) -> None:
    user = record.get("input")
    output = record.get("expectedOutput")
    if not isinstance(user, dict) or not isinstance(output, dict):
        return
    evidence_ids = {
        item.get("evidenceId")
        for item in user.get("evidence", [])
        if isinstance(item, dict)
    }
    unknown = set(output.get("evidenceIds", [])) - evidence_ids
    if unknown:
        errors.append(f"{location}: trace usa evidenceId ignoti {sorted(unknown)}")
    decision = output.get("decision")
    primary = output.get("primaryActionId")
    message = output.get("message")
    if decision == "SHOW" and (not primary or not message):
        errors.append(f"{location}: proiezione oracle SHOW incoerente")
    if decision in {"SUPPRESS", "REQUEST_REFRESH"} and (
        primary is not None or message != ""
    ):
        errors.append(f"{location}: proiezione oracle di astensione incoerente")


def validate_file(path: Path, schema_dir: Path) -> list[str]:
    records, errors = load_jsonl(path)
    errors.extend(validate_records(records, schema_dir))
    return errors


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical",
        type=Path,
        default=root / "datasets/canonical/releases/2.0.0/scenarios.jsonl",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=root / "contracts/task-schemas",
    )
    args = parser.parse_args(argv)
    errors = validate_file(args.canonical, args.schema_dir)
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Canonical: {'FAIL' if errors else 'PASS'} ({len(errors)} errori)")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
