from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeAlias


FINE_TUNING_ROOT = Path(__file__).resolve().parents[2]
if str(FINE_TUNING_ROOT) not in sys.path:
    sys.path.insert(0, str(FINE_TUNING_ROOT))

from generators.ontology_registry import (  # noqa: E402
    OntologyError,
    OntologyRegistry,
)


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

SPLITS: tuple[str, ...] = ("train", "valid", "test", "challenge")
REQUIRED_ENTRY_FIELDS: frozenset[str] = frozenset(
    {"id", "description", "stable"}
)
SCHEMA_ENUM_LOCATIONS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "tasks": ("common.schema.json", ("$defs", "task", "enum")),
    "decisions": (
        "advisor-output.schema.json",
        ("properties", "decision", "enum"),
    ),
    "priorities": (
        "advisor-output.schema.json",
        ("properties", "priority", "enum"),
    ),
    "source-types": (
        "canonical-scenario.schema.json",
        ("properties", "sourceType", "enum"),
    ),
    "review-statuses": (
        "canonical-scenario.schema.json",
        ("properties", "reviewStatus", "enum"),
    ),
}


def load_object(path: Path) -> dict[str, Any]:
    """Load an object-shaped UTF-8 JSON document."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: deve essere un oggetto JSON")
    return value


def validate_entry_structure(
    registry_name: str,
    entry: Any,
    location: str,
    errors: list[str],
) -> None:
    """Validate common and reason-code-specific entry fields."""
    if not isinstance(entry, dict):
        errors.append(f"{location}: deve essere un oggetto")
        return

    missing = REQUIRED_ENTRY_FIELDS - entry.keys()
    if missing:
        errors.append(f"{location}: campi mancanti: {sorted(missing)}")

    if not isinstance(entry.get("id"), str) or not entry.get("id"):
        errors.append(f"{location}.id: deve essere una stringa non vuota")
    if not isinstance(entry.get("description"), str) or not entry.get("description"):
        errors.append(
            f"{location}.description: deve essere una stringa non vuota"
        )
    if entry.get("stable") is not True:
        errors.append(f"{location}.stable: deve essere true")

    if registry_name != "reason-codes":
        return

    for field in ("allowedTasks", "decisionCompatibility"):
        values = entry.get(field)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
        ):
            errors.append(
                f"{location}.{field}: deve essere una lista non vuota di stringhe"
            )

    evidence_categories = entry.get("requiredEvidenceCategories")
    if evidence_categories is not None and (
        not isinstance(evidence_categories, list)
        or not evidence_categories
        or not all(
            isinstance(value, str) and value for value in evidence_categories
        )
    ):
        errors.append(
            f"{location}.requiredEvidenceCategories: "
            "deve essere una lista non vuota di stringhe"
        )


def validate_documents(
    ontology_dir: Path,
    registry: OntologyRegistry,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate every manifest-declared ontology JSON document."""
    documents: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    manifest_path = ontology_dir / "manifest.json"
    manifest = load_object(manifest_path)
    documents["manifest"] = manifest

    if manifest.get("ontologyVersion") != registry.version:
        errors.append(f"{manifest_path}: ontologyVersion incoerente")
    manifest_values = manifest.get("values")
    if not isinstance(manifest_values, list):
        return documents, [f"{manifest_path}.values: deve essere una lista"]
    for index, entry in enumerate(manifest_values):
        validate_entry_structure(
            "manifest",
            entry,
            f"{manifest_path}: $.values[{index}]",
            errors,
        )

    for registry_name in registry.registry_names:
        manifest_entry = next(
            (
                entry
                for entry in manifest_values
                if isinstance(entry, dict) and entry.get("id") == registry_name
            ),
            None,
        )
        if not isinstance(manifest_entry, dict):
            errors.append(f"{manifest_path}: registro mancante: {registry_name}")
            continue
        filename = manifest_entry.get("file")
        if not isinstance(filename, str):
            errors.append(f"{manifest_path}: file non valido per {registry_name}")
            continue

        path = ontology_dir / filename
        document = load_object(path)
        documents[registry_name] = document
        if document.get("ontologyVersion") != registry.version:
            errors.append(f"{path}: ontologyVersion incoerente")
        values = document.get("values")
        if not isinstance(values, list):
            errors.append(f"{path}: $.values deve essere una lista")
            continue
        for index, entry in enumerate(values):
            validate_entry_structure(
                registry_name,
                entry,
                f"{path}: $.values[{index}]",
                errors,
            )

    return documents, errors


def validate_reason_references(
    document: Mapping[str, Any],
    registry: OntologyRegistry,
) -> list[str]:
    """Verify every reason-code taxonomy reference."""
    errors: list[str] = []
    values = document.get("values")
    if not isinstance(values, list):
        return errors
    for index, entry in enumerate(values):
        if not isinstance(entry, dict):
            continue
        reason_id = entry.get("id", f"values[{index}]")
        for task in entry.get("allowedTasks", []):
            if isinstance(task, str) and not registry.contains("tasks", task):
                errors.append(
                    f"reason-codes.json: {reason_id}.allowedTasks: "
                    f"task inesistente {task!r}"
                )
        for decision in entry.get("decisionCompatibility", []):
            if isinstance(decision, str) and not registry.contains(
                "decisions", decision
            ):
                errors.append(
                    f"reason-codes.json: {reason_id}.decisionCompatibility: "
                    f"decisione inesistente {decision!r}"
                )
        for category in entry.get("requiredEvidenceCategories", []):
            if isinstance(category, str) and not registry.contains(
                "evidence-types", category
            ):
                errors.append(
                    f"reason-codes.json: {reason_id}."
                    f"requiredEvidenceCategories: evidenza inesistente "
                    f"{category!r}"
                )
    return errors


def nested_value(document: Mapping[str, Any], path: Iterable[str]) -> Any:
    """Read a nested mapping value by path."""
    value: Any = document
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def validate_schema_enums(
    schema_dir: Path,
    registry: OntologyRegistry,
) -> list[str]:
    """Require ontology values to match every enum defined by Batch 1A."""
    errors: list[str] = []
    for registry_name, (filename, enum_path) in SCHEMA_ENUM_LOCATIONS.items():
        schema_path = schema_dir / filename
        schema = load_object(schema_path)
        raw_values = nested_value(schema, enum_path)
        if not isinstance(raw_values, list) or not all(
            isinstance(value, str) for value in raw_values
        ):
            errors.append(
                f"{schema_path}: enum non trovata in {'.'.join(enum_path)}"
            )
            continue
        schema_values = set(raw_values)
        ontology_values = set(registry.ids(registry_name))
        if schema_values != ontology_values:
            errors.append(
                f"{schema_path}: enum {registry_name} non allineata; "
                f"solo schema={sorted(schema_values - ontology_values)}, "
                f"solo ontologia={sorted(ontology_values - schema_values)}"
            )
    return errors


def add_strings(target: set[str], value: Any) -> None:
    """Add string members from a JSON array to a set."""
    if isinstance(value, list):
        target.update(item for item in value if isinstance(item, str))


def collect_input_values(
    user: Mapping[str, Any],
    used: dict[str, set[str]],
) -> None:
    """Collect ontology-backed values from one parsed user document."""
    task = user.get("task")
    if isinstance(task, str):
        used["tasks"].add(task)
    add_strings(used["evidence-types"], [
        item.get("type")
        for item in user.get("evidence", [])
        if isinstance(item, dict)
    ])

    for advice in user.get("recentAdvice", []):
        if not isinstance(advice, dict):
            continue
        category = advice.get("category")
        decision = advice.get("decision")
        if isinstance(category, str):
            used["tasks"].add(category)
        if isinstance(decision, str):
            used["decisions"].add(decision)
        add_strings(used["reason-codes"], advice.get("reasonCodes"))

    payload = user.get("payload")
    if not isinstance(payload, dict):
        return
    candidates = payload.get("candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            action_type = candidate.get("type")
            if isinstance(action_type, str):
                used["action-types"].add(action_type)
            add_strings(
                used["team-functions"], candidate.get("supportsFunctions")
            )

    for key in ("allyFunctions", "enemyFunctions", "playerFunctions"):
        add_strings(used["team-functions"], payload.get(key))
    for key in ("allyArchetypes", "enemyArchetypes"):
        add_strings(used["archetypes"], payload.get(key))
    add_strings(used["threat-patterns"], payload.get("opponentThreatPatterns"))

    team_plan = payload.get("teamPlan")
    if isinstance(team_plan, dict):
        add_strings(used["team-functions"], team_plan.get("missingFunctions"))
        add_strings(used["archetypes"], team_plan.get("compositionTags"))
        win_condition = team_plan.get("primaryWinCondition")
        if isinstance(win_condition, str):
            used["archetypes"].add(win_condition)

    threats = payload.get("threats")
    if isinstance(threats, list):
        for threat in threats:
            if isinstance(threat, dict):
                add_strings(used["threat-patterns"], threat.get("patterns"))

    lane_state = payload.get("laneState")
    if isinstance(lane_state, str):
        used["wave-states"].add(lane_state)
    wave_states = payload.get("waveStates")
    if isinstance(wave_states, dict):
        used["wave-states"].update(
            value for value in wave_states.values() if isinstance(value, str)
        )

    candidate_advice = payload.get("candidateAdvice")
    if isinstance(candidate_advice, dict):
        category = candidate_advice.get("category")
        if isinstance(category, str):
            used["tasks"].add(category)
        add_strings(
            used["reason-codes"], candidate_advice.get("reasonCodes")
        )


def collect_output_values(
    assistant: Mapping[str, Any],
    used: dict[str, set[str]],
) -> None:
    """Collect ontology-backed values from one parsed assistant document."""
    scalar_fields = {
        "decision": "decisions",
        "category": "tasks",
        "priority": "priorities",
    }
    for field, registry_name in scalar_fields.items():
        value = assistant.get(field)
        if isinstance(value, str):
            used[registry_name].add(value)
    add_strings(used["reason-codes"], assistant.get("reasonCodes"))
    add_strings(used["recheck-triggers"], assistant.get("recheckTriggers"))


def validate_reason_usage(
    assistant: Mapping[str, Any],
    registry: OntologyRegistry,
    location: str,
) -> list[str]:
    """Validate reason existence and task/decision compatibility."""
    errors: list[str] = []
    task = assistant.get("category")
    decision = assistant.get("decision")
    reasons = assistant.get("reasonCodes")
    if not isinstance(reasons, list):
        return errors
    for reason in reasons:
        if not isinstance(reason, str):
            continue
        if not registry.contains("reason-codes", reason):
            errors.append(f"{location}.reasonCodes: valore inesistente {reason!r}")
            continue
        entry = registry.require("reason-codes", reason)
        allowed_tasks = entry.get("allowedTasks", ())
        compatible_decisions = entry.get("decisionCompatibility", ())
        if isinstance(task, str) and task not in allowed_tasks:
            errors.append(
                f"{location}.reasonCodes: {reason!r} non ammesso per {task!r}"
            )
        if isinstance(decision, str) and decision not in compatible_decisions:
            errors.append(
                f"{location}.reasonCodes: {reason!r} incompatibile con "
                f"{decision!r}"
            )
    return errors


def validate_release(
    release_dir: Path,
    registry: OntologyRegistry,
) -> tuple[int, dict[str, set[str]], list[str]]:
    """Validate ontology coverage across every release row."""
    used = {name: set() for name in registry.registry_names}
    errors: list[str] = []
    rows = 0
    for split in SPLITS:
        path = release_dir / f"{split}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                rows += 1
                location = f"{path}:{line_number}"
                try:
                    row = json.loads(raw_line)
                    user = json.loads(row["messages"][1]["content"])
                    assistant = json.loads(row["messages"][2]["content"])
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                    errors.append(f"{location}: struttura non leggibile: {error}")
                    continue
                if not isinstance(user, dict) or not isinstance(assistant, dict):
                    errors.append(f"{location}: user e assistant devono essere oggetti")
                    continue
                collect_input_values(user, used)
                collect_output_values(assistant, used)
                errors.extend(
                    validate_reason_usage(
                        assistant,
                        registry,
                        f"{location}: assistant",
                    )
                )

    for registry_name, used_ids in used.items():
        unknown = used_ids - registry.ids(registry_name)
        if unknown:
            errors.append(
                f"release 1.0.0: valori {registry_name} non registrati: "
                f"{sorted(unknown)}"
            )
    return rows, used, errors


def unused_warnings(
    registry: OntologyRegistry,
    used: Mapping[str, set[str]],
) -> list[str]:
    """Report ontology values not exercised by the immutable release."""
    warnings: list[str] = []
    for registry_name in registry.registry_names:
        unused = registry.ids(registry_name) - used.get(registry_name, set())
        if unused:
            warnings.append(
                f"{registry_name}: valori non usati nella release 1.0.0: "
                f"{', '.join(sorted(unused))}"
            )
    return warnings


def main() -> int:
    """Validate ontology structure, references, schemas and release coverage."""
    ontology_dir = FINE_TUNING_ROOT / "ontology"
    schema_dir = FINE_TUNING_ROOT / "contracts" / "task-schemas"
    release_dir = FINE_TUNING_ROOT / "datasets" / "releases" / "1.0.0"
    try:
        registry = OntologyRegistry(ontology_dir)
        documents, errors = validate_documents(ontology_dir, registry)
        reason_document = documents.get("reason-codes", {})
        errors.extend(validate_reason_references(reason_document, registry))
        errors.extend(validate_schema_enums(schema_dir, registry))
        rows, used, release_errors = validate_release(release_dir, registry)
        errors.extend(release_errors)
    except (OSError, ValueError, json.JSONDecodeError, OntologyError) as error:
        print(f"Validazione ontologia fallita: {error}", file=sys.stderr)
        return 1

    warnings = unused_warnings(registry, used)
    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"Validazione ontologia fallita: {len(errors)} errori.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Ontologia valida: {len(registry.registry_names)} registri, "
        f"{rows} righe verificate, {len(warnings)} avvisi."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
