from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

SPLIT_NAMES: tuple[str, ...] = ("train", "valid", "test", "challenge")
SCHEMA_NAMES: tuple[str, ...] = (
    "common.schema.json",
    "canonical-scenario.schema.json",
    "advisor-input.schema.json",
    "advisor-output.schema.json",
    "release-row.schema.json",
    "composition-plan.payload.schema.json",
    "matchup-plan.payload.schema.json",
    "itemization-decision.payload.schema.json",
    "macro-priority.payload.schema.json",
    "threat-assessment.payload.schema.json",
    "advice-suppression.payload.schema.json",
)


def load_json(path: Path) -> JsonValue:
    """Load one UTF-8 JSON document."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_schemas(schema_dir: Path) -> dict[str, Mapping[str, Any]]:
    """Load every required schema and reject missing or non-object documents."""
    schemas: dict[str, Mapping[str, Any]] = {}
    for name in SCHEMA_NAMES:
        path = schema_dir / name
        value = load_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: lo schema deve essere un oggetto JSON")
        schemas[name] = value
    return schemas


def build_registry(schemas: Mapping[str, Mapping[str, Any]]) -> Registry[Any]:
    """Build a registry keyed by each local schema filename."""
    resources = (
        (name, Resource.from_contents(schema))
        for name, schema in schemas.items()
    )
    return Registry().with_resources(resources)


def build_validators(
    schemas: Mapping[str, Mapping[str, Any]],
    registry: Registry[Any],
) -> dict[str, Draft202012Validator]:
    """Check each schema and create validators sharing the local registry."""
    validators: dict[str, Draft202012Validator] = {}
    for name, schema in schemas.items():
        Draft202012Validator.check_schema(schema)
        validators[name] = Draft202012Validator(schema, registry=registry)
    return validators


def json_path(parts: Iterable[Any]) -> str:
    """Render a jsonschema path as an unambiguous JSON path."""
    result = "$"
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            encoded = json.dumps(str(part), ensure_ascii=False)
            result += f"[{encoded}]"
    return result


def format_validation_error(
    path: Path,
    line_number: int,
    document_name: str,
    error: ValidationError,
) -> str:
    """Format one validation failure with file, line and instance path."""
    instance_path = json_path(error.absolute_path)
    return (
        f"{path}:{line_number}: {document_name} {instance_path}: "
        f"{error.message}"
    )


def sorted_errors(
    validator: Draft202012Validator,
    instance: JsonValue,
) -> list[ValidationError]:
    """Return deterministic validation errors ordered by their JSON path."""
    return sorted(
        validator.iter_errors(instance),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )


def parse_message_json(
    content: Any,
    path: Path,
    line_number: int,
    message_name: str,
    errors: list[str],
) -> JsonValue | None:
    """Parse JSON serialized in a message and report its precise location."""
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        errors.append(
            f"{path}:{line_number}: {message_name} $: JSON non valido: "
            f"{error.msg} (colonna {error.colno})"
        )
        return None


def validate_message_document(
    validator: Draft202012Validator,
    document: JsonValue,
    path: Path,
    line_number: int,
    document_name: str,
    errors: list[str],
) -> None:
    """Validate one parsed user or assistant document."""
    errors.extend(
        format_validation_error(path, line_number, document_name, error)
        for error in sorted_errors(validator, document)
    )


def validate_row(
    row: JsonValue,
    path: Path,
    line_number: int,
    validators: Mapping[str, Draft202012Validator],
    errors: list[str],
) -> None:
    """Validate the release envelope and its serialized model documents."""
    row_errors = sorted_errors(validators["release-row.schema.json"], row)
    errors.extend(
        format_validation_error(path, line_number, "release-row", error)
        for error in row_errors
    )

    if not isinstance(row, dict):
        return
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return
    user_message = messages[1]
    assistant_message = messages[2]
    if not isinstance(user_message, dict) or not isinstance(assistant_message, dict):
        return

    user = parse_message_json(
        user_message.get("content"), path, line_number, "user", errors
    )
    assistant = parse_message_json(
        assistant_message.get("content"), path, line_number, "assistant", errors
    )

    if user is not None:
        validate_message_document(
            validators["advisor-input.schema.json"],
            user,
            path,
            line_number,
            "user",
            errors,
        )
    if assistant is not None:
        validate_message_document(
            validators["advisor-output.schema.json"],
            assistant,
            path,
            line_number,
            "assistant",
            errors,
        )


def validate_split(
    path: Path,
    validators: Mapping[str, Draft202012Validator],
) -> tuple[int, list[str]]:
    """Validate every non-empty JSONL row in one release split."""
    errors: list[str] = []
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            row_count += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                errors.append(
                    f"{path}:{line_number}: release-row $: JSONL non valido: "
                    f"{error.msg} (colonna {error.colno})"
                )
                continue
            validate_row(row, path, line_number, validators, errors)
    return row_count, errors


def project_paths() -> tuple[Path, Path]:
    """Resolve contract and immutable release directories from this script."""
    fine_tuning_root = Path(__file__).resolve().parents[2]
    return (
        fine_tuning_root / "contracts" / "task-schemas",
        fine_tuning_root / "datasets" / "releases" / "1.0.0",
    )


def main() -> int:
    """Load contracts and validate all release 1.0.0 splits."""
    schema_dir, release_dir = project_paths()
    try:
        schemas = load_schemas(schema_dir)
        registry = build_registry(schemas)
        validators = build_validators(schemas, registry)
    except (OSError, ValueError, json.JSONDecodeError, SchemaError) as error:
        print(f"Errore caricamento contratti: {error}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    total_rows = 0
    for split_name in SPLIT_NAMES:
        split_path = release_dir / f"{split_name}.jsonl"
        try:
            row_count, errors = validate_split(split_path, validators)
        except OSError as error:
            all_errors.append(f"{split_path}: {error}")
            continue
        total_rows += row_count
        all_errors.extend(errors)
        print(
            f"{split_path}: {row_count} righe, "
            f"{len(errors)} errori"
        )

    if all_errors:
        print("\nErrori di validazione:", file=sys.stderr)
        for error in all_errors:
            print(error, file=sys.stderr)
        print(
            f"\nValidazione fallita: {len(all_errors)} errori su "
            f"{total_rows} righe.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Validazione completata: {len(schemas)} schema caricati e "
        f"{total_rows} righe valide."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
