from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from fine_tuning.generators.ontology_registry import (
    DuplicateOntologyIdError,
    InconsistentOntologyVersionError,
    OntologyRegistry,
    UnknownOntologyValueError,
)


ONTOLOGY_DIR = Path(__file__).resolve().parents[1] / "ontology"


def copy_ontology(tmp_path: Path) -> Path:
    target = tmp_path / "ontology"
    shutil.copytree(ONTOLOGY_DIR, target)
    return target


def read_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def write_document(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_duplicate_id_fails(tmp_path: Path) -> None:
    ontology_dir = copy_ontology(tmp_path)
    path = ontology_dir / "tasks.json"
    document = read_document(path)
    values = document["values"]
    assert isinstance(values, list)
    values.append(dict(values[0]))
    write_document(path, document)

    with pytest.raises(DuplicateOntologyIdError):
        OntologyRegistry(ontology_dir)


def test_inconsistent_version_fails(tmp_path: Path) -> None:
    ontology_dir = copy_ontology(tmp_path)
    path = ontology_dir / "decisions.json"
    document = read_document(path)
    document["ontologyVersion"] = "9.9.9"
    write_document(path, document)

    with pytest.raises(InconsistentOntologyVersionError):
        OntologyRegistry(ontology_dir)


def test_unknown_reason_code_fails() -> None:
    registry = OntologyRegistry(ONTOLOGY_DIR)

    with pytest.raises(UnknownOntologyValueError):
        registry.require("reason-codes", "NOT_A_REASON_CODE")


def test_invalid_task_fails() -> None:
    registry = OntologyRegistry(ONTOLOGY_DIR)

    with pytest.raises(UnknownOntologyValueError):
        registry.require("tasks", "NOT_A_TASK")


def test_complete_ontology_loads_read_only() -> None:
    registry = OntologyRegistry(ONTOLOGY_DIR)

    assert registry.version == "1.0.0"
    assert len(registry.registry_names) == 13
    assert registry.contains("tasks", "COMPOSITION_PLAN")
    assert registry.require("decisions", "SHOW")["stable"] is True
    assert registry.values("priorities")

    entry = registry.require("reason-codes", "STALE_CONTEXT")
    with pytest.raises(TypeError):
        entry["description"] = "changed"  # type: ignore[index]
