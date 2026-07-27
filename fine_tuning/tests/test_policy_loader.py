from __future__ import annotations

import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from fine_tuning.generators.policy_loader import (
    InconsistentPolicyVersionError,
    MissingPolicyKeyError,
    PolicyError,
    load_policies,
)


FINE_TUNING_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = FINE_TUNING_ROOT / "policies"
ONTOLOGY_DIR = FINE_TUNING_ROOT / "ontology"


def copy_policies(tmp_path: Path) -> Path:
    target = tmp_path / "policies"
    shutil.copytree(POLICY_DIR, target)
    return target


def read_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def write_document(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_complete_policy_bundle_loads() -> None:
    bundle = load_policies(POLICY_DIR, ONTOLOGY_DIR)

    assert bundle.policy_version == "1.0.0"
    assert bundle.ontology_version == "1.0.0"
    assert len(bundle.scoring.components) == 11
    assert bundle.decision.precedence[0].decision == "REQUEST_REFRESH"
    assert sum(bundle.generation.task_distribution.values()) == pytest.approx(1)
    assert bundle.split.allow_random_per_row is False


def test_policy_dataclasses_and_mappings_are_immutable() -> None:
    bundle = load_policies(POLICY_DIR, ONTOLOGY_DIR)

    with pytest.raises(FrozenInstanceError):
        bundle.policy_version = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        bundle.generation.task_distribution["COMPOSITION_PLAN"] = 0.0  # type: ignore[index]


def test_missing_key_fails(tmp_path: Path) -> None:
    policy_dir = copy_policies(tmp_path)
    path = policy_dir / "decision-policy.json"
    document = read_document(path)
    thresholds = document["thresholds"]
    assert isinstance(thresholds, dict)
    del thresholds["minimumCompleteness"]
    write_document(path, document)

    with pytest.raises(MissingPolicyKeyError):
        load_policies(policy_dir, ONTOLOGY_DIR)


def test_inconsistent_version_fails(tmp_path: Path) -> None:
    policy_dir = copy_policies(tmp_path)
    path = policy_dir / "confidence-policy.json"
    document = read_document(path)
    document["policyVersion"] = "9.9.9"
    write_document(path, document)

    with pytest.raises(InconsistentPolicyVersionError):
        load_policies(policy_dir, ONTOLOGY_DIR)


def test_invalid_distribution_fails(tmp_path: Path) -> None:
    policy_dir = copy_policies(tmp_path)
    path = policy_dir / "generation-policy.json"
    document = read_document(path)
    distribution = document["decisionDistribution"]
    assert isinstance(distribution, dict)
    distribution["SHOW"] = 0.9
    write_document(path, document)

    with pytest.raises(PolicyError, match="sommar"):
        load_policies(policy_dir, ONTOLOGY_DIR)


def test_unknown_reason_code_fails(tmp_path: Path) -> None:
    policy_dir = copy_policies(tmp_path)
    path = policy_dir / "scoring-policy.json"
    document = read_document(path)
    components = document["components"]
    assert isinstance(components, list)
    first_component = components[0]
    assert isinstance(first_component, dict)
    first_component["reasonCodes"] = ["NOT_A_REASON_CODE"]
    write_document(path, document)

    with pytest.raises(PolicyError, match="reason code inesistente"):
        load_policies(policy_dir, ONTOLOGY_DIR)


def test_invalid_task_reference_fails(tmp_path: Path) -> None:
    policy_dir = copy_policies(tmp_path)
    path = policy_dir / "generation-policy.json"
    document = read_document(path)
    distribution = document["taskDistribution"]
    assert isinstance(distribution, dict)
    distribution["NOT_A_TASK"] = distribution.pop("COMPOSITION_PLAN")
    write_document(path, document)

    with pytest.raises(PolicyError, match="chiavi non allineate"):
        load_policies(policy_dir, ONTOLOGY_DIR)
