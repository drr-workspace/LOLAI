from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from generators.realistic_adapter import (
    RealisticAdapter,
    SnapshotValidationError,
)
from generators.release_renderer import ReleaseRenderer
from generators.review_cli import main as review_main
from generators.review_queue import ReviewQueue


def valid_snapshot() -> dict[str, object]:
    return {
        "schemaVersion": "1.0.0",
        "snapshotId": "snapshot_opaque_1",
        "task": "THREAT_ASSESSMENT",
        "outputLocale": "en-US",
        "provenance": {
            "sourceKind": "RUNTIME",
            "sourceId": "runtime_source_opaque",
            "capturedAt": "2026-07-25T10:00:00Z",
            "collectorVersion": "1.0.0",
        },
        "context": {
            "observedAtGameSecond": 800,
            "freshnessSeconds": 2,
            "completeness": 95,
            "uncertainFields": [],
            "stateSignature": "abstract_state_a",
        },
        "team": {
            "allyFunctions": ["FRONTLINE"],
            "allyArchetypes": ["PROTECT_CARRY"],
            "enemyFunctions": ["BACKLINE_ACCESS"],
            "enemyArchetypes": ["DIVE"],
            "missingFunctions": ["PEEL"],
            "primaryWinCondition": "PROTECT_CARRY",
        },
        "threats": [
            {
                "entityId": "entity_opaque_a",
                "priority": 95,
                "patterns": ["BACKLINE_ACCESS"],
                "damageProfile": {
                    "physical": 60,
                    "magic": 30,
                    "true": 10,
                },
                "evidenceIds": ["evidence_opaque_a"],
            }
        ],
        "waveStates": {"region_opaque": "NEUTRAL"},
        "objectiveUrgency": 70,
        "visionConfidence": 80,
        "resourceAvailabilityNormalized": 65,
        "candidates": [
            {
                "actionId": "action_opaque_primary",
                "type": "PEEL",
                "effects": ["PEEL"],
                "supportsFunctions": ["PEEL"],
                "evidenceIds": ["evidence_opaque_a"],
                "counteredThreatIds": ["entity_opaque_a"],
                "winConditionTags": ["PROTECT_CARRY"],
                "feasibility": 95,
                "urgencyAlignment": 90,
                "opportunityCost": 5,
                "executionBurden": 5,
                "equivalenceKey": "equivalence_opaque_a",
            },
            {
                "actionId": "action_opaque_secondary",
                "type": "HOLD",
                "effects": [],
                "supportsFunctions": [],
                "evidenceIds": ["evidence_opaque_a"],
                "counteredThreatIds": [],
                "winConditionTags": [],
                "feasibility": 50,
                "urgencyAlignment": 20,
                "opportunityCost": 70,
                "executionBurden": 20,
                "equivalenceKey": "equivalence_opaque_b",
            },
        ],
        "evidence": [
            {
                "evidenceId": "evidence_opaque_a",
                "type": "OBSERVED_RUNTIME",
                "confidence": 95,
                "freshnessSeconds": 2,
                "conflictsWithEvidenceIds": [],
                "fact": {"abstractSignal": "primary_pressure"},
            }
        ],
        "recentAdvice": [],
    }


def test_adapter_validates_normalizes_and_labels() -> None:
    adapted = RealisticAdapter().adapt(valid_snapshot())

    assert adapted.scenario.source_type == "REALISTIC_ABSTRACTED"
    assert adapted.review_status == "NEEDS_REVIEW"
    assert adapted.scenario.context.completeness == 0.95
    assert adapted.scenario.candidates[0].feasibility == 0.95
    assert sum(adapted.scenario.threats[0].damage_profile) == pytest.approx(1)
    assert adapted.oracle_result.decision.decision == "SHOW"


def test_adapter_is_deterministic() -> None:
    adapter = RealisticAdapter()

    assert adapter.adapt(valid_snapshot()) == adapter.adapt(valid_snapshot())


def test_provenance_is_separate_from_model_messages() -> None:
    adapted = RealisticAdapter().adapt(valid_snapshot())
    row = ReleaseRenderer().render(
        adapted.scenario,
        adapted.oracle_result,
        message_intent="abstract_runtime_advice",
    )
    serialized = json.dumps(row)

    assert adapted.provenance["sourceId"] == "runtime_source_opaque"
    assert "runtime_source_opaque" not in serialized
    assert "capturedAt" not in serialized


@pytest.mark.parametrize(
    "forbidden_field",
    ("championName", "itemName", "runeName", "patchVersion"),
)
def test_snapshot_rejects_concrete_or_volatile_fields(
    forbidden_field: str,
) -> None:
    snapshot = valid_snapshot()
    snapshot[forbidden_field] = "not_allowed"

    with pytest.raises(SnapshotValidationError):
        RealisticAdapter().validate(snapshot)


def test_snapshot_rejects_patch_knowledge_evidence() -> None:
    snapshot = valid_snapshot()
    snapshot["evidence"][0]["type"] = "PATCH_KNOWLEDGE"

    with pytest.raises(SnapshotValidationError, match="PATCH_KNOWLEDGE"):
        RealisticAdapter().validate(snapshot)


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    (
        (("team", "allyFunctions"), ["UNKNOWN_FUNCTION"]),
        (("waveStates", "region_opaque"), "UNKNOWN_WAVE"),
        (("threats", 0, "patterns"), ["UNKNOWN_PATTERN"]),
        (("candidates", 0, "type"), "UNKNOWN_ACTION"),
    ),
)
def test_snapshot_rejects_non_ontological_values(
    path: tuple[object, ...], invalid_value: object
) -> None:
    snapshot = valid_snapshot()
    target: object = snapshot
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = invalid_value

    with pytest.raises((SnapshotValidationError, KeyError)):
        RealisticAdapter().validate(snapshot)


def test_snapshot_rejects_unknown_references() -> None:
    snapshot = valid_snapshot()
    snapshot["candidates"][0]["evidenceIds"] = ["unknown_evidence"]

    with pytest.raises(SnapshotValidationError, match="riferimenti"):
        RealisticAdapter().validate(snapshot)


def test_review_queue_prioritizes_and_deduplicates(tmp_path: Path) -> None:
    adapted = RealisticAdapter().adapt(valid_snapshot())
    queue = ReviewQueue(
        tmp_path / "review.jsonl", tmp_path / "audit.jsonl"
    )

    assert queue.enqueue(adapted)
    assert not queue.enqueue(adapted)
    entries = queue.list()

    assert len(entries) == 1
    assert entries[0]["reviewStatus"] == "NEEDS_REVIEW"
    assert entries[0]["reviewPriority"] > 0


def test_review_mutations_have_append_only_audit(tmp_path: Path) -> None:
    adapted = RealisticAdapter().adapt(valid_snapshot())
    queue_path = tmp_path / "review.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    queue = ReviewQueue(queue_path, audit_path)
    queue.enqueue(adapted)
    scenario_id = adapted.scenario.scenario_id

    queue.add_note(scenario_id, "Checked causal evidence.", "reviewer_a")
    queue.edit_expected(
        scenario_id, "confidence", 0.9, "reviewer_a"
    )
    queue.set_status(scenario_id, "APPROVED", "reviewer_a")

    audit = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["action"] for event in audit] == [
        "add-note",
        "edit-expected",
        "approved",
    ]
    assert all(event["reviewer"] == "reviewer_a" for event in audit)


def test_review_cli_supports_required_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    adapted = RealisticAdapter().adapt(valid_snapshot())
    queue_path = tmp_path / "review.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    export_path = tmp_path / "approved.jsonl"
    queue = ReviewQueue(queue_path, audit_path)
    queue.enqueue(adapted)
    scenario_id = adapted.scenario.scenario_id
    common = ["--queue", str(queue_path), "--audit", str(audit_path)]

    assert review_main([*common, "list"]) == 0
    assert review_main([*common, "show", scenario_id]) == 0
    assert review_main(
        [
            *common,
            "add-note",
            scenario_id,
            "Reviewed",
            "--reviewer",
            "reviewer_b",
        ]
    ) == 0
    assert review_main(
        [
            *common,
            "edit-expected",
            scenario_id,
            "confidence",
            "0.88",
            "--reviewer",
            "reviewer_b",
        ]
    ) == 0
    assert review_main(
        [
            *common,
            "approve",
            scenario_id,
            "--reviewer",
            "reviewer_b",
        ]
    ) == 0
    assert review_main(
        [*common, "export-approved", str(export_path)]
    ) == 0
    assert len(export_path.read_text(encoding="utf-8").splitlines()) == 1
    assert capsys.readouterr().err == ""


def test_reject_command_is_audited(tmp_path: Path) -> None:
    snapshot = deepcopy(valid_snapshot())
    snapshot["snapshotId"] = "snapshot_reject"
    adapted = RealisticAdapter().adapt(snapshot)
    queue_path = tmp_path / "review.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    queue = ReviewQueue(queue_path, audit_path)
    queue.enqueue(adapted)

    assert review_main(
        [
            "--queue",
            str(queue_path),
            "--audit",
            str(audit_path),
            "reject",
            adapted.scenario.scenario_id,
            "--reviewer",
            "reviewer_c",
        ]
    ) == 0
    assert queue.show(adapted.scenario.scenario_id)["reviewStatus"] == (
        "REJECTED"
    )
