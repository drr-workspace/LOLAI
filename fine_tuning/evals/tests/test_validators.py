from __future__ import annotations

import json
from pathlib import Path

from evals.validators import (
    validate_canonical,
    validate_counterfactuals,
    validate_episodes,
    validate_release,
    validate_split_leakage,
)


def _jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _scenario(identifier: str, decision: str = "SHOW") -> dict:
    return {
        "scenarioId": identifier,
        "familyId": "family",
        "splitGroup": "group",
        "sourceType": "COUNTERFACTUAL",
        "seed": 1,
        "parentScenarioId": None,
        "counterfactualPairId": "pair",
        "episodeId": None,
        "episodeStep": None,
        "causalSignature": "a" * 64,
        "reviewStatus": "GENERATED",
        "input": {"requestId": identifier, "value": identifier},
        "expectedOutput": {
            "decision": decision,
            "primaryActionId": "action" if decision == "SHOW" else None,
        },
    }


def test_counterfactual_validator_detects_missing_parent(
    tmp_path: Path,
) -> None:
    scenarios = tmp_path / "scenarios.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    _jsonl(scenarios, [_scenario("child")])
    _jsonl(
        pairs,
        [
            {
                "parentScenarioId": "missing",
                "counterfactualScenarioId": "child",
            }
        ],
    )
    errors = validate_counterfactuals.validate(scenarios, pairs)
    assert any("parent inesistente" in error for error in errors)


def test_episode_validator_detects_non_monotonic_time(
    tmp_path: Path,
) -> None:
    scenarios = tmp_path / "scenarios.jsonl"
    episodes = tmp_path / "episodes.jsonl"
    values = []
    for index in range(3):
        item = _scenario(f"s{index}")
        item.update({"episodeId": "episode", "episodeStep": index})
        values.append(item)
    _jsonl(scenarios, values)
    _jsonl(
        episodes,
        [
            {
                "episodeId": "episode",
                "steps": [
                    {
                        "scenarioId": f"s{index}",
                        "elapsedSeconds": 0,
                        "delta": {"changed": True},
                        "expectedTransition": "SHOW_TO_SHOW",
                    }
                    for index in range(3)
                ],
            }
        ],
    )
    errors = validate_episodes.validate(scenarios, episodes)
    assert any("tempo non monotono" in error for error in errors)


def test_release_validator_reports_missing_files(tmp_path: Path) -> None:
    report = validate_release.validate(tmp_path)
    assert not report["valid"]
    assert any("file mancante" in error for error in report["errors"])


def test_canonical_volatile_field_is_detected() -> None:
    paths = tuple(
        validate_canonical.volatile_paths(
            {"input": {"context": {"patchVersion": "volatile"}}}
        )
    )
    assert paths == ("$.input.context.patchVersion",)


def test_canonical_semantics_ignore_opaque_identifier_renaming() -> None:
    def record(suffix: str) -> dict:
        return {
            "input": {
                "evidence": [
                    {
                        "evidenceId": f"evidence_{suffix}",
                        "type": "OBSERVED_RUNTIME",
                    }
                ],
                "payload": {
                    "candidates": [
                        {
                            "actionId": f"action_{suffix}",
                            "type": "PURCHASE",
                            "effects": ["PEEL"],
                            "supportsFunctions": ["PEEL"],
                            "feasibility": 0.9,
                        }
                    ]
                },
            },
            "expectedOutput": {
                "decision": "SHOW",
                "primaryActionId": f"action_{suffix}",
                "reasonCodes": ["FILLS_MISSING_FUNCTION"],
                "evidenceIds": [f"evidence_{suffix}"],
            },
        }

    assert validate_canonical._semantic_output(
        record("first")
    ) == validate_canonical._semantic_output(record("second"))


def test_split_leakage_detects_identical_inputs(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    user = {"requestId": "request", "task": "MACRO_PRIORITY"}
    assistant = {"decision": "SHOW"}
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps(user)},
            {"role": "assistant", "content": json.dumps(assistant)},
        ]
    }
    for split in ("train", "valid", "test", "challenge"):
        current = json.loads(json.dumps(row))
        current["messages"][1]["content"] = json.dumps(
            {**user, "requestId": f"{split}-request"}
        )
        _jsonl(release / f"{split}.jsonl", [current])
    train = json.loads((release / "train.jsonl").read_text())
    test = json.loads((release / "test.jsonl").read_text())
    test["messages"][1]["content"] = train["messages"][1]["content"]
    _jsonl(release / "test.jsonl", [test])
    canonical = tmp_path / "scenarios.jsonl"
    _jsonl(canonical, [])
    errors = validate_split_leakage.validate(release, canonical)
    assert any(
        "input identico" in error or "requestId duplicato" in error
        for error in errors
    )
