from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evals.validators.validate_contracts import (
    build_registry,
    build_validators,
    load_schemas,
    validate_row,
)
from generators.message_renderer import MessageRenderer
from generators.episodes import EpisodeGenerator
from generators.release_renderer import ReleaseRenderer
from generators.sampler import ScenarioSampler


SAMPLER = ScenarioSampler()
RENDERER = ReleaseRenderer()
SCHEMA_DIR = (
    Path(__file__).resolve().parents[2] / "contracts" / "task-schemas"
)
SCHEMAS = load_schemas(SCHEMA_DIR)
VALIDATORS = build_validators(SCHEMAS, build_registry(SCHEMAS))
TASKS = (
    "COMPOSITION_PLAN",
    "MATCHUP_PLAN",
    "ITEMIZATION_DECISION",
    "MACRO_PRIORITY",
    "THREAT_ASSESSMENT",
    "ADVICE_SUPPRESSION",
)
VOLATILE_FIELDS = {
    "championName",
    "itemName",
    "runeName",
    "patchVersion",
    "winRate",
    "pickRate",
    "tier",
    "cooldownSeconds",
    "objectiveRespawnSeconds",
}
CANONICAL_METADATA = {
    "scenarioId",
    "familyId",
    "splitGroup",
    "sourceType",
    "seed",
    "causalSignature",
    "parentScenarioId",
    "counterfactualPairId",
    "episodeId",
    "episodeStep",
}


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in _walk_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _walk_keys(child)}
    return set()


@pytest.mark.parametrize("locale", ("it-IT", "en-US"))
def test_show_message_uses_requested_locale(locale: str) -> None:
    sampled = SAMPLER.sample_with_result(
        "composition_protect_vs_dive", 301, output_locale=locale
    )
    message = MessageRenderer().render(
        sampled.scenario,
        sampled.oracle_result.decision,
        intent=sampled.message_intent,
    )

    assert message
    assert len(message) <= 180
    if locale == "it-IT":
        assert any(word in message for word in ("Scegli", "Usa", "Preferisci", "Procedi", "Esegui", "priorità"))
    else:
        assert any(word in message for word in ("Prioritize", "Choose", "Use", "Favor", "Commit", "Take"))


@pytest.mark.parametrize(
    "family_id",
    ("suppression_exact_duplicate", "macro_objective_state_stale"),
)
def test_non_show_message_is_empty(family_id: str) -> None:
    sampled = SAMPLER.sample_with_result(family_id, 302)

    assert sampled.oracle_result.decision.decision != "SHOW"
    assert (
        MessageRenderer().render(
            sampled.scenario, sampled.oracle_result.decision
        )
        == ""
    )


@pytest.mark.parametrize("task", TASKS)
def test_release_rows_and_nested_json_match_contracts(task: str) -> None:
    family = SAMPLER._repository.families(task=task)[0]
    sampled = SAMPLER.sample_with_result(family, 400 + TASKS.index(task))
    row = RENDERER.render(
        sampled.scenario,
        sampled.oracle_result,
        message_intent=sampled.message_intent,
    )
    errors: list[str] = []

    validate_row(
        row,
        Path("generated.jsonl"),
        1,
        VALIDATORS,
        errors,
    )

    assert errors == []
    assert json.loads(row["messages"][1]["content"])
    assert json.loads(row["messages"][2]["content"])


def test_jsonl_renderer_emits_exactly_one_line() -> None:
    sampled = SAMPLER.sample_with_result(
        "composition_protect_vs_dive", 501
    )
    line = RENDERER.render_jsonl_line(
        sampled.scenario,
        sampled.oracle_result,
        message_intent=sampled.message_intent,
    )

    assert "\n" not in line
    assert json.loads(line)["messages"][0]["role"] == "system"


def test_release_contains_no_volatile_or_canonical_metadata_fields() -> None:
    sampled = SAMPLER.sample_with_result(
        "macro_objective_setup", 502, id_style="misleading"
    )
    row = RENDERER.render(
        sampled.scenario,
        sampled.oracle_result,
        message_intent=sampled.message_intent,
    )
    user = json.loads(row["messages"][1]["content"])
    assistant = json.loads(row["messages"][2]["content"])
    keys = _walk_keys({"user": user, "assistant": assistant})

    assert keys.isdisjoint(VOLATILE_FIELDS)
    assert keys.isdisjoint(CANONICAL_METADATA)


def test_rendering_is_deterministic() -> None:
    first = SAMPLER.sample_with_result("matchup_break_enemy_freeze", 503)
    second = SAMPLER.sample_with_result("matchup_break_enemy_freeze", 503)

    assert RENDERER.render_jsonl_line(
        first.scenario,
        first.oracle_result,
        message_intent=first.message_intent,
    ) == RENDERER.render_jsonl_line(
        second.scenario,
        second.oracle_result,
        message_intent=second.message_intent,
    )


def test_temporal_family_uses_controlled_fallback_intent() -> None:
    episode = EpisodeGenerator().generate("episode_wave_safe_push", 504)
    step = episode.steps[0]

    row = RENDERER.render(step.scenario, step.oracle_result)

    assert json.loads(row["messages"][1]["content"])
    assert json.loads(row["messages"][2]["content"])


def test_empty_evidence_renders_zero_information_quality() -> None:
    sampled = SAMPLER.sample_with_result("macro_objective_setup", 505)
    scenario = replace(sampled.scenario, evidence=())

    row = RENDERER.render(scenario)
    user = json.loads(row["messages"][1]["content"])

    assert user["evidence"] == []
    assert user["payload"]["visionConfidence"] == 0.0
