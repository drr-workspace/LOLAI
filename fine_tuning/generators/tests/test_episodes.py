from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from generators.causal_signature import (
    CausalSignatureBuilder,
    DatasetRecord,
)
from generators.episode_validator import (
    EpisodeValidationError,
    episode_errors,
    validate_episode,
)
from generators.episodes import EpisodeGenerator


GENERATOR = EpisodeGenerator()
TEMPLATE_IDS = tuple(GENERATOR.templates)


def test_at_least_twenty_episode_templates_exist() -> None:
    assert len(TEMPLATE_IDS) >= 20


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_every_declared_episode_is_valid(template_id: str) -> None:
    template = GENERATOR.templates[template_id]
    episode = GENERATOR.generate(template_id, 1001)

    validate_episode(episode, template)
    assert 3 <= len(episode.steps) <= 8


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_episode_generation_is_deterministic(template_id: str) -> None:
    assert GENERATOR.generate(
        template_id, 1002
    ) == GENERATOR.generate(template_id, 1002)


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_episode_steps_remain_in_one_split(template_id: str) -> None:
    episode = GENERATOR.generate(template_id, 1003)

    assert len(
        {step.scenario.split_group for step in episode.steps}
    ) == 1
    assert all(
        step.scenario.episode_id == episode.episode_id
        for step in episode.steps
    )


def test_vision_setup_suppress_then_contest() -> None:
    episode = GENERATOR.generate("episode_vision_contest_basic", 1101)

    assert [
        step.oracle_result.decision.decision for step in episode.steps
    ] == ["SHOW", "SUPPRESS", "SHOW"]
    selected = episode.steps[-1].oracle_result.decision.primary_action_id
    action = next(
        item
        for item in episode.steps[-1].scenario.candidates
        if item.action_id == selected
    )
    assert action.action_type == "CONTEST_OBJECTIVE"


def test_itemization_changes_action_after_primary_threat_change() -> None:
    episode = GENERATOR.generate(
        "episode_itemization_threat_physical", 1102
    )

    assert episode.steps[0].oracle_result.decision.primary_action_id != (
        episode.steps[2].oracle_result.decision.primary_action_id
    )
    assert episode.steps[1].oracle_result.decision.decision == "SUPPRESS"


def test_wave_change_enables_recall_or_push() -> None:
    episode = GENERATOR.generate("episode_wave_safe_recall", 1103)
    selected = episode.steps[2].oracle_result.decision.primary_action_id
    action = next(
        item
        for item in episode.steps[2].scenario.candidates
        if item.action_id == selected
    )

    assert action.action_type in {"RECALL", "PUSH_FAST"}
    assert episode.steps[1].delta["event"] == "WAVE_UNCHANGED"
    assert episode.steps[2].delta["event"] == "WAVE_STATE_CHANGED"


def test_macro_conflict_requests_refresh_then_recovers() -> None:
    episode = GENERATOR.generate(
        "episode_macro_conflict_objective", 1104
    )

    assert [
        step.oracle_result.decision.decision for step in episode.steps
    ] == ["SHOW", "REQUEST_REFRESH", "SHOW"]


def test_incomplete_context_remains_refresh_until_data_arrives() -> None:
    episode = GENERATOR.generate(
        "episode_refresh_missing_context", 1105
    )

    assert [
        step.oracle_result.decision.decision for step in episode.steps
    ] == ["REQUEST_REFRESH", "REQUEST_REFRESH", "SHOW"]
    assert episode.steps[2].delta["event"] == "RELIABLE_DATA_ADDED"


def test_validity_expiry_allows_new_evaluation() -> None:
    episode = GENERATOR.generate(
        "episode_validity_expiry_short", 1106
    )
    first = episode.steps[0].oracle_result.decision

    assert episode.steps[1].oracle_result.decision.decision == "SUPPRESS"
    assert episode.steps[2].elapsed_seconds > first.valid_for_seconds
    assert episode.steps[2].oracle_result.decision.decision == "SHOW"


def test_validator_rejects_non_monotonic_time() -> None:
    episode = GENERATOR.generate("episode_vision_contest_basic", 1201)
    bad_step = replace(episode.steps[1], elapsed_seconds=0)
    invalid = replace(
        episode, steps=(episode.steps[0], bad_step, episode.steps[2])
    )

    with pytest.raises(EpisodeValidationError, match="monotono"):
        validate_episode(invalid, GENERATOR.templates[episode.template_id])


def test_validator_rejects_incoherent_recent_advice() -> None:
    episode = GENERATOR.generate("episode_vision_contest_basic", 1202)
    scenario = replace(episode.steps[1].scenario, recent_advice=())
    bad_step = replace(episode.steps[1], scenario=scenario)
    invalid = replace(
        episode, steps=(episode.steps[0], bad_step, episode.steps[2])
    )

    errors = episode_errors(
        invalid, GENERATOR.templates[episode.template_id]
    )

    assert any("recentAdvice" in error for error in errors)


def test_validator_rejects_undeclared_state_change() -> None:
    episode = GENERATOR.generate("episode_wave_safe_recall", 1203)
    delta = MappingProxyType(
        {
            "event": episode.steps[2].delta["event"],
            "changedFields": ["context", "recent_advice"],
            "details": episode.steps[2].delta["details"],
        }
    )
    bad_step = replace(episode.steps[2], delta=delta)
    invalid = replace(
        episode, steps=(episode.steps[0], episode.steps[1], bad_step)
    )

    with pytest.raises(EpisodeValidationError, match="delta non coerente"):
        validate_episode(invalid, GENERATOR.templates[episode.template_id])


def test_validator_rejects_broken_identifier_continuity() -> None:
    episode = GENERATOR.generate(
        "episode_itemization_threat_physical", 1204
    )
    scenario = episode.steps[2].scenario
    renamed = replace(
        scenario.candidates[0], action_id="unexpected_new_action"
    )
    bad_scenario = replace(
        scenario, candidates=(renamed, *scenario.candidates[1:])
    )
    bad_step = replace(episode.steps[2], scenario=bad_scenario)
    invalid = replace(
        episode, steps=(episode.steps[0], episode.steps[1], bad_step)
    )

    errors = episode_errors(
        invalid, GENERATOR.templates[episode.template_id]
    )

    assert any("actionId non continui" in error for error in errors)


def test_every_non_initial_step_declares_a_delta() -> None:
    for template_id in TEMPLATE_IDS:
        episode = GENERATOR.generate(template_id, 1301)
        for step in episode.steps[1:]:
            assert step.delta["event"]
            assert step.delta["changedFields"]


def test_episode_seeds_create_temporal_causal_capacity() -> None:
    builder = CausalSignatureBuilder()
    signatures: set[tuple[str, ...]] = set()
    for seed in range(2_000, 2_625):
        episode = GENERATOR.generate(
            "episode_vision_contest_basic", seed
        )
        signatures.add(
            tuple(
                builder.build(
                    DatasetRecord(step.scenario, step.oracle_result)
                ).digest
                for step in episode.steps
            )
        )

    assert len(signatures) == 625
