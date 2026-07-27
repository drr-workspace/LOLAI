from __future__ import annotations

from collections.abc import Mapping

from generators.domain_models import CanonicalScenario
from generators.episodes import Episode, EpisodeTemplate
from generators.oracle import StrategicOracle


SEMANTIC_FIELDS = (
    "context",
    "evidence",
    "candidates",
    "threats",
    "team_plan",
    "recent_advice",
)


class EpisodeValidationError(ValueError):
    pass


def _changed_fields(
    previous: CanonicalScenario, current: CanonicalScenario
) -> frozenset[str]:
    return frozenset(
        field
        for field in SEMANTIC_FIELDS
        if getattr(previous, field) != getattr(current, field)
    )


def _declared_fields(delta: Mapping[str, object]) -> frozenset[str]:
    value = delta.get("changedFields")
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(str(field) for field in value)


def episode_errors(
    episode: Episode,
    template: EpisodeTemplate | None = None,
    oracle: StrategicOracle | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    engine = oracle or StrategicOracle()
    if not 3 <= len(episode.steps) <= 8:
        errors.append("l'episodio deve contenere da 3 a 8 step")
    if not episode.episode_id:
        errors.append("episodeId mancante")

    prior_show_actions: dict[str, int] = {}
    previous_elapsed = -1
    previous_observed = -1
    for index, step in enumerate(episode.steps):
        scenario = step.scenario
        location = f"step[{index}]"
        if step.index != index or scenario.episode_step != index:
            errors.append(f"{location}: indice non coerente")
        if scenario.episode_id != episode.episode_id:
            errors.append(f"{location}: episodeId non coerente")
        if scenario.split_group != episode.split_group:
            errors.append(f"{location}: splitGroup non coerente")
        if scenario.source_type != "TEMPORAL":
            errors.append(f"{location}: sourceType deve essere TEMPORAL")
        if step.elapsed_seconds <= previous_elapsed:
            errors.append(f"{location}: tempo trascorso non monotono")
        observed = scenario.context.observed_at_game_second
        if observed <= previous_observed:
            errors.append(f"{location}: tempo osservato non monotono")
        previous_elapsed = step.elapsed_seconds
        previous_observed = observed
        if not isinstance(step.delta, Mapping) or not step.delta.get("event"):
            errors.append(f"{location}: delta esplicito mancante")

        recomputed = engine.decide(scenario)
        if recomputed.decision != step.oracle_result.decision:
            errors.append(f"{location}: risultato oracle non aggiornato")
        actual_decision = recomputed.decision.decision
        expected_decision = (
            template.expected_decisions[index]
            if template is not None
            else step.expected_transition.rsplit("_TO_", 1)[-1]
        )
        if actual_decision != expected_decision:
            errors.append(
                f"{location}: decisione {actual_decision}, "
                f"attesa {expected_decision}"
            )

        if index == 0:
            if step.previous_state is not None:
                errors.append("step[0]: previousState deve essere null")
            if step.previous_advice is not None:
                errors.append("step[0]: previousAdvice deve essere null")
            if _declared_fields(step.delta):
                errors.append("step[0]: INITIAL_STATE non deve dichiarare cambi")
        else:
            previous = episode.steps[index - 1]
            if step.previous_state != previous.scenario:
                errors.append(f"{location}: previousState non coerente")
            if step.previous_advice != previous.oracle_result.decision:
                errors.append(f"{location}: previousAdvice non coerente")
            expected_transition = (
                f"{previous.oracle_result.decision.decision}_TO_"
                f"{actual_decision}"
            )
            if step.expected_transition != expected_transition:
                errors.append(f"{location}: expectedTransition non coerente")
            if scenario.parent_scenario_id != previous.scenario.scenario_id:
                errors.append(f"{location}: parentScenarioId non coerente")
            actual_changed = _changed_fields(previous.scenario, scenario)
            declared_changed = _declared_fields(step.delta)
            if actual_changed != declared_changed:
                errors.append(
                    f"{location}: delta non coerente; "
                    f"effettivi={sorted(actual_changed)}, "
                    f"dichiarati={sorted(declared_changed)}"
                )

            previous_action = (
                previous.oracle_result.decision.primary_action_id
            )
            if previous_action is not None:
                prior_show_actions[previous_action] = previous.elapsed_seconds
                matching = [
                    advice
                    for advice in scenario.recent_advice
                    if advice.action_id == previous_action
                    and advice.decision == "SHOW"
                ]
                if not matching:
                    errors.append(
                        f"{location}: recentAdvice non contiene "
                        "il consiglio SHOW precedente"
                    )

        known_prior_ids = set(prior_show_actions)
        for advice in scenario.recent_advice:
            if advice.action_id not in known_prior_ids and index > 0:
                errors.append(
                    f"{location}: recentAdvice riferisce un consiglio ignoto"
                )
                continue
            shown_at = prior_show_actions.get(advice.action_id)
            if shown_at is not None:
                expected_age = step.elapsed_seconds - shown_at
                if advice.age_seconds != expected_age:
                    errors.append(
                        f"{location}: età recentAdvice "
                        f"{advice.age_seconds}, attesa {expected_age}"
                    )

        if index > 0:
            previous_scenario = episode.steps[index - 1].scenario
            if {
                item.action_id for item in scenario.candidates
            } != {
                item.action_id for item in previous_scenario.candidates
            }:
                errors.append(f"{location}: actionId non continui")
            if {
                item.evidence_id for item in scenario.evidence
            } != {
                item.evidence_id for item in previous_scenario.evidence
            }:
                errors.append(f"{location}: evidenceId non continui")
            if {
                item.entity_id for item in scenario.threats
            } != {
                item.entity_id for item in previous_scenario.threats
            }:
                errors.append(f"{location}: entityId non continui")

    if template is not None:
        if episode.template_id != template.template_id:
            errors.append("templateId non coerente")
        if episode.split_group != template.split_group:
            errors.append("splitGroup diverso dal template")
        if tuple(
            step.elapsed_seconds for step in episode.steps
        ) != template.elapsed_seconds:
            errors.append("tempi diversi dal template")
    return tuple(errors)


def validate_episode(
    episode: Episode,
    template: EpisodeTemplate | None = None,
    oracle: StrategicOracle | None = None,
) -> None:
    errors = episode_errors(episode, template, oracle)
    if errors:
        raise EpisodeValidationError("; ".join(errors))
