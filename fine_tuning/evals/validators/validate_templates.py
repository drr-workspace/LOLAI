from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator


from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    Evidence,
    RecentAdvice,
    ScenarioContext,
    TeamPlan,
    Threat,
)
from generators.oracle import StrategicOracle
from generators.template_loader import (
    TemplateError,
    TemplateFamily,
    TemplateRepository,
    default_template_dir,
    load_templates,
)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TemplateError(f"{location}: atteso oggetto")
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise TemplateError(f"{location}: attesa lista")
    return value


def _profile_name(reference: Mapping[str, Any]) -> str:
    value = reference.get("profile")
    if not isinstance(value, str):
        raise TemplateError("riferimento profilo non valido")
    return value


def _selected_profile(
    family: TemplateFamily,
    base_key: str,
    edge_key: str,
    *,
    edge_case: bool,
) -> str:
    source = family.edge_case if edge_case else family.parameter_ranges
    key = edge_key if edge_case else base_key
    value = source.get(key)
    if not isinstance(value, str):
        raise TemplateError(f"{family.family_id}.{key}: profilo non valido")
    return value


def _build_context(
    repository: TemplateRepository,
    family: TemplateFamily,
    *,
    edge_case: bool,
) -> ScenarioContext:
    profile_name = _selected_profile(
        family, "contextProfile", "contextProfile", edge_case=edge_case
    )
    raw = _mapping(
        repository.resolve_profile(family, "contexts", profile_name),
        f"{family.family_id}.context",
    )
    return ScenarioContext(
        observed_at_game_second=int(raw["observedAtGameSecond"]),
        freshness_seconds=int(raw["freshnessSeconds"]),
        completeness=float(raw["completeness"]),
        uncertain_fields=tuple(cast(Sequence[str], raw["uncertainFields"])),
        required_fields=frozenset(family.required_context_fields),
        available_fields=frozenset(
            cast(Sequence[str], raw["availableFields"])
        ),
        state_signature=str(raw["stateSignature"]),
    )


def _build_evidence(
    repository: TemplateRepository,
    family: TemplateFamily,
    *,
    edge_case: bool,
) -> tuple[Evidence, ...]:
    profile_name = (
        _selected_profile(
            family, "unused", "evidenceProfile", edge_case=True
        )
        if edge_case
        else _profile_name(family.evidence_blueprints)
    )
    raw_items = _sequence(
        repository.resolve_profile(family, "evidence", profile_name),
        f"{family.family_id}.evidence",
    )
    return tuple(
        Evidence(
            evidence_id=f"ev_{item['role']}",
            category=str(item["category"]),
            confidence=float(item["confidence"]),
            freshness_seconds=int(item["freshnessSeconds"]),
            conflicts_with_evidence_ids=frozenset(
                f"ev_{role}" for role in item["conflictsWithRoles"]
            ),
        )
        for raw_item in raw_items
        for item in [_mapping(raw_item, f"{family.family_id}.evidence")]
    )


def _build_threats(
    repository: TemplateRepository,
    family: TemplateFamily,
    *,
    edge_case: bool,
) -> tuple[Threat, ...]:
    profile_name = (
        _selected_profile(family, "unused", "threatProfile", edge_case=True)
        if edge_case
        else _profile_name(family.threat_blueprints)
    )
    raw_items = _sequence(
        repository.resolve_profile(family, "threats", profile_name),
        f"{family.family_id}.threats",
    )
    return tuple(
        Threat(
            entity_id=f"threat_{item['role']}",
            priority=float(item["priority"]),
            evidence_ids=("ev_primary",),
        )
        for raw_item in raw_items
        for item in [_mapping(raw_item, f"{family.family_id}.threat")]
    )


def _build_candidates(
    repository: TemplateRepository,
    family: TemplateFamily,
    *,
    edge_case: bool,
) -> tuple[CandidateAction, ...]:
    profile_name = (
        _selected_profile(
            family, "unused", "candidateProfile", edge_case=True
        )
        if edge_case
        else _profile_name(family.candidate_blueprints)
    )
    raw_items = _sequence(
        repository.resolve_profile(family, "candidates", profile_name),
        f"{family.family_id}.candidates",
    )
    return tuple(
        CandidateAction(
            action_id=f"action_{item['role']}",
            action_type=str(item["actionType"]),
            evidence_ids=tuple(
                f"ev_{role}" for role in item["evidenceRoles"]
            ),
            feasibility=float(item["feasibility"]),
            supports_functions=frozenset(item["supportsFunctions"]),
            countered_threat_ids=frozenset(
                f"threat_{role}" for role in item["counteredThreatRoles"]
            ),
            win_condition_tags=frozenset(item["winConditionTags"]),
            urgency_alignment=float(item["urgencyAlignment"]),
            opportunity_cost=float(item["opportunityCost"]),
            execution_burden=float(item["executionBurden"]),
            equivalence_key=str(item["equivalenceKey"]),
        )
        for raw_item in raw_items
        for item in [_mapping(raw_item, f"{family.family_id}.candidate")]
    )


def _build_team_plan(
    repository: TemplateRepository,
    family: TemplateFamily,
    *,
    edge_case: bool,
) -> TeamPlan:
    profile_name = _selected_profile(
        family, "teamPlanProfile", "teamPlanProfile", edge_case=edge_case
    )
    raw = _mapping(
        repository.resolve_profile(family, "teamPlans", profile_name),
        f"{family.family_id}.teamPlan",
    )
    return TeamPlan(
        primary_win_condition=str(raw["primaryWinCondition"]),
        win_condition_tags=frozenset(raw["winConditionTags"]),
        missing_functions=frozenset(raw["missingFunctions"]),
        covered_functions=frozenset(raw["coveredFunctions"]),
    )


def _build_recent_advice(
    repository: TemplateRepository,
    family: TemplateFamily,
    *,
    edge_case: bool,
) -> tuple[RecentAdvice, ...]:
    profile_name = (
        _selected_profile(
            family, "unused", "recentAdviceProfile", edge_case=True
        )
        if edge_case
        else _profile_name(family.recent_advice_blueprint)
    )
    raw_items = _sequence(
        repository.resolve_profile(family, "recentAdvice", profile_name),
        f"{family.family_id}.recentAdvice",
    )
    return tuple(
        RecentAdvice(
            action_id=f"action_{item['actionRole']}",
            equivalence_key=str(item["equivalenceKey"]),
            age_seconds=int(item["ageSeconds"]),
            decision=str(item["decision"]),
            state_signature=str(item["stateSignature"]),
        )
        for raw_item in raw_items
        for item in [_mapping(raw_item, f"{family.family_id}.recentAdvice")]
    )


def build_scenario(
    repository: TemplateRepository,
    family: TemplateFamily,
    *,
    edge_case: bool = False,
) -> CanonicalScenario:
    """Materialize one deterministic minimum scenario for a template family."""
    seed_range = _mapping(
        family.parameter_ranges["seed"], f"{family.family_id}.seed"
    )
    return CanonicalScenario(
        scenario_id=(
            f"validation_{family.family_id}_edge"
            if edge_case
            else f"validation_{family.family_id}"
        ),
        family_id=family.family_id,
        split_group=family.split_group,
        source_type=family.source_eligibility[0],
        seed=int(seed_range["minimum"]),
        task=family.task,
        context=_build_context(repository, family, edge_case=edge_case),
        evidence=_build_evidence(repository, family, edge_case=edge_case),
        candidates=_build_candidates(repository, family, edge_case=edge_case),
        threats=_build_threats(repository, family, edge_case=edge_case),
        team_plan=_build_team_plan(repository, family, edge_case=edge_case),
        recent_advice=_build_recent_advice(
            repository, family, edge_case=edge_case
        ),
    )


def validate_constraint(
    family: TemplateFamily,
    scenario: CanonicalScenario,
    oracle: StrategicOracle,
    *,
    edge_case: bool = False,
) -> tuple[str, str]:
    """Run one family scenario and enforce its expected decision constraints."""
    result = oracle.decide(scenario)
    constraints = (
        cast(
            Mapping[str, Any],
            family.edge_case["expectedDecisionConstraints"],
        )
        if edge_case
        else family.expected_decision_constraints
    )
    allowed_reason_codes = (
        cast(Sequence[str], family.edge_case["allowedReasonCodes"])
        if edge_case
        else family.allowed_reason_codes
    )
    decisions = set(cast(Sequence[str], constraints["decisions"]))
    gates = set(cast(Sequence[str], constraints["triggeredGates"]))
    selected_roles = set(
        cast(Sequence[str], constraints["selectedCandidateRoles"])
    )
    selected_ids = {f"action_{role}" for role in selected_roles}

    errors: list[str] = []
    if result.decision.decision not in decisions:
        errors.append(
            f"decision={result.decision.decision}, attese={sorted(decisions)}"
        )
    if result.trace.triggered_gate not in gates:
        errors.append(
            f"gate={result.trace.triggered_gate}, attesi={sorted(gates)}"
        )
    selected = result.decision.primary_action_id
    if selected_roles and selected not in selected_ids:
        errors.append(
            f"selected={selected}, attesi={sorted(selected_ids)}"
        )
    if not selected_roles and selected is not None:
        errors.append(f"selected deve essere null, ottenuto {selected}")
    unexpected_reasons = set(result.decision.reason_codes) - set(
        allowed_reason_codes
    )
    if unexpected_reasons:
        errors.append(
            f"reason code non consentiti={sorted(unexpected_reasons)}"
        )
    if errors:
        raise TemplateError(f"{family.family_id}: {'; '.join(errors)}")
    return result.decision.decision, result.trace.triggered_gate


def validate_schema_file(template_dir: Path) -> None:
    """Validate the template schema as a Draft 2020-12 schema."""
    import json

    schema_path = template_dir / "template.schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator.check_schema(schema)


def main() -> int:
    """Validate schema, families and one oracle scenario per family."""
    template_dir = default_template_dir()
    try:
        validate_schema_file(template_dir)
        repository = load_templates(template_dir)
        itemization_count = len(
            repository.families(task="ITEMIZATION_DECISION")
        )
        threat_count = len(
            repository.families(task="THREAT_ASSESSMENT")
        )
        composition_count = len(
            repository.families(task="COMPOSITION_PLAN")
        )
        matchup_count = len(repository.families(task="MATCHUP_PLAN"))
        macro_count = len(repository.families(task="MACRO_PRIORITY"))
        suppression_count = len(
            repository.families(task="ADVICE_SUPPRESSION")
        )
        if itemization_count < 24:
            raise TemplateError(
                f"famiglie ITEMIZATION_DECISION insufficienti: {itemization_count}"
            )
        if threat_count < 18:
            raise TemplateError(
                f"famiglie THREAT_ASSESSMENT insufficienti: {threat_count}"
            )
        if composition_count < 20:
            raise TemplateError(
                f"famiglie COMPOSITION_PLAN insufficienti: {composition_count}"
            )
        if matchup_count < 24:
            raise TemplateError(
                f"famiglie MATCHUP_PLAN insufficienti: {matchup_count}"
            )
        if macro_count < 26:
            raise TemplateError(
                f"famiglie MACRO_PRIORITY insufficienti: {macro_count}"
            )
        if suppression_count < 14:
            raise TemplateError(
                "famiglie ADVICE_SUPPRESSION insufficienti: "
                f"{suppression_count}"
            )
        if len(repository.families()) <= 120:
            raise TemplateError(
                "il totale delle famiglie deve superare 120"
            )
        semantic_signatures: set[tuple[str, tuple[str, ...]]] = set()
        for family in repository.families():
            signature = (
                family.invariant_principle.casefold().strip(),
                tuple(sorted(family.semantic_comparison_fields)),
            )
            if signature in semantic_signatures:
                raise TemplateError(
                    f"{family.family_id}: firma semantica duplicata"
                )
            semantic_signatures.add(signature)

        oracle = StrategicOracle()
        reports: list[tuple[str, str, str, str]] = []
        for family in repository.families():
            scenario = build_scenario(repository, family)
            decision, gate = validate_constraint(family, scenario, oracle)
            reports.append((family.family_id, "base", decision, gate))
            if family.edge_case:
                edge_scenario = build_scenario(
                    repository, family, edge_case=True
                )
                edge_decision, edge_gate = validate_constraint(
                    family, edge_scenario, oracle, edge_case=True
                )
                reports.append(
                    (family.family_id, "edge", edge_decision, edge_gate)
                )
    except (OSError, ValueError, TemplateError) as error:
        print(f"Validazione template fallita: {error}", file=sys.stderr)
        return 1

    for family_id, variant, decision, gate in reports:
        print(
            f"PASS {family_id} [{variant}]: "
            f"decision={decision}, gate={gate}"
        )
    print(
        f"Template validi: {len(repository.families())} famiglie "
        f"({itemization_count} itemization, {threat_count} threat, "
        f"{composition_count} composition, {matchup_count} matchup, "
        f"{macro_count} macro, {suppression_count} suppression), "
        f"{len(reports)} scenari oracle verificati."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
