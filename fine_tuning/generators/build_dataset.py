from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evals.validators import validate_contracts
from evals.validators import (
    validate_ontology,
    validate_policies,
    validate_templates,
)
from evals.validators.validate_dataset import (
    validate_split as validate_release_split,
    verify_dataset_card,
    verify_no_cross_split_leakage,
)
from generators.adversarial.apply import (
    apply_adversarial,
    load_rules as load_adversarial_rules,
)
from generators.causal_signature import (
    CausalSignatureBuilder,
    DatasetRecord,
)
from generators.checksums import write_checksums
from generators.counterfactuals.apply import (
    apply_counterfactual,
    load_rules as load_counterfactual_rules,
)
from generators.dataset_card_builder import (
    build_dataset_card,
    write_dataset_card,
)
from generators.deduplicate import deduplicate
from generators.domain_models import (
    CandidateAction,
    CanonicalScenario,
    Evidence,
    OracleDecision,
    RecentAdvice,
    ScenarioContext,
    TeamPlan,
    Threat,
)
from generators.episodes import Episode, EpisodeGenerator
from generators.ontology_registry import OntologyRegistry
from generators.oracle import StrategicOracle
from generators.policy_loader import (
    PolicyBundle,
    load_policies,
)
from generators.release_manifest import (
    build_manifest,
    write_manifest,
)
from generators.release_renderer import (
    ReleaseRenderer,
    canonical_json,
)
from generators.sampler import ScenarioSampler
from generators.splitter import SPLIT_NAMES
from generators.template_loader import load_templates


ROOT = Path(__file__).resolve().parents[1]


class BuildError(RuntimeError):
    pass


def _log(message: str) -> None:
    print(f"[build] {message}", flush=True)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dataset_version: str
    schema_version: str
    ontology_version: str
    seed: int
    split_counts: Mapping[str, int]
    source_distribution: Mapping[str, float]
    decision_ranges: Mapping[str, tuple[float, float]]
    locales: tuple[str, ...]
    approved_realistic_sources: tuple[Path, ...]

    @property
    def total_examples(self) -> int:
        return sum(self.split_counts.values())


@dataclass(frozen=True, slots=True)
class GeneratedPool:
    records: tuple[DatasetRecord, ...]
    counterfactual_pairs: tuple[dict[str, object], ...]
    episodes: tuple[Episode, ...]
    realistic_available: int
    warnings: tuple[str, ...]


def load_config(path: Path) -> DatasetConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "datasetVersion",
        "schemaVersion",
        "ontologyVersion",
        "seed",
        "trainExamples",
        "validExamples",
        "testExamples",
        "challengeExamples",
        "trainSourceDistribution",
        "decisionDistributionRanges",
        "locales",
        "approvedRealisticSources",
    }
    if set(raw) != required:
        raise BuildError(
            f"config: chiavi mancanti/inattese: "
            f"{sorted(required ^ set(raw))}"
        )
    counts = {
        name: _positive_int(raw[f"{name}Examples"], f"{name}Examples")
        for name in SPLIT_NAMES
    }
    source = _distribution(
        raw["trainSourceDistribution"], "trainSourceDistribution"
    )
    expected_sources = {
        "SYNTHETIC",
        "COUNTERFACTUAL",
        "TEMPORAL",
        "REALISTIC_ABSTRACTED",
        "ADVERSARIAL",
    }
    if set(source) != expected_sources:
        raise BuildError("trainSourceDistribution incompleta")
    decision_ranges: dict[str, tuple[float, float]] = {}
    for decision in ("SHOW", "SUPPRESS", "REQUEST_REFRESH"):
        bounds = raw["decisionDistributionRanges"].get(decision)
        if not isinstance(bounds, dict) or set(bounds) != {
            "minimum",
            "maximum",
        }:
            raise BuildError(f"range decisione non valido: {decision}")
        minimum = _unit(bounds["minimum"], f"{decision}.minimum")
        maximum = _unit(bounds["maximum"], f"{decision}.maximum")
        if minimum > maximum:
            raise BuildError(f"range invertito: {decision}")
        decision_ranges[decision] = (minimum, maximum)
    locales = tuple(str(value) for value in raw["locales"])
    if not locales or not set(locales) <= {"it-IT", "en-US"}:
        raise BuildError("locales non valide")
    return DatasetConfig(
        dataset_version=str(raw["datasetVersion"]),
        schema_version=str(raw["schemaVersion"]),
        ontology_version=str(raw["ontologyVersion"]),
        seed=_positive_int(raw["seed"], "seed", allow_zero=True),
        split_counts=counts,
        source_distribution=source,
        decision_ranges=decision_ranges,
        locales=locales,
        approved_realistic_sources=tuple(
            (ROOT / str(value)).resolve()
            for value in raw["approvedRealisticSources"]
        ),
    )


def _positive_int(
    value: object, location: str, *, allow_zero: bool = False
) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BuildError(f"{location}: intero >= {minimum} richiesto")
    return value


def _unit(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= float(value) <= 1
    ):
        raise BuildError(f"{location}: numero tra 0 e 1 richiesto")
    return float(value)


def _distribution(value: object, location: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise BuildError(f"{location}: oggetto richiesto")
    parsed = {str(key): _unit(item, f"{location}.{key}") for key, item in value.items()}
    if abs(sum(parsed.values()) - 1.0) > 1e-9:
        raise BuildError(f"{location}: i valori devono sommare a 1")
    return parsed


def preflight(config: DatasetConfig) -> tuple[PolicyBundle, object]:
    validators = (
        ("contratti", validate_contracts.main),
        ("ontologia", validate_ontology.main),
        ("policy", validate_policies.main),
        ("template", validate_templates.main),
    )
    for name, validator in validators:
        if validator() != 0:
            raise BuildError(f"validazione preliminare fallita: {name}")
    schema_dir = ROOT / "contracts" / "task-schemas"
    schemas = validate_contracts.load_schemas(schema_dir)
    validate_contracts.build_registry(schemas)
    ontology = OntologyRegistry(ROOT / "ontology")
    policies = load_policies()
    templates = load_templates()
    if ontology.version != config.ontology_version:
        raise BuildError("ontologyVersion non coincide con l'ontologia")
    if policies.ontology_version != config.ontology_version:
        raise BuildError("ontologyVersion non coincide con le policy")
    if config.schema_version != "1.0.0":
        raise BuildError("schemaVersion non supportata")
    if not templates.families():
        raise BuildError("nessuna famiglia di template disponibile")
    return policies, templates


def effective_counts(
    config: DatasetConfig, limit_per_task: int | None
) -> dict[str, int]:
    if limit_per_task is None:
        return dict(config.split_counts)
    total = min(config.total_examples, limit_per_task * 6)
    return _scaled_counts(config, total)


def _scaled_counts(
    config: DatasetConfig, total: int
) -> dict[str, int]:
    weights = {
        name: count / config.total_examples
        for name, count in config.split_counts.items()
    }
    counts = {
        name: int(total * weights[name]) for name in SPLIT_NAMES
    }
    for name in ("challenge", "test", "valid", "train"):
        if total >= 4 and counts[name] == 0:
            counts[name] = 1
    while sum(counts.values()) > total:
        counts[max(counts, key=counts.get)] -= 1
    index = 0
    while sum(counts.values()) < total:
        counts[SPLIT_NAMES[index % len(SPLIT_NAMES)]] += 1
        index += 1
    return counts


def generate_pool(
    config: DatasetConfig,
    target_total: int,
    policies: PolicyBundle,
    templates: Any,
) -> GeneratedPool:
    oracle = StrategicOracle(policies=policies)
    sampler = ScenarioSampler(templates, policies, oracle)
    families = _round_robin_families(templates.families())
    desired = _source_counts(target_total, config.source_distribution)
    realistic, warnings = _load_approved_realistic(
        config.approved_realistic_sources, oracle
    )
    realistic = realistic[: desired["REALISTIC_ABSTRACTED"]]
    if len(realistic) < desired["REALISTIC_ABSTRACTED"]:
        shortage = desired["REALISTIC_ABSTRACTED"] - len(realistic)
        desired["SYNTHETIC"] += shortage
        warnings.append(
            f"quota realistica ridistribuita a SYNTHETIC: {shortage}"
        )
    if desired["COUNTERFACTUAL"] % 2:
        desired["COUNTERFACTUAL"] -= 1
        desired["SYNTHETIC"] += 1

    base_needed = (
        desired["SYNTHETIC"]
        + max(desired["COUNTERFACTUAL"], desired["ADVERSARIAL"])
    )
    base: list[DatasetRecord] = []
    for ordinal in range(base_needed):
        family = families[ordinal % len(families)]
        scenario_seed = config.seed + ordinal * 104729
        sampled = sampler.sample_with_result(
            family,
            scenario_seed,
            ordinal=ordinal,
            output_locale=config.locales[ordinal % len(config.locales)],
        )
        scenario = replace(sampled.scenario, source_type="SYNTHETIC")
        base.append(DatasetRecord(scenario, oracle.decide(scenario)))
        if (ordinal + 1) % 5_000 == 0:
            _log(f"scenari base: {ordinal + 1}/{base_needed}")

    records: list[DatasetRecord] = [
        *base[: desired["SYNTHETIC"]],
        *realistic,
    ]
    pair_rows: list[dict[str, object]] = []
    cf_rules = tuple(load_counterfactual_rules())
    cf_index = 0
    cf_attempts = 0
    while cf_index < desired["COUNTERFACTUAL"]:
        parent = base[cf_attempts % len(base)]
        rule_id = cf_rules[cf_attempts % len(cf_rules)]
        cf_attempts += 1
        try:
            pair = apply_counterfactual(
                parent.scenario, rule_id, oracle=oracle, verify=True
            )
        except ValueError:
            if cf_attempts > desired["COUNTERFACTUAL"] * len(cf_rules) * 4:
                raise BuildError(
                    "impossibile soddisfare la quota controfattuale"
                )
            continue
        pair_id = pair.counterfactual.counterfactual_pair_id
        parent_scenario = replace(
            pair.parent,
            scenario_id=f"{pair.parent.scenario_id}_cfbase_{cf_index}",
            source_type="COUNTERFACTUAL",
            parent_scenario_id=pair.parent.scenario_id,
            counterfactual_pair_id=pair_id,
        )
        child_scenario = replace(
            pair.counterfactual,
            parent_scenario_id=parent_scenario.scenario_id,
            counterfactual_pair_id=pair_id,
        )
        parent_record = DatasetRecord(
            parent_scenario,
            oracle.decide(parent_scenario),
            counterfactual_variable=rule_id,
        )
        child = DatasetRecord(
            child_scenario,
            oracle.decide(child_scenario),
            counterfactual_variable=rule_id,
        )
        records.extend((parent_record, child))
        pair_rows.append(
            {
                "counterfactualPairId": pair_id,
                "parentScenarioId": parent_scenario.scenario_id,
                "counterfactualScenarioId": child_scenario.scenario_id,
                "ruleId": rule_id,
                "expectedEffect": pair.expected_effect,
                "observedEffect": pair.observed_effect,
            }
        )
        cf_index += 2

    episode_generator = EpisodeGenerator(oracle=oracle)
    episode_ids = tuple(episode_generator.templates)
    episodes: list[Episode] = []
    temporal_count = 0
    episode_index = 0
    while temporal_count < desired["TEMPORAL"]:
        episode = episode_generator.generate(
            episode_ids[episode_index % len(episode_ids)],
            config.seed + 5_000_000 + episode_index,
        )
        remaining = desired["TEMPORAL"] - temporal_count
        if len(episode.steps) > remaining:
            break
        episodes.append(episode)
        for step in episode.steps:
            scenario = replace(step.scenario, source_type="TEMPORAL")
            records.append(
                DatasetRecord(scenario, step.oracle_result)
            )
        temporal_count += len(episode.steps)
        episode_index += 1
    desired["SYNTHETIC"] += desired["TEMPORAL"] - temporal_count

    adv_rules = tuple(load_adversarial_rules())
    adv_index = 0
    adv_attempts = 0
    while adv_index < desired["ADVERSARIAL"]:
        parent = base[adv_attempts % len(base)]
        rule_id = adv_rules[adv_attempts % len(adv_rules)]
        adv_attempts += 1
        try:
            application = apply_adversarial(
                parent.scenario,
                rule_id,
                mutation_seed=config.seed + adv_attempts,
                oracle=oracle,
                verify=True,
            )
        except ValueError:
            if adv_attempts > desired["ADVERSARIAL"] * len(adv_rules) * 4:
                raise BuildError(
                    "impossibile soddisfare la quota adversariale"
                )
            continue
        records.append(
            DatasetRecord(
                application.mutated, application.mutated_result
            )
        )
        adv_index += 1

    while len(records) < target_total:
        records.append(base[len(records) % len(base)])
    signed = _sign_records(records[:target_total], policies)
    return GeneratedPool(
        records=signed,
        counterfactual_pairs=tuple(pair_rows),
        episodes=tuple(episodes),
        realistic_available=len(realistic),
        warnings=tuple(warnings),
    )


def _source_counts(
    total: int, distribution: Mapping[str, float]
) -> dict[str, int]:
    counts = {
        source: int(total * share)
        for source, share in distribution.items()
    }
    counts["SYNTHETIC"] += total - sum(counts.values())
    return counts


def _release_source_counts(
    total: int,
    distribution: Mapping[str, float],
    *,
    realistic_available: int,
) -> dict[str, int]:
    counts = _source_counts(total, distribution)
    realistic_shortage = max(
        0,
        counts["REALISTIC_ABSTRACTED"] - realistic_available,
    )
    counts["REALISTIC_ABSTRACTED"] -= realistic_shortage
    counts["SYNTHETIC"] += realistic_shortage
    if counts["COUNTERFACTUAL"] % 2:
        counts["COUNTERFACTUAL"] -= 1
        counts["SYNTHETIC"] += 1
    temporal_remainder = counts["TEMPORAL"] % 3
    counts["TEMPORAL"] -= temporal_remainder
    counts["SYNTHETIC"] += temporal_remainder
    return counts


def _round_robin_families(families: Sequence[Any]) -> tuple[Any, ...]:
    by_task: dict[str, list[Any]] = {}
    for family in families:
        by_task.setdefault(family.task, []).append(family)
    ordered: list[Any] = []
    index = 0
    while any(index < len(items) for items in by_task.values()):
        for task in sorted(by_task):
            if index < len(by_task[task]):
                ordered.append(by_task[task][index])
        index += 1
    return tuple(ordered)


def _sign_records(
    records: Sequence[DatasetRecord], policies: PolicyBundle
) -> tuple[DatasetRecord, ...]:
    builder = CausalSignatureBuilder(policies.deduplication)
    result: list[DatasetRecord] = []
    for record in records:
        signature = builder.build(record).digest
        result.append(
            replace(
                record,
                scenario=replace(
                    record.scenario, causal_signature=signature
                ),
            )
        )
    return tuple(result)


def _load_approved_realistic(
    paths: Sequence[Path], oracle: StrategicOracle
) -> tuple[list[DatasetRecord], list[str]]:
    records: list[DatasetRecord] = []
    warnings: list[str] = []
    for path in paths:
        if not path.exists():
            warnings.append(f"sorgente realistica assente: {path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("reviewStatus") != "APPROVED":
                    continue
                try:
                    scenario = _parse_review_scenario(entry["scenario"])
                    result = oracle.decide(scenario)
                    decision = _parse_decision(entry["expectedOutput"])
                    records.append(
                        DatasetRecord(
                            scenario,
                            replace(result, decision=decision),
                            review_status="APPROVED",
                        )
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise BuildError(
                        f"{path}:{line_number}: record approvato non valido: "
                        f"{error}"
                    ) from error
    return records, warnings


def _parse_review_scenario(raw: Mapping[str, Any]) -> CanonicalScenario:
    context = raw["context"]
    evidence = tuple(
        Evidence(
            evidence_id=item["evidenceId"],
            category=item["type"],
            confidence=float(item["confidence"]),
            freshness_seconds=int(item["freshnessSeconds"]),
            conflicts_with_evidence_ids=frozenset(
                item["conflictsWithEvidenceIds"]
            ),
            fact=item["fact"],
        )
        for item in raw["evidence"]
    )
    threats = tuple(
        Threat(
            entity_id=item["entityId"],
            priority=float(item["priority"]),
            evidence_ids=tuple(item["evidenceIds"]),
            patterns=tuple(item["patterns"]),
            damage_profile=(
                float(item["damageProfile"]["physical"]),
                float(item["damageProfile"]["magic"]),
                float(item["damageProfile"]["true"]),
            ),
        )
        for item in raw["threats"]
    )
    candidates = tuple(
        CandidateAction(
            action_id=item["actionId"],
            action_type=item["type"],
            evidence_ids=tuple(item["evidenceIds"]),
            feasibility=float(item["feasibility"]),
            supports_functions=frozenset(item["supportsFunctions"]),
            countered_threat_ids=frozenset(item["counteredThreatIds"]),
            win_condition_tags=frozenset(item["winConditionTags"]),
            urgency_alignment=float(item["urgencyAlignment"]),
            opportunity_cost=float(item["opportunityCost"]),
            execution_burden=float(item["executionBurden"]),
            equivalence_key=item["equivalenceKey"],
            effects=tuple(item["effects"]),
            resource_required=item.get("resourceRequired"),
        )
        for item in raw["candidates"]
    )
    plan = raw["teamPlan"]
    return CanonicalScenario(
        scenario_id=raw["scenarioId"],
        family_id=raw["familyId"],
        split_group=raw["splitGroup"],
        source_type="REALISTIC_ABSTRACTED",
        seed=int(raw["seed"]),
        task=raw["task"],
        context=ScenarioContext(
            observed_at_game_second=int(context["observedAtGameSecond"]),
            freshness_seconds=int(context["freshnessSeconds"]),
            completeness=float(context["completeness"]),
            uncertain_fields=tuple(context["uncertainFields"]),
            required_fields=frozenset(context["requiredFields"]),
            available_fields=frozenset(context["availableFields"]),
            state_signature=context["stateSignature"],
        ),
        evidence=evidence,
        candidates=candidates,
        threats=threats,
        team_plan=TeamPlan(
            primary_win_condition=plan["primaryWinCondition"],
            win_condition_tags=frozenset(plan["winConditionTags"]),
            missing_functions=frozenset(plan["missingFunctions"]),
            covered_functions=frozenset(plan["coveredFunctions"]),
        ),
        recent_advice=tuple(
            RecentAdvice(
                action_id=item["actionId"],
                equivalence_key=item["equivalenceKey"],
                age_seconds=int(item["ageSeconds"]),
                decision=item["decision"],
                state_signature=item["stateSignature"],
                category=item["category"],
                reason_codes=tuple(item["reasonCodes"]),
            )
            for item in raw["recentAdvice"]
        ),
        parent_scenario_id=raw.get("parentScenarioId"),
        counterfactual_pair_id=raw.get("counterfactualPairId"),
        episode_id=raw.get("episodeId"),
        episode_step=raw.get("episodeStep"),
        output_locale=raw.get("outputLocale", "en-US"),
        causal_signature=raw["causalSignature"],
    )


def _parse_decision(raw: Mapping[str, Any]) -> OracleDecision:
    return OracleDecision(
        schema_version=raw["schemaVersion"],
        decision=raw["decision"],
        category=raw["category"],
        primary_action_id=raw["primaryActionId"],
        alternative_action_ids=tuple(raw["alternativeActionIds"]),
        priority=raw["priority"],
        confidence=float(raw["confidence"]),
        reason_codes=tuple(raw["reasonCodes"]),
        evidence_ids=tuple(raw["evidenceIds"]),
        valid_for_seconds=int(raw["validForSeconds"]),
        recheck_triggers=tuple(raw["recheckTriggers"]),
    )


def _canonical_document(
    record: DatasetRecord, renderer: ReleaseRenderer
) -> dict[str, object]:
    row = renderer.render(record.scenario, record.oracle_result)
    user = json.loads(row["messages"][1]["content"])
    assistant = json.loads(row["messages"][2]["content"])
    scenario = record.scenario
    return {
        "scenarioId": scenario.scenario_id,
        "familyId": scenario.family_id,
        "splitGroup": scenario.split_group,
        "sourceType": scenario.source_type,
        "seed": scenario.seed,
        "parentScenarioId": scenario.parent_scenario_id,
        "counterfactualPairId": scenario.counterfactual_pair_id,
        "episodeId": scenario.episode_id,
        "episodeStep": scenario.episode_step,
        "causalSignature": scenario.causal_signature,
        "reviewStatus": record.review_status,
        "input": user,
        "expectedOutput": assistant,
    }


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(canonical_json(value))
            handle.write("\n")


def _report(
    records_by_split: Mapping[str, Sequence[DatasetRecord]],
    *,
    removed: int,
    warnings: Sequence[str],
) -> dict[str, object]:
    all_records = tuple(
        item for records in records_by_split.values() for item in records
    )
    total = len(all_records)
    count = lambda values: dict(sorted(Counter(values).items()))
    reviewed = sum(item.review_status == "APPROVED" for item in all_records)
    counterfactual = sum(
        item.scenario.source_type == "COUNTERFACTUAL"
        for item in all_records
    )
    temporal = sum(
        item.scenario.source_type == "TEMPORAL" for item in all_records
    )
    return {
        "counts": {
            name: len(records) for name, records in records_by_split.items()
        },
        "tasks": count(item.scenario.task for item in all_records),
        "decisions": count(
            item.oracle_result.decision.decision for item in all_records
        ),
        "sourceTypes": count(
            item.scenario.source_type for item in all_records
        ),
        "families": len(
            {item.scenario.family_id for item in all_records}
        ),
        "uniqueCausalSignatures": len(
            {item.scenario.causal_signature for item in all_records}
        ),
        "duplicatesRemoved": removed,
        "reviewedPercentage": _percentage(reviewed, total),
        "counterfactualPercentage": _percentage(counterfactual, total),
        "temporalPercentage": _percentage(temporal, total),
        "warnings": list(warnings),
    }


def _percentage(value: int, total: int) -> float:
    return round(100 * value / total, 3) if total else 0.0


def _validate_decision_ranges(
    train: Sequence[DatasetRecord],
    ranges: Mapping[str, tuple[float, float]],
    *,
    smoke_mode: bool,
) -> None:
    if smoke_mode:
        return
    counts = Counter(
        item.oracle_result.decision.decision for item in train
    )
    for decision, (minimum, maximum) in ranges.items():
        ratio = counts[decision] / len(train)
        if not minimum <= ratio <= maximum:
            raise BuildError(
                f"distribuzione train {decision}={ratio:.3f}, "
                f"attesa tra {minimum:.3f} e {maximum:.3f}"
            )


def _validate_rendered_release(
    output: Path, records_by_split: Mapping[str, Sequence[DatasetRecord]]
) -> None:
    schemas = validate_contracts.load_schemas(
        ROOT / "contracts" / "task-schemas"
    )
    registry = validate_contracts.build_registry(schemas)
    contract_validators = validate_contracts.build_validators(
        schemas, registry
    )
    contract_errors: list[str] = []
    for name in SPLIT_NAMES:
        _, errors = validate_contracts.validate_split(
            output / f"{name}.jsonl", contract_validators
        )
        contract_errors.extend(errors)
    results = {
        name: validate_release_split(output / f"{name}.jsonl")
        for name in SPLIT_NAMES
    }
    errors = list(contract_errors)
    errors.extend(
        error
        for result in results.values()
        for error in result["errors"]
    )
    errors.extend(verify_no_cross_split_leakage(results))
    errors.extend(verify_dataset_card(output, results))
    if errors:
        raise BuildError(
            "validazione release fallita:\n- " + "\n- ".join(errors[:30])
        )
    if any(
        results[name]["stats"]["examples"] != len(records_by_split[name])
        for name in SPLIT_NAMES
    ):
        raise BuildError("conteggi validati non coerenti")


def build(
    config: DatasetConfig,
    output: Path,
    *,
    force: bool,
    dry_run: bool,
    limit_per_task: int | None,
) -> dict[str, object]:
    canonical_output = (
        ROOT / "datasets" / "canonical" / "releases" / config.dataset_version
    )
    if (
        not dry_run
        and (output.exists() or canonical_output.exists())
        and not force
    ):
        existing = output if output.exists() else canonical_output
        raise BuildError(
            f"release già esistente: {existing}; usare --force"
        )
    _log("validazione preliminare")
    policies, templates = preflight(config)
    counts = effective_counts(config, limit_per_task)
    target_total = sum(counts.values())
    pool_target = max(target_total * 4, target_total + 32)
    _log(
        f"generazione pool: {pool_target} record per un target di "
        f"{target_total}"
    )
    pool = generate_pool(
        config, pool_target, policies, templates
    )
    _log("deduplicazione strutturale")
    deduplication = deduplicate(
        pool.records,
        policies.deduplication,
        progress=lambda done, total, kept: _log(
            f"deduplicazione: {done}/{total} unità, {kept} record conservati"
        ),
    )
    source_targets = _release_source_counts(
        target_total,
        config.source_distribution,
        realistic_available=pool.realistic_available,
    )
    selected = _select_cohesive(
        deduplication.kept,
        target_total,
        limit_per_task=limit_per_task,
        decision_ranges=config.decision_ranges,
        source_targets=(
            source_targets if limit_per_task is None else None
        ),
    )
    build_warnings = list(pool.warnings)
    if len(selected) < target_total:
        unique_signatures = len(
            {
                record.scenario.causal_signature
                for record in deduplication.kept
            }
        )
        if limit_per_task is None:
            raise BuildError(
                "la deduplicazione ha lasciato meno record del target: "
                f"{len(selected)}/{target_total}; firme causali uniche: "
                f"{unique_signatures}; capacità massima osservata con "
                f"maximumExamplesPerSignature="
                f"{policies.deduplication.maximum_examples_per_signature}: "
                f"{unique_signatures * policies.deduplication.maximum_examples_per_signature}"
            )
        build_warnings.append(
            "preview ridotta dopo deduplicazione: "
            f"{len(selected)}/{target_total} record unici"
        )
        counts = _scaled_counts(config, len(selected))
    _log(f"selezione completata: {len(selected)} record")
    selected_decisions = Counter(
        item.oracle_result.decision.decision for item in selected
    )
    _log(
        "decisioni selezionate: "
        + ", ".join(
            f"{decision}={count / len(selected):.1%}"
            for decision, count in sorted(selected_decisions.items())
        )
    )
    records_by_split = _exact_split(
        selected,
        counts,
        seed=config.seed,
        decision_ranges=config.decision_ranges,
    )
    _validate_decision_ranges(
        records_by_split["train"],
        config.decision_ranges,
        smoke_mode=limit_per_task is not None,
    )
    report = _report(
        records_by_split,
        removed=deduplication.removed_count,
        warnings=build_warnings,
    )
    if dry_run:
        return report
    staging = output.with_name(f".{output.name}.building")
    canonical_staging = canonical_output.with_name(
        f".{canonical_output.name}.building"
    )
    for path in (staging, canonical_staging):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    renderer = ReleaseRenderer()
    split_paths: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        path = staging / f"{name}.jsonl"
        _write_jsonl(
            path,
            (
                renderer.render(record.scenario, record.oracle_result)
                for record in records_by_split[name]
            ),
        )
        split_paths[name] = path
    _write_jsonl(
        canonical_staging / "scenarios.jsonl",
        (
            _canonical_document(record, renderer)
            for record in selected
        ),
    )
    _write_jsonl(
        canonical_staging / "counterfactual-pairs.jsonl",
        (
            pair
            for pair in pool.counterfactual_pairs
            if {
                pair["parentScenarioId"],
                pair["counterfactualScenarioId"],
            }
            <= {item.scenario.scenario_id for item in selected}
        ),
    )
    _write_jsonl(
        canonical_staging / "episodes.jsonl",
        (
            {
                "episodeId": episode.episode_id,
                "templateId": episode.template_id,
                "splitGroup": episode.split_group,
                "steps": [
                    {
                        "scenarioId": step.scenario.scenario_id,
                        "elapsedSeconds": step.elapsed_seconds,
                        "delta": dict(step.delta),
                        "expectedTransition": step.expected_transition,
                    }
                    for step in episode.steps
                ],
            }
            for episode in pool.episodes
            if all(
                step.scenario.scenario_id
                in {item.scenario.scenario_id for item in selected}
                for step in episode.steps
            )
        ),
    )
    report_text = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    (staging / "build-report.json").write_text(
        report_text, encoding="utf-8"
    )
    (canonical_staging / "build-report.json").write_text(
        report_text, encoding="utf-8"
    )
    card = build_dataset_card(
        dataset_version=config.dataset_version,
        schema_version=config.schema_version,
        ontology_version=config.ontology_version,
        seed=config.seed,
        split_records=records_by_split,
        split_paths=split_paths,
    )
    write_dataset_card(staging / "dataset-card.json", card)
    manifest_files = {
        **split_paths,
        "datasetCard": staging / "dataset-card.json",
        "buildReport": staging / "build-report.json",
    }
    manifest = build_manifest(
        dataset_version=config.dataset_version,
        schema_version=config.schema_version,
        ontology_version=config.ontology_version,
        seed=config.seed,
        files=manifest_files,
        build_report=report,
    )
    write_manifest(staging / "manifest.json", manifest)
    write_checksums(
        staging / "checksums.sha256",
        (
            *split_paths.values(),
            staging / "dataset-card.json",
            staging / "manifest.json",
            staging / "build-report.json",
        ),
        relative_to=staging,
    )
    _validate_rendered_release(staging, records_by_split)
    if output.exists():
        shutil.rmtree(output)
    if canonical_output.exists():
        shutil.rmtree(canonical_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical_output.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(output)
    canonical_staging.replace(canonical_output)
    return report


def _exact_split(
    records: Sequence[DatasetRecord],
    counts: Mapping[str, int],
    *,
    seed: int,
    decision_ranges: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[DatasetRecord, ...]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        scenario = record.scenario
        keys = [
            ("family", scenario.family_id),
            ("splitGroup", scenario.split_group),
        ]
        if scenario.episode_id:
            keys.append(("episode", scenario.episode_id))
        if scenario.counterfactual_pair_id:
            keys.append(("pair", scenario.counterfactual_pair_id))
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    grouped: dict[int, list[DatasetRecord]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    units = [
        tuple(sorted(items, key=lambda item: item.scenario.scenario_id))
        for items in grouped.values()
    ]
    units.sort(
        key=lambda unit: (
            -len(unit),
            hashlib.sha256(
                f"{seed}:{unit[0].scenario.scenario_id}".encode("utf-8")
            ).hexdigest(),
        )
    )
    assigned: dict[str, list[DatasetRecord]] = {
        name: [] for name in SPLIT_NAMES
    }
    remaining_units = list(units)
    decision_weights = {
        decision: (bounds[0] + bounds[1]) / 2
        for decision, bounds in decision_ranges.items()
    }
    for name in ("challenge", "test", "valid"):
        indexes = _subset_indexes_for_total(
            remaining_units,
            counts[name],
            decision_weights=decision_weights,
        )
        if indexes is None:
            raise BuildError(
                "conteggi split esatti incompatibili con i gruppi coesivi; "
                f"split={name}, target={counts[name]}, "
                f"unità_disponibili={len(remaining_units)}"
            )
        selected_indexes = set(indexes)
        assigned[name].extend(
            record
            for index in indexes
            for record in remaining_units[index]
        )
        remaining_units = [
            unit
            for index, unit in enumerate(remaining_units)
            if index not in selected_indexes
        ]
    assigned["train"].extend(
        record for unit in remaining_units for record in unit
    )
    actual = {
        name: len(split_records)
        for name, split_records in assigned.items()
    }
    if actual != dict(counts):
        raise BuildError(
            f"conteggi split non raggiunti: attesi={dict(counts)}, "
            f"ottenuti={actual}"
        )
    return {
        name: tuple(assigned[name]) for name in SPLIT_NAMES
    }


def _subset_indexes_for_total(
    units: Sequence[tuple[DatasetRecord, ...]],
    target: int,
    *,
    decision_weights: Mapping[str, float],
) -> tuple[int, ...] | None:
    paths: dict[
        int, tuple[tuple[int, ...], Counter[str]]
    ] = {0: ((), Counter())}
    for index, unit in enumerate(units):
        size = len(unit)
        unit_decisions = Counter(
            item.oracle_result.decision.decision for item in unit
        )
        for subtotal, (indexes, decisions) in tuple(
            sorted(paths.items(), reverse=True)
        ):
            candidate = subtotal + size
            if candidate > target:
                continue
            candidate_decisions = decisions + unit_decisions
            existing = paths.get(candidate)
            if (
                existing is None
                or _decision_deviation(
                    candidate_decisions,
                    candidate,
                    decision_weights,
                )
                < _decision_deviation(
                    existing[1],
                    candidate,
                    decision_weights,
                )
            ):
                paths[candidate] = (
                    (*indexes, index),
                    candidate_decisions,
                )
    result = paths.get(target)
    return result[0] if result is not None else None


def _decision_deviation(
    counts: Mapping[str, int],
    total: int,
    weights: Mapping[str, float],
) -> float:
    weight_total = sum(weights.values())
    return sum(
        abs(counts.get(decision, 0) - total * weight / weight_total)
        for decision, weight in weights.items()
    )


def _select_cohesive(
    records: Sequence[DatasetRecord],
    target: int,
    *,
    limit_per_task: int | None,
    decision_ranges: Mapping[str, tuple[float, float]],
    source_targets: Mapping[str, int] | None,
) -> tuple[DatasetRecord, ...]:
    grouped: dict[tuple[str, str], list[DatasetRecord]] = {}
    singles: list[DatasetRecord] = []
    for record in records:
        scenario = record.scenario
        if scenario.episode_id:
            grouped.setdefault(
                ("episode", scenario.episode_id), []
            ).append(record)
        elif scenario.counterfactual_pair_id:
            grouped.setdefault(
                ("pair", scenario.counterfactual_pair_id), []
            ).append(record)
        else:
            singles.append(record)
    units = [
        tuple(sorted(items, key=lambda item: item.scenario.scenario_id))
        for _, items in sorted(grouped.items())
    ]
    units.extend((item,) for item in singles)
    units.sort(key=lambda unit: unit[0].scenario.scenario_id)
    if source_targets is not None:
        return _select_source_quotas(
            units,
            target,
            source_targets=source_targets,
            decision_ranges=decision_ranges,
        )
    queues: dict[str, dict[str, deque[tuple[DatasetRecord, ...]]]] = {}
    mixed: dict[str, deque[tuple[DatasetRecord, ...]]] = {}
    for unit in units:
        task = unit[0].scenario.task
        decisions = {
            item.oracle_result.decision.decision for item in unit
        }
        if len(decisions) == 1:
            decision = next(iter(decisions))
            queues.setdefault(task, {}).setdefault(
                decision, deque()
            ).append(unit)
        else:
            mixed.setdefault(task, deque()).append(unit)
    midpoints = {
        decision: (bounds[0] + bounds[1]) / 2
        for decision, bounds in decision_ranges.items()
    }
    total_weight = sum(midpoints.values())
    desired_decisions = {
        decision: round(target * weight / total_weight)
        for decision, weight in midpoints.items()
    }
    desired_decisions["SHOW"] += target - sum(desired_decisions.values())
    selected: list[DatasetRecord] = []
    task_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    tasks = sorted({unit[0].scenario.task for unit in units})
    while len(selected) < target:
        progress = False
        for task in tasks:
            task_queues = queues.get(task, {})
            ranked_decisions = sorted(
                desired_decisions,
                key=lambda decision: (
                    -(
                        desired_decisions[decision]
                        - decision_counts[decision]
                    ),
                    decision,
                ),
            )
            candidates = [
                task_queues[decision][0]
                for decision in ranked_decisions
                if task_queues.get(decision)
            ]
            if mixed.get(task):
                candidates.append(mixed[task][0])
            for unit in candidates:
                if len(selected) + len(unit) > target:
                    continue
                if (
                    limit_per_task is not None
                    and task_counts[task] + len(unit) > limit_per_task
                ):
                    continue
                selected.extend(unit)
                task_counts.update(
                    item.scenario.task for item in unit
                )
                decision_counts.update(
                    item.oracle_result.decision.decision for item in unit
                )
                unit_decisions = {
                    item.oracle_result.decision.decision for item in unit
                }
                if len(unit_decisions) == 1:
                    task_queues[next(iter(unit_decisions))].popleft()
                else:
                    mixed[task].popleft()
                progress = True
                break
            if len(selected) == target:
                break
        if not progress:
            break
    return tuple(selected)


def _select_source_quotas(
    units: Sequence[tuple[DatasetRecord, ...]],
    target: int,
    *,
    source_targets: Mapping[str, int],
    decision_ranges: Mapping[str, tuple[float, float]],
) -> tuple[DatasetRecord, ...]:
    queue_key = "__MIXED__"
    queues: dict[
        str,
        dict[str, dict[str, deque[tuple[DatasetRecord, ...]]]],
    ] = {}
    available: Counter[str] = Counter()
    for unit in units:
        sources = {item.scenario.source_type for item in unit}
        if len(sources) != 1:
            raise BuildError("unità atomica con sourceType misti")
        source = next(iter(sources))
        task = unit[0].scenario.task
        decisions = {
            item.oracle_result.decision.decision for item in unit
        }
        decision_key = (
            next(iter(decisions)) if len(decisions) == 1 else queue_key
        )
        queues.setdefault(source, {}).setdefault(task, {}).setdefault(
            decision_key, deque()
        ).append(unit)
        available[source] += len(unit)
    shortages = {
        source: required - available[source]
        for source, required in source_targets.items()
        if available[source] < required
    }
    if shortages:
        raise BuildError(
            "capacità sourceType insufficiente dopo deduplicazione: "
            + ", ".join(
                f"{source} mancano {missing}"
                for source, missing in sorted(shortages.items())
            )
        )

    midpoints = {
        decision: (bounds[0] + bounds[1]) / 2
        for decision, bounds in decision_ranges.items()
    }
    weight_total = sum(midpoints.values())
    desired_decisions = {
        decision: round(target * weight / weight_total)
        for decision, weight in midpoints.items()
    }
    desired_decisions["SHOW"] += target - sum(desired_decisions.values())
    decision_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    selected: list[DatasetRecord] = []
    source_order = (
        "TEMPORAL",
        "COUNTERFACTUAL",
        "ADVERSARIAL",
        "REALISTIC_ABSTRACTED",
        "SYNTHETIC",
    )
    for source in source_order:
        required = source_targets.get(source, 0)
        source_selected = 0
        source_queues = queues.get(source, {})
        tasks = sorted(source_queues)
        while source_selected < required:
            remaining = required - source_selected
            candidates: list[
                tuple[str, str, tuple[DatasetRecord, ...]]
            ] = []
            for task in tasks:
                task_queues = source_queues[task]
                for decision_key, queue in task_queues.items():
                    if queue and len(queue[0]) <= remaining:
                        candidates.append(
                            (task, decision_key, queue[0])
                        )
            if not candidates:
                raise BuildError(
                    f"quota {source} non componibile: "
                    f"{source_selected}/{required}"
                )

            def candidate_score(
                candidate: tuple[
                    str, str, tuple[DatasetRecord, ...]
                ],
            ) -> tuple[float, int, str]:
                task, _, unit = candidate
                next_decisions = decision_counts + Counter(
                    item.oracle_result.decision.decision
                    for item in unit
                )
                decision_deficit = sum(
                    abs(
                        desired_decisions[decision]
                        - next_decisions[decision]
                    )
                    for decision in desired_decisions
                )
                return (
                    float(decision_deficit),
                    task_counts[task],
                    unit[0].scenario.scenario_id,
                )

            task, decision_key, unit = min(
                candidates, key=candidate_score
            )
            source_queues[task][decision_key].popleft()
            selected.extend(unit)
            source_selected += len(unit)
            task_counts.update(
                item.scenario.task for item in unit
            )
            decision_counts.update(
                item.oracle_result.decision.decision for item in unit
            )
    if len(selected) != target:
        raise BuildError(
            f"quote sourceType non raggiunte: {len(selected)}/{target}"
        )
    return tuple(selected)


def print_report(report: Mapping[str, object]) -> None:
    rows = (
        ("conteggi", report["counts"]),
        ("task", report["tasks"]),
        ("decisioni", report["decisions"]),
        ("sourceType", report["sourceTypes"]),
        ("famiglie", report["families"]),
        ("firme uniche", report["uniqueCausalSignatures"]),
        ("duplicati rimossi", report["duplicatesRemoved"]),
        ("% revisionata", report["reviewedPercentage"]),
        ("% controfattuale", report["counterfactualPercentage"]),
        ("% temporale", report["temporalPercentage"]),
    )
    width = max(len(label) for label, _ in rows)
    print(f"{'METRICA':<{width}} | VALORE")
    print(f"{'-' * width}-+-{'-' * 60}")
    for label, value in rows:
        rendered = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else str(value)
        )
        print(f"{label:<{width}} | {rendered}")
    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera una release riproducibile del dataset LOLAI."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-per-task", type=int)
    args = parser.parse_args(argv)
    if args.limit_per_task is not None and args.limit_per_task < 1:
        parser.error("--limit-per-task deve essere >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config.resolve())
        report = build(
            config,
            args.output.resolve(),
            force=args.force,
            dry_run=args.dry_run,
            limit_per_task=args.limit_per_task,
        )
    except (BuildError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"BUILD FALLITA: {error}", file=sys.stderr)
        return 1
    print_report(report)
    print("DRY RUN COMPLETATA" if args.dry_run else "BUILD COMPLETATA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
