from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from generators.ontology_registry import (
    OntologyRegistry,
    default_ontology_dir,
)


class PolicyError(ValueError):
    """Base error raised for an invalid policy bundle."""


class MissingPolicyKeyError(PolicyError):
    """Raised when a required policy key is absent."""


class UnexpectedPolicyKeyError(PolicyError):
    """Raised when a policy contains an unsupported key."""


class InconsistentPolicyVersionError(PolicyError):
    """Raised when policy files do not share manifest versions."""


@dataclass(frozen=True, slots=True)
class NumericRange:
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class CountRange:
    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class ScoringComponent:
    id: str
    description: str
    direction: str
    weight: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoringPolicy:
    description: str
    components: tuple[ScoringComponent, ...]
    input_range: NumericRange
    output_range: NumericRange


@dataclass(frozen=True, slots=True)
class DecisionRule:
    rank: int
    decision: str
    description: str
    condition_keys: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    stale_context_seconds: int
    minimum_completeness: float
    minimum_evidence_confidence: float
    meaningful_score_margin: float
    recently_advised_seconds: int
    low_action_value_threshold: float
    maximum_contradiction_count: int


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    description: str
    precedence: tuple[DecisionRule, ...]
    thresholds: DecisionThresholds


@dataclass(frozen=True, slots=True)
class ConfidenceFactor:
    id: str
    description: str
    direction: str
    weight: float


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    description: str
    factors: tuple[ConfidenceFactor, ...]
    output_range: NumericRange
    rounding_digits: int


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    description: str
    global_seed: int
    task_distribution: Mapping[str, float]
    decision_distribution: Mapping[str, float]
    candidate_count: CountRange
    evidence_count: CountRange
    scenario_percentages: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class DeduplicationPolicy:
    description: str
    causal_signature_fields: tuple[str, ...]
    numeric_buckets: Mapping[str, tuple[float, ...]]
    maximum_examples_per_signature: int
    structural_similarity_threshold: float


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    description: str
    strategy: str
    allow_random_per_row: bool
    group_keys: tuple[str, ...]
    keep_counterfactual_pairs_together: bool
    keep_episodes_together: bool
    prevent_train_test_overlap: bool
    family_key: str
    distribution: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    policy_version: str
    ontology_version: str
    scoring: ScoringPolicy
    decision: DecisionPolicy
    confidence: ConfidencePolicy
    generation: GenerationPolicy
    deduplication: DeduplicationPolicy
    split: SplitPolicy


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PolicyError(f"{path}: deve essere un oggetto JSON")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    required: set[str],
    location: str,
) -> None:
    missing = required - value.keys()
    if missing:
        raise MissingPolicyKeyError(
            f"{location}: chiavi mancanti: {sorted(missing)}"
        )
    unexpected = value.keys() - required
    if unexpected:
        raise UnexpectedPolicyKeyError(
            f"{location}: chiavi non previste: {sorted(unexpected)}"
        )


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{location}: deve essere un oggetto")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise PolicyError(f"{location}: deve essere una lista")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{location}: deve essere una stringa non vuota")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(f"{location}: deve essere booleano")
    return value


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise PolicyError(
            f"{location}: deve essere un intero >= {minimum}"
        )
    return value


def _number(
    value: Any,
    location: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise PolicyError(
            f"{location}: deve essere numerico tra {minimum} e {maximum}"
        )
    return float(value)


def _strings(value: Any, location: str, minimum_items: int = 1) -> tuple[str, ...]:
    items = _array(value, location)
    if len(items) < minimum_items:
        raise PolicyError(
            f"{location}: richiede almeno {minimum_items} elementi"
        )
    result = tuple(
        _string(item, f"{location}[{index}]")
        for index, item in enumerate(items)
    )
    if len(set(result)) != len(result):
        raise PolicyError(f"{location}: contiene valori duplicati")
    return result


def _numeric_range(
    value: Any,
    location: str,
    allowed_minimum: float,
    allowed_maximum: float,
) -> NumericRange:
    obj = _object(value, location)
    _strict_keys(obj, {"minimum", "maximum"}, location)
    minimum = _number(
        obj["minimum"], f"{location}.minimum", allowed_minimum, allowed_maximum
    )
    maximum = _number(
        obj["maximum"], f"{location}.maximum", allowed_minimum, allowed_maximum
    )
    if minimum >= maximum:
        raise PolicyError(f"{location}: minimum deve essere minore di maximum")
    return NumericRange(minimum=minimum, maximum=maximum)


def _count_range(value: Any, location: str) -> CountRange:
    obj = _object(value, location)
    _strict_keys(obj, {"minimum", "maximum"}, location)
    minimum = _integer(obj["minimum"], f"{location}.minimum", minimum=1)
    maximum = _integer(obj["maximum"], f"{location}.maximum", minimum=1)
    if minimum > maximum:
        raise PolicyError(f"{location}: minimum non può superare maximum")
    return CountRange(minimum=minimum, maximum=maximum)


def _distribution(
    value: Any,
    location: str,
    allowed_ids: frozenset[str] | None = None,
) -> Mapping[str, float]:
    obj = _object(value, location)
    if not obj:
        raise PolicyError(f"{location}: non può essere vuota")
    if allowed_ids is not None and set(obj) != set(allowed_ids):
        raise PolicyError(
            f"{location}: chiavi non allineate; "
            f"mancanti={sorted(allowed_ids - obj.keys())}, "
            f"extra={sorted(obj.keys() - allowed_ids)}"
        )
    result = {
        _string(key, f"{location}.key"): _number(
            amount, f"{location}.{key}", 0.0, 1.0
        )
        for key, amount in obj.items()
    }
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise PolicyError(
            f"{location}: la distribuzione deve sommare a 1, "
            f"ottenuto {sum(result.values())}"
        )
    return MappingProxyType(result)


def _policy_header(
    document: Mapping[str, Any],
    location: str,
    policy_version: str,
    ontology_version: str,
    body_keys: set[str],
) -> None:
    _strict_keys(
        document,
        {"policyVersion", "ontologyVersion", "description"} | body_keys,
        location,
    )
    if document["policyVersion"] != policy_version:
        raise InconsistentPolicyVersionError(
            f"{location}: policyVersion incoerente"
        )
    if document["ontologyVersion"] != ontology_version:
        raise InconsistentPolicyVersionError(
            f"{location}: ontologyVersion incoerente"
        )
    _string(document["description"], f"{location}.description")


def _reason_codes(
    value: Any,
    location: str,
    ontology: OntologyRegistry,
) -> tuple[str, ...]:
    reason_codes = _strings(value, location)
    for reason_code in reason_codes:
        if not ontology.contains("reason-codes", reason_code):
            raise PolicyError(
                f"{location}: reason code inesistente {reason_code!r}"
            )
    return reason_codes


def _load_scoring(
    document: Mapping[str, Any],
    location: str,
    policy_version: str,
    ontology_version: str,
    ontology: OntologyRegistry,
) -> ScoringPolicy:
    _policy_header(
        document,
        location,
        policy_version,
        ontology_version,
        {"components", "inputRange", "outputRange"},
    )
    components: list[ScoringComponent] = []
    component_ids: set[str] = set()
    for index, raw_component in enumerate(
        _array(document["components"], f"{location}.components")
    ):
        item_location = f"{location}.components[{index}]"
        component = _object(raw_component, item_location)
        _strict_keys(
            component,
            {"id", "description", "direction", "weight", "reasonCodes"},
            item_location,
        )
        component_id = _string(component["id"], f"{item_location}.id")
        if component_id in component_ids:
            raise PolicyError(f"{item_location}.id: duplicato {component_id!r}")
        component_ids.add(component_id)
        direction = _string(
            component["direction"], f"{item_location}.direction"
        )
        if direction not in {"REWARD", "PENALTY"}:
            raise PolicyError(f"{item_location}.direction: valore non valido")
        components.append(
            ScoringComponent(
                id=component_id,
                description=_string(
                    component["description"], f"{item_location}.description"
                ),
                direction=direction,
                weight=_number(
                    component["weight"], f"{item_location}.weight", 0.0, 1.0
                ),
                reason_codes=_reason_codes(
                    component["reasonCodes"],
                    f"{item_location}.reasonCodes",
                    ontology,
                ),
            )
        )
    if not components:
        raise PolicyError(f"{location}.components: non può essere vuoto")
    weight_sum = sum(component.weight for component in components)
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
        raise PolicyError(
            f"{location}.components: i pesi devono sommare a 1, "
            f"ottenuto {weight_sum}"
        )
    return ScoringPolicy(
        description=_string(document["description"], f"{location}.description"),
        components=tuple(components),
        input_range=_numeric_range(
            document["inputRange"], f"{location}.inputRange", 0.0, 1.0
        ),
        output_range=_numeric_range(
            document["outputRange"], f"{location}.outputRange", -1.0, 1.0
        ),
    )


def _load_decision(
    document: Mapping[str, Any],
    location: str,
    policy_version: str,
    ontology_version: str,
    ontology: OntologyRegistry,
) -> DecisionPolicy:
    _policy_header(
        document,
        location,
        policy_version,
        ontology_version,
        {"precedence", "thresholds"},
    )
    rules: list[DecisionRule] = []
    for index, raw_rule in enumerate(
        _array(document["precedence"], f"{location}.precedence")
    ):
        item_location = f"{location}.precedence[{index}]"
        rule = _object(raw_rule, item_location)
        _strict_keys(
            rule,
            {
                "rank",
                "decision",
                "description",
                "conditionKeys",
                "reasonCodes",
            },
            item_location,
        )
        decision = _string(rule["decision"], f"{item_location}.decision")
        if not ontology.contains("decisions", decision):
            raise PolicyError(
                f"{item_location}.decision: decisione inesistente {decision!r}"
            )
        rules.append(
            DecisionRule(
                rank=_integer(rule["rank"], f"{item_location}.rank", minimum=1),
                decision=decision,
                description=_string(
                    rule["description"], f"{item_location}.description"
                ),
                condition_keys=_strings(
                    rule["conditionKeys"], f"{item_location}.conditionKeys"
                ),
                reason_codes=_reason_codes(
                    rule["reasonCodes"],
                    f"{item_location}.reasonCodes",
                    ontology,
                ),
            )
        )
    if tuple(rule.rank for rule in rules) != tuple(range(1, len(rules) + 1)):
        raise PolicyError(f"{location}.precedence: rank non consecutivi")
    if {rule.decision for rule in rules} != set(ontology.ids("decisions")):
        raise PolicyError(
            f"{location}.precedence: deve includere ogni decisione una volta"
        )

    thresholds = _object(document["thresholds"], f"{location}.thresholds")
    threshold_keys = {
        "staleContextSeconds",
        "minimumCompleteness",
        "minimumEvidenceConfidence",
        "meaningfulScoreMargin",
        "recentlyAdvisedSeconds",
        "lowActionValueThreshold",
        "maximumContradictionCount",
    }
    _strict_keys(thresholds, threshold_keys, f"{location}.thresholds")
    parsed_thresholds = DecisionThresholds(
        stale_context_seconds=_integer(
            thresholds["staleContextSeconds"],
            f"{location}.thresholds.staleContextSeconds",
            minimum=1,
        ),
        minimum_completeness=_number(
            thresholds["minimumCompleteness"],
            f"{location}.thresholds.minimumCompleteness",
            0.0,
            1.0,
        ),
        minimum_evidence_confidence=_number(
            thresholds["minimumEvidenceConfidence"],
            f"{location}.thresholds.minimumEvidenceConfidence",
            0.0,
            1.0,
        ),
        meaningful_score_margin=_number(
            thresholds["meaningfulScoreMargin"],
            f"{location}.thresholds.meaningfulScoreMargin",
            0.0,
            1.0,
        ),
        recently_advised_seconds=_integer(
            thresholds["recentlyAdvisedSeconds"],
            f"{location}.thresholds.recentlyAdvisedSeconds",
            minimum=1,
        ),
        low_action_value_threshold=_number(
            thresholds["lowActionValueThreshold"],
            f"{location}.thresholds.lowActionValueThreshold",
            0.0,
            1.0,
        ),
        maximum_contradiction_count=_integer(
            thresholds["maximumContradictionCount"],
            f"{location}.thresholds.maximumContradictionCount",
            minimum=0,
        ),
    )
    return DecisionPolicy(
        description=_string(document["description"], f"{location}.description"),
        precedence=tuple(rules),
        thresholds=parsed_thresholds,
    )


def _load_confidence(
    document: Mapping[str, Any],
    location: str,
    policy_version: str,
    ontology_version: str,
) -> ConfidencePolicy:
    _policy_header(
        document,
        location,
        policy_version,
        ontology_version,
        {"factors", "outputRange", "roundingDigits"},
    )
    factors: list[ConfidenceFactor] = []
    factor_ids: set[str] = set()
    for index, raw_factor in enumerate(
        _array(document["factors"], f"{location}.factors")
    ):
        item_location = f"{location}.factors[{index}]"
        factor = _object(raw_factor, item_location)
        _strict_keys(
            factor,
            {"id", "description", "direction", "weight"},
            item_location,
        )
        factor_id = _string(factor["id"], f"{item_location}.id")
        if factor_id in factor_ids:
            raise PolicyError(f"{item_location}.id: duplicato {factor_id!r}")
        factor_ids.add(factor_id)
        direction = _string(factor["direction"], f"{item_location}.direction")
        if direction not in {"POSITIVE", "NEGATIVE"}:
            raise PolicyError(f"{item_location}.direction: valore non valido")
        factors.append(
            ConfidenceFactor(
                id=factor_id,
                description=_string(
                    factor["description"], f"{item_location}.description"
                ),
                direction=direction,
                weight=_number(
                    factor["weight"], f"{item_location}.weight", 0.0, 1.0
                ),
            )
        )
    weight_sum = sum(factor.weight for factor in factors)
    if not factors or not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
        raise PolicyError(
            f"{location}.factors: i pesi devono sommare a 1, "
            f"ottenuto {weight_sum}"
        )
    return ConfidencePolicy(
        description=_string(document["description"], f"{location}.description"),
        factors=tuple(factors),
        output_range=_numeric_range(
            document["outputRange"], f"{location}.outputRange", 0.0, 1.0
        ),
        rounding_digits=_integer(
            document["roundingDigits"],
            f"{location}.roundingDigits",
            minimum=0,
        ),
    )


def _load_generation(
    document: Mapping[str, Any],
    location: str,
    policy_version: str,
    ontology_version: str,
    ontology: OntologyRegistry,
) -> GenerationPolicy:
    _policy_header(
        document,
        location,
        policy_version,
        ontology_version,
        {
            "globalSeed",
            "taskDistribution",
            "decisionDistribution",
            "candidateCount",
            "evidenceCount",
            "scenarioPercentages",
        },
    )
    percentages_obj = _object(
        document["scenarioPercentages"], f"{location}.scenarioPercentages"
    )
    percentage_keys = {
        "counterfactual",
        "adversarial",
        "temporalEpisode",
        "realisticAbstracted",
        "humanReview",
    }
    _strict_keys(
        percentages_obj,
        percentage_keys,
        f"{location}.scenarioPercentages",
    )
    percentages = MappingProxyType(
        {
            key: _number(
                percentages_obj[key],
                f"{location}.scenarioPercentages.{key}",
                0.0,
                1.0,
            )
            for key in percentage_keys
        }
    )
    return GenerationPolicy(
        description=_string(document["description"], f"{location}.description"),
        global_seed=_integer(
            document["globalSeed"], f"{location}.globalSeed", minimum=0
        ),
        task_distribution=_distribution(
            document["taskDistribution"],
            f"{location}.taskDistribution",
            ontology.ids("tasks"),
        ),
        decision_distribution=_distribution(
            document["decisionDistribution"],
            f"{location}.decisionDistribution",
            ontology.ids("decisions"),
        ),
        candidate_count=_count_range(
            document["candidateCount"], f"{location}.candidateCount"
        ),
        evidence_count=_count_range(
            document["evidenceCount"], f"{location}.evidenceCount"
        ),
        scenario_percentages=percentages,
    )


def _load_deduplication(
    document: Mapping[str, Any],
    location: str,
    policy_version: str,
    ontology_version: str,
) -> DeduplicationPolicy:
    _policy_header(
        document,
        location,
        policy_version,
        ontology_version,
        {
            "causalSignatureFields",
            "numericBuckets",
            "maximumExamplesPerSignature",
            "structuralSimilarityThreshold",
        },
    )
    bucket_obj = _object(
        document["numericBuckets"], f"{location}.numericBuckets"
    )
    if not bucket_obj:
        raise PolicyError(f"{location}.numericBuckets: non può essere vuoto")
    buckets: dict[str, tuple[float, ...]] = {}
    for field, raw_boundaries in bucket_obj.items():
        field_name = _string(field, f"{location}.numericBuckets.key")
        boundaries = tuple(
            _number(
                boundary,
                f"{location}.numericBuckets.{field_name}[{index}]",
                0.0,
                float("inf"),
            )
            for index, boundary in enumerate(
                _array(
                    raw_boundaries,
                    f"{location}.numericBuckets.{field_name}",
                )
            )
        )
        if len(boundaries) < 2 or any(
            left >= right
            for left, right in zip(boundaries, boundaries[1:])
        ):
            raise PolicyError(
                f"{location}.numericBuckets.{field_name}: "
                "confini non strettamente crescenti"
            )
        buckets[field_name] = boundaries
    return DeduplicationPolicy(
        description=_string(document["description"], f"{location}.description"),
        causal_signature_fields=_strings(
            document["causalSignatureFields"],
            f"{location}.causalSignatureFields",
        ),
        numeric_buckets=MappingProxyType(buckets),
        maximum_examples_per_signature=_integer(
            document["maximumExamplesPerSignature"],
            f"{location}.maximumExamplesPerSignature",
            minimum=1,
        ),
        structural_similarity_threshold=_number(
            document["structuralSimilarityThreshold"],
            f"{location}.structuralSimilarityThreshold",
            0.0,
            1.0,
        ),
    )


def _load_split(
    document: Mapping[str, Any],
    location: str,
    policy_version: str,
    ontology_version: str,
) -> SplitPolicy:
    _policy_header(
        document,
        location,
        policy_version,
        ontology_version,
        {
            "strategy",
            "allowRandomPerRow",
            "groupKeys",
            "cohesionRules",
            "templateFamilyIsolation",
            "distribution",
        },
    )
    cohesion = _object(
        document["cohesionRules"], f"{location}.cohesionRules"
    )
    _strict_keys(
        cohesion,
        {"keepCounterfactualPairsTogether", "keepEpisodesTogether"},
        f"{location}.cohesionRules",
    )
    isolation = _object(
        document["templateFamilyIsolation"],
        f"{location}.templateFamilyIsolation",
    )
    _strict_keys(
        isolation,
        {"preventTrainTestOverlap", "familyKey"},
        f"{location}.templateFamilyIsolation",
    )
    return SplitPolicy(
        description=_string(document["description"], f"{location}.description"),
        strategy=_string(document["strategy"], f"{location}.strategy"),
        allow_random_per_row=_boolean(
            document["allowRandomPerRow"], f"{location}.allowRandomPerRow"
        ),
        group_keys=_strings(document["groupKeys"], f"{location}.groupKeys"),
        keep_counterfactual_pairs_together=_boolean(
            cohesion["keepCounterfactualPairsTogether"],
            f"{location}.cohesionRules.keepCounterfactualPairsTogether",
        ),
        keep_episodes_together=_boolean(
            cohesion["keepEpisodesTogether"],
            f"{location}.cohesionRules.keepEpisodesTogether",
        ),
        prevent_train_test_overlap=_boolean(
            isolation["preventTrainTestOverlap"],
            f"{location}.templateFamilyIsolation.preventTrainTestOverlap",
        ),
        family_key=_string(
            isolation["familyKey"],
            f"{location}.templateFamilyIsolation.familyKey",
        ),
        distribution=_distribution(
            document["distribution"], f"{location}.distribution"
        ),
    )


def default_policy_dir() -> Path:
    """Return the repository policy directory."""
    return Path(__file__).resolve().parents[1] / "policies"


def load_policies(
    policy_dir: Path | None = None,
    ontology_dir: Path | None = None,
) -> PolicyBundle:
    """Load and strictly validate the complete immutable policy bundle."""
    resolved_policy_dir = (policy_dir or default_policy_dir()).resolve()
    ontology = OntologyRegistry(ontology_dir or default_ontology_dir())
    manifest_path = resolved_policy_dir / "manifest.json"
    manifest = _load_object(manifest_path)
    _strict_keys(
        manifest,
        {"policyVersion", "ontologyVersion", "description", "policies"},
        str(manifest_path),
    )
    policy_version = _string(
        manifest["policyVersion"], f"{manifest_path}.policyVersion"
    )
    ontology_version = _string(
        manifest["ontologyVersion"], f"{manifest_path}.ontologyVersion"
    )
    if ontology_version != ontology.version:
        raise InconsistentPolicyVersionError(
            f"{manifest_path}: ontologyVersion {ontology_version!r}, "
            f"attesa {ontology.version!r}"
        )

    manifest_entries = _array(manifest["policies"], f"{manifest_path}.policies")
    filenames: dict[str, str] = {}
    for index, raw_entry in enumerate(manifest_entries):
        location = f"{manifest_path}.policies[{index}]"
        entry = _object(raw_entry, location)
        _strict_keys(entry, {"id", "file", "description"}, location)
        policy_id = _string(entry["id"], f"{location}.id")
        if policy_id in filenames:
            raise PolicyError(f"{location}.id: duplicato {policy_id!r}")
        filenames[policy_id] = _string(entry["file"], f"{location}.file")
        _string(entry["description"], f"{location}.description")

    required_policy_ids = {
        "scoring",
        "decision",
        "confidence",
        "generation",
        "deduplication",
        "split",
    }
    if set(filenames) != required_policy_ids:
        raise PolicyError(
            f"{manifest_path}: policy non allineate; "
            f"mancanti={sorted(required_policy_ids - filenames.keys())}, "
            f"extra={sorted(filenames.keys() - required_policy_ids)}"
        )

    documents = {
        policy_id: _load_object(resolved_policy_dir / filename)
        for policy_id, filename in filenames.items()
    }
    return PolicyBundle(
        policy_version=policy_version,
        ontology_version=ontology_version,
        scoring=_load_scoring(
            documents["scoring"],
            filenames["scoring"],
            policy_version,
            ontology_version,
            ontology,
        ),
        decision=_load_decision(
            documents["decision"],
            filenames["decision"],
            policy_version,
            ontology_version,
            ontology,
        ),
        confidence=_load_confidence(
            documents["confidence"],
            filenames["confidence"],
            policy_version,
            ontology_version,
        ),
        generation=_load_generation(
            documents["generation"],
            filenames["generation"],
            policy_version,
            ontology_version,
            ontology,
        ),
        deduplication=_load_deduplication(
            documents["deduplication"],
            filenames["deduplication"],
            policy_version,
            ontology_version,
        ),
        split=_load_split(
            documents["split"],
            filenames["split"],
            policy_version,
            ontology_version,
        ),
    )
