from __future__ import annotations

import sys

from generators.policy_loader import (
    PolicyBundle,
    PolicyError,
    load_policies,
)


EXPECTED_DECISION_PRECEDENCE: tuple[str, ...] = (
    "REQUEST_REFRESH",
    "SUPPRESS",
    "SHOW",
)
REQUIRED_SCORING_COMPONENTS: frozenset[str] = frozenset(
    {
        "primaryThreatCounter",
        "secondaryThreatCounter",
        "supportsWinCondition",
        "fillsMissingFunction",
        "feasibility",
        "urgencyAlignment",
        "opportunityCost",
        "redundancy",
        "executionBurden",
        "staleEvidencePenalty",
        "conflictingEvidencePenalty",
    }
)
REQUIRED_CONFIDENCE_FACTORS: frozenset[str] = frozenset(
    {
        "evidenceQuality",
        "evidenceFreshness",
        "contextCompleteness",
        "candidateScoreMargin",
        "conflictCount",
        "uncertainFieldCount",
    }
)
REQUIRED_SPLIT_KEYS: frozenset[str] = frozenset(
    {"familyId", "splitGroup", "episodeId", "counterfactualPairId"}
)


def validate_scoring(bundle: PolicyBundle) -> list[str]:
    """Validate the required abstract scoring vocabulary."""
    errors: list[str] = []
    component_ids = {component.id for component in bundle.scoring.components}
    if component_ids != REQUIRED_SCORING_COMPONENTS:
        errors.append(
            "scoring-policy: componenti non allineati; "
            f"mancanti={sorted(REQUIRED_SCORING_COMPONENTS - component_ids)}, "
            f"extra={sorted(component_ids - REQUIRED_SCORING_COMPONENTS)}"
        )
    if not any(
        component.direction == "REWARD"
        for component in bundle.scoring.components
    ):
        errors.append("scoring-policy: manca almeno un componente REWARD")
    if not any(
        component.direction == "PENALTY"
        for component in bundle.scoring.components
    ):
        errors.append("scoring-policy: manca almeno un componente PENALTY")
    return errors


def validate_decision(bundle: PolicyBundle) -> list[str]:
    """Validate precedence and sensible decision thresholds."""
    errors: list[str] = []
    actual_precedence = tuple(
        rule.decision for rule in bundle.decision.precedence
    )
    if actual_precedence != EXPECTED_DECISION_PRECEDENCE:
        errors.append(
            "decision-policy: precedenza attesa "
            f"{EXPECTED_DECISION_PRECEDENCE}, ottenuta {actual_precedence}"
        )

    thresholds = bundle.decision.thresholds
    if thresholds.stale_context_seconds > 3600:
        errors.append(
            "decision-policy: staleContextSeconds supera il range sensato"
        )
    if thresholds.recently_advised_seconds > 3600:
        errors.append(
            "decision-policy: recentlyAdvisedSeconds supera il range sensato"
        )
    if thresholds.maximum_contradiction_count > 100:
        errors.append(
            "decision-policy: maximumContradictionCount supera il range sensato"
        )
    return errors


def validate_confidence(bundle: PolicyBundle) -> list[str]:
    """Validate required confidence inputs and bounded output."""
    errors: list[str] = []
    factor_ids = {factor.id for factor in bundle.confidence.factors}
    if factor_ids != REQUIRED_CONFIDENCE_FACTORS:
        errors.append(
            "confidence-policy: fattori non allineati; "
            f"mancanti={sorted(REQUIRED_CONFIDENCE_FACTORS - factor_ids)}, "
            f"extra={sorted(factor_ids - REQUIRED_CONFIDENCE_FACTORS)}"
        )
    if bundle.confidence.output_range.minimum != 0.0:
        errors.append("confidence-policy: output minimum deve essere 0")
    if bundle.confidence.output_range.maximum != 1.0:
        errors.append("confidence-policy: output maximum deve essere 1")
    if bundle.confidence.rounding_digits > 6:
        errors.append("confidence-policy: roundingDigits supera il range sensato")
    return errors


def validate_generation(bundle: PolicyBundle) -> list[str]:
    """Validate practical generation bounds and percentage targets."""
    errors: list[str] = []
    policy = bundle.generation
    if policy.candidate_count.maximum > 100:
        errors.append("generation-policy: candidateCount.maximum troppo alto")
    if policy.evidence_count.maximum > 100:
        errors.append("generation-policy: evidenceCount.maximum troppo alto")
    percentage_total = sum(policy.scenario_percentages.values())
    if percentage_total > 1.0 + 1e-9:
        errors.append(
            "generation-policy: le percentuali di scenario superano 1"
        )
    return errors


def validate_deduplication(bundle: PolicyBundle) -> list[str]:
    """Validate causal signature and structural deduplication safeguards."""
    errors: list[str] = []
    policy = bundle.deduplication
    if not any(
        field.startswith("input.") for field in policy.causal_signature_fields
    ):
        errors.append(
            "deduplication-policy: la firma deve includere campi input"
        )
    if not any(
        field.startswith("expectedOutput.")
        for field in policy.causal_signature_fields
    ):
        errors.append(
            "deduplication-policy: la firma deve includere expectedOutput"
        )
    if policy.maximum_examples_per_signature > 1000:
        errors.append(
            "deduplication-policy: maximumExamplesPerSignature troppo alto"
        )
    if policy.structural_similarity_threshold < 0.5:
        errors.append(
            "deduplication-policy: structuralSimilarityThreshold troppo bassa"
        )
    return errors


def validate_split(bundle: PolicyBundle) -> list[str]:
    """Validate all leakage-prevention requirements."""
    errors: list[str] = []
    policy = bundle.split
    if policy.allow_random_per_row:
        errors.append("split-policy: lo split casuale per riga deve essere vietato")
    if not REQUIRED_SPLIT_KEYS.issubset(policy.group_keys):
        errors.append(
            "split-policy: groupKeys mancanti: "
            f"{sorted(REQUIRED_SPLIT_KEYS - set(policy.group_keys))}"
        )
    if not policy.keep_counterfactual_pairs_together:
        errors.append(
            "split-policy: le coppie controfattuali devono restare unite"
        )
    if not policy.keep_episodes_together:
        errors.append("split-policy: gli episodi devono restare uniti")
    if not policy.prevent_train_test_overlap:
        errors.append(
            "split-policy: deve impedire overlap di famiglia tra train e test"
        )
    if policy.family_key != "familyId":
        errors.append("split-policy: familyKey deve essere familyId")
    required_splits = {"train", "valid", "test", "challenge"}
    if set(policy.distribution) != required_splits:
        errors.append(
            "split-policy: distribuzione split non completa; "
            f"attesi={sorted(required_splits)}"
        )
    return errors


def main() -> int:
    """Load and validate the complete policy bundle."""
    try:
        bundle = load_policies()
    except (OSError, PolicyError) as error:
        print(f"Validazione policy fallita: {error}", file=sys.stderr)
        return 1

    errors = [
        *validate_scoring(bundle),
        *validate_decision(bundle),
        *validate_confidence(bundle),
        *validate_generation(bundle),
        *validate_deduplication(bundle),
        *validate_split(bundle),
    ]
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"Validazione policy fallita: {len(errors)} errori.",
            file=sys.stderr,
        )
        return 1

    print(
        "Policy valide: 6 file caricati, versioni coerenti, "
        "distribuzioni e riferimenti ontologici verificati."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
