from __future__ import annotations

import math

from generators.domain_models import OracleTrace


def reconstruct_candidate_total(
    trace: OracleTrace,
    action_id: str,
) -> float:
    """Reconstruct an unclamped candidate total from trace contributions."""
    for candidate_score in trace.candidate_scores:
        if candidate_score.action_id == action_id:
            return sum(
                contribution.contribution
                for contribution in candidate_score.contributions
            )
    raise KeyError(f"azione non presente nel trace: {action_id}")


def reconstruct_confidence(trace: OracleTrace) -> float:
    """Reconstruct the unrounded confidence from trace contributions."""
    return sum(
        contribution.contribution
        for contribution in trace.confidence_contributions
    )


def trace_is_consistent(trace: OracleTrace) -> bool:
    """Return whether all trace totals can be reconstructed exactly."""
    scores_consistent = all(
        math.isclose(
            reconstruct_candidate_total(trace, score.action_id),
            score.unclamped_total,
            abs_tol=1e-12,
        )
        for score in trace.candidate_scores
    )
    confidence_consistent = math.isclose(
        reconstruct_confidence(trace),
        trace.confidence_unrounded,
        abs_tol=1e-12,
    )
    return scores_consistent and confidence_consistent
