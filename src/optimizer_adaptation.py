"""
Adaptive-search decisions for Optimizer Experience.

This module contains pure decision logic. It receives normalized Experience
history and returns Adaptive Evolution and Search Allocation settings without
performing filesystem I/O.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from optimizer_experience_stats import _build_search_source_statistics


DEFAULT_SEARCH_ALLOCATION = {
    "experience": 3,
    "random": 4,
    "local": 6,
    "evolution": 4,
}
MIN_SEARCH_ALLOCATION_SAMPLES = 5

DEFAULT_EVOLUTION_COUNT = 4
DEFAULT_MUTATION_RATE = 0.25
DEFAULT_MUTATION_SCALE = 0.08
MIN_ADAPTATION_SAMPLES = 5


def build_evolution_adaptation(
    normalized_history: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return Evolution count and mutation settings from normalized history."""
    source_statistics = _build_search_source_statistics(
        normalized_history
    )
    sample_count = len(normalized_history)

    evolution_stats = source_statistics["evolution"]
    evolution_win_count = int(
        evolution_stats["count"]
    )
    evolution_win_rate = (
        evolution_win_count / sample_count
        if sample_count > 0
        else 0.0
    )

    default_result = {
        "adaptive": False,
        "reason": "insufficient_history",
        "sample_count": sample_count,
        "evolution_win_count": evolution_win_count,
        "evolution_win_rate": round(
            evolution_win_rate,
            6,
        ),
        "count": DEFAULT_EVOLUTION_COUNT,
        "mutation_rate": DEFAULT_MUTATION_RATE,
        "mutation_scale": DEFAULT_MUTATION_SCALE,
        "source_statistics": source_statistics,
    }

    if sample_count < MIN_ADAPTATION_SAMPLES:
        return default_result

    if evolution_win_rate >= 0.40:
        count = 6
        mutation_rate = 0.18
        mutation_scale = 0.06
        reason = "evolution_high_performance"
    elif evolution_win_rate >= 0.20:
        count = DEFAULT_EVOLUTION_COUNT
        mutation_rate = DEFAULT_MUTATION_RATE
        mutation_scale = DEFAULT_MUTATION_SCALE
        reason = "evolution_normal_performance"
    else:
        count = 3
        mutation_rate = 0.35
        mutation_scale = 0.10
        reason = "evolution_low_performance"

    return {
        **default_result,
        "adaptive": True,
        "reason": reason,
        "count": count,
        "mutation_rate": mutation_rate,
        "mutation_scale": mutation_scale,
    }


def build_search_allocation(
    normalized_history: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return the adaptive weight-config search allocation."""
    default_counts = dict(
        DEFAULT_SEARCH_ALLOCATION
    )
    source_statistics = _build_search_source_statistics(
        normalized_history
    )
    sample_count = len(normalized_history)

    default_result = {
        "adaptive": False,
        "reason": "insufficient_history",
        "sample_count": sample_count,
        "total_count": sum(
            default_counts.values()
        ),
        "counts": default_counts,
        "receiver": None,
        "donor": None,
        "source_scores": {},
        "source_statistics": source_statistics,
    }

    if sample_count < MIN_SEARCH_ALLOCATION_SAMPLES:
        return default_result

    source_names = tuple(
        DEFAULT_SEARCH_ALLOCATION
    )
    adaptive_win_count = sum(
        int(
            source_statistics[source]["count"]
        )
        for source in source_names
    )

    if adaptive_win_count == 0:
        return {
            **default_result,
            "reason": "no_adaptive_source_wins",
        }

    average_scores = {
        source: float(
            source_statistics[source][
                "average_selection_score"
            ]
        )
        for source in source_names
        if int(
            source_statistics[source]["count"]
        ) > 0
    }
    minimum_average = min(
        average_scores.values(),
        default=0.0,
    )
    maximum_average = max(
        average_scores.values(),
        default=0.0,
    )
    average_span = (
        maximum_average - minimum_average
    )

    source_scores: dict[str, float] = {}

    for source in source_names:
        stats = source_statistics[source]
        count = int(stats["count"])
        win_share = (
            count / adaptive_win_count
        )
        recent_share = float(
            stats["recent_share"]
        )
        average_score = float(
            stats["average_selection_score"]
        )

        if count == 0:
            normalized_average = 0.0
        elif average_span > 0.0:
            normalized_average = (
                average_score - minimum_average
            ) / average_span
        else:
            normalized_average = 1.0

        confidence = min(
            count / 5.0,
            1.0,
        )
        combined_score = (
            0.45 * win_share
            + 0.35 * recent_share
            + 0.20 * normalized_average
        ) * (
            0.5 + 0.5 * confidence
        )

        source_scores[source] = round(
            combined_score,
            6,
        )

    receiver = max(
        source_names,
        key=lambda source: (
            source_scores[source],
            int(
                source_statistics[source][
                    "recent_count"
                ]
            ),
            int(
                source_statistics[source]["count"]
            ),
        ),
    )

    donor_candidates = [
        source
        for source in source_names
        if (
            source != receiver
            and default_counts[source] > 1
        )
    ]
    donor = min(
        donor_candidates,
        key=lambda source: (
            source_scores[source],
            int(
                source_statistics[source][
                    "recent_count"
                ]
            ),
            int(
                source_statistics[source]["count"]
            ),
        ),
    )

    counts = dict(default_counts)
    counts[receiver] += 1
    counts[donor] -= 1

    return {
        "adaptive": True,
        "reason": "performance_reallocation",
        "sample_count": sample_count,
        "total_count": sum(
            counts.values()
        ),
        "counts": counts,
        "receiver": receiver,
        "donor": donor,
        "source_scores": source_scores,
        "source_statistics": source_statistics,
    }


__all__ = [
    "build_evolution_adaptation",
    "build_search_allocation",
]
