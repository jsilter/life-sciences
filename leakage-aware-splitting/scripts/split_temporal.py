#!/usr/bin/env python3
"""Temporal / deposition-date splitting: train on the past, test on the future.

Time-split is the closest cheap proxy to true prospective performance (Sheridan
2013): random selection is too optimistic, leave-class-out too pessimistic. AlphaFold2
validated this way ("all structures deposited after our training cutoff"). It mimics
real deployment and captures all leakage types as they actually emerge.

Caveat encoded here: a temporal split alone does NOT guarantee novelty - the
post-cutoff set can still contain close homologs of pre-cutoff entries. Best practice
combines a temporal cutoff with a similarity filter; this module produces the
temporal assignment and reports which test items still have a near-duplicate in train
so the caller can additionally filter them.
"""

from __future__ import annotations

from typing import Callable, Sequence


def temporal_split(
    dates: Sequence,
    fractions: dict[str, float] | None = None,
) -> list[str]:
    """Assign splits by chronological order so max(train) <= min(val) <= min(test).

    `dates` may be ISO strings, datetimes, or ints (anything orderable). Items are
    sorted by date; the earliest `train` fraction become train, then val, then the
    latest become test. Ties on the boundary date are resolved by keeping all items
    with the same date on the earlier side, which can shift realized fractions - that
    is the honest behaviour (you cannot place identical-date items on both sides
    without leaking the cutoff).
    """
    if fractions is None:
        fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    order = [s for s in ("train", "val", "test") if fractions.get(s, 0) > 0]
    n = len(dates)
    assignment: list = [None] * n
    if n == 0 or not order:
        return assignment
    idx_sorted = sorted(range(n), key=lambda i: dates[i])

    # Group indices that share an identical date, in chronological order. A whole
    # date-group is assigned to a single split, so one calendar date can never straddle a
    # boundary; when a date falls on a cut point it stays on the EARLIER side (the honest
    # choice - you cannot place identical-date items on both sides without leaking the
    # cutoff). Assigning per-group (rather than per-boundary post-correction) is what makes
    # this robust when a single date value spans more than one split boundary - the old
    # per-boundary push-back could re-split such a date across non-adjacent splits.
    groups: list[list[int]] = []
    for i in idx_sorted:
        if groups and dates[i] == dates[groups[-1][0]]:
            groups[-1].append(i)
        else:
            groups.append([i])

    # Cumulative item-count target for the right edge of each split except the last.
    targets = []
    acc = 0.0
    for s in order[:-1]:
        acc += fractions[s]
        targets.append(acc * n)

    # Greedily assign whole date-groups in chronological order, advancing to the next
    # split only once the current split has met its cumulative target. Because each split
    # receives a contiguous run of distinct date values, max(earlier) < min(later) holds
    # strictly (see check_temporal_monotonicity).
    split_i = 0
    placed = 0
    for g in groups:
        while split_i < len(targets) and placed >= targets[split_i]:
            split_i += 1
        for i in g:
            assignment[i] = order[split_i]
        placed += len(g)
    return assignment


def residual_homology_in_temporal_split(
    items: Sequence,
    assignment: Sequence[str],
    similarity: Callable[[object, object], float],
    threshold: float,
) -> list[int]:
    """Test-item indices that still have a >=threshold neighbour in train.

    A temporal split can pass chronology yet leak by homology. This flags the test
    items the caller should additionally filter (or at least disclose).
    """
    test_idx = [i for i, s in enumerate(assignment) if s == "test"]
    train_items = [items[i] for i, s in enumerate(assignment) if s == "train"]
    flagged = []
    for i in test_idx:
        if any(similarity(items[i], tr) >= threshold for tr in train_items):
            flagged.append(i)
    return flagged
