#!/usr/bin/env python3
"""Protein-ligand (drug-target) splitting: control leakage on BOTH axes.

Under a random split, deep DTI models reach AUROC > 0.98 - but this reflects hidden
ligand bias (correct predictions from drug features alone, not interaction patterns).
The honest split is "double-cold" (cold-drug + cold-target): test pairs whose protein
AND ligand are both unseen in training. PDBbind->CASF leakage is the canonical case
(>700 training complexes share near-duplicate similarity with CASF, ~45% of test
complexes), and de-leaking it drops top models toward baseline.

This module implements a greedy double-cold split as a dependency-free fallback and
documents DataSAIL as the production tool that solves the 2D assignment optimally.

A non-reported protein-ligand pair is *untested*, not a confirmed negative - keep
that in mind when constructing negatives; this splitter only partitions the pairs you
give it.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence


def double_cold_split(
    protein_cluster: Sequence[int],
    ligand_cluster: Sequence[int],
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> tuple[list[str | None], dict]:
    """Greedy cold-drug + cold-target split.

    Each pair i has a protein cluster and a ligand cluster. We partition the set of
    protein clusters and (independently) the set of ligand clusters into the requested
    splits. A pair is assigned to split S only if BOTH its protein cluster and its
    ligand cluster were assigned to S; pairs falling in the off-diagonal blocks
    (protein in train, ligand in test, etc.) are DISCARDED, because keeping them would
    leak one axis. Discarding is inherent to double-cold splits - the function reports
    how much was dropped so the caller can disclose it.

    Returns (assignment, info) where assignment[i] is a split name or None (discarded).
    """
    if fractions is None:
        fractions = {"train": 0.8, "test": 0.2}
    fractions = {k: v for k, v in fractions.items() if v > 0}
    splits = list(fractions.keys())

    # Only the diagonal (protein AND ligand axis assigned to the same split) is kept, so a
    # pair lands in split S with probability ~ axis_frac(S)^2. Assigning each axis with the
    # requested fractions directly would therefore collapse the kept fractions to ~target^2
    # (an 80/20 request keeps ~64/4, i.e. test ~5x too small). Assign each axis by
    # sqrt(target) (renormalised) instead, so the KEPT pairs approximate the requested ratio.
    axis_fr = {s: math.sqrt(fractions[s]) for s in splits}
    z = sum(axis_fr.values())
    axis_fr = {s: v / z for s, v in axis_fr.items()}

    # Size-weight each axis by the number of PAIRS touching each cluster, so the realized
    # item-level ratio tracks the request (a cluster-count balance would let a few huge
    # clusters swing the split arbitrarily). Mirrors core.assign_clusters_to_splits.
    prot_sizes = Counter(protein_cluster)
    lig_sizes = Counter(ligand_cluster)
    prot_clusters = sorted(prot_sizes)
    lig_clusters = sorted(lig_sizes)

    prot_assign = _assign_groups_by_fraction(prot_sizes, axis_fr, seed)
    lig_assign = _assign_groups_by_fraction(lig_sizes, axis_fr, seed + 1)

    assignment: list[str | None] = []
    kept = {s: 0 for s in splits}
    discarded = 0
    for p, l in zip(protein_cluster, ligand_cluster):
        ps, ls = prot_assign[p], lig_assign[l]
        if ps == ls:
            assignment.append(ps)
            kept[ps] += 1
        else:
            assignment.append(None)
            discarded += 1

    n = len(protein_cluster)
    n_kept = sum(kept.values())
    info = {
        "kept_per_split": kept,
        # Realized fractions over the KEPT pairs, so a caller sees the actual split sizes
        # rather than assuming they match the request (they cannot, exactly, because whole
        # off-diagonal blocks are dropped).
        "realized_fractions": {s: (kept[s] / n_kept if n_kept else 0.0) for s in splits},
        "axis_fractions": axis_fr,
        "discarded": discarded,
        "discarded_fraction": discarded / n if n else 0.0,
        "n_protein_clusters": len(prot_clusters),
        "n_ligand_clusters": len(lig_clusters),
        "note": "off-diagonal pairs discarded to keep both axes cold; axis fractions are "
                "sqrt-scaled so the kept pairs approximate the requested ratio. Use DataSAIL "
                "for an optimal 2D assignment that retains more data.",
    }
    return assignment, info


def _assign_groups_by_fraction(sizes: dict[int, int], fractions: dict[str, float], seed: int) -> dict[int, str]:
    """Deterministically assign cluster ids to splits hitting the item-level fractions.

    ``sizes`` maps cluster id -> item count (here, the number of pairs touching that
    cluster). Targets and the running counter are size-weighted (not cluster-counted) so
    skewed cluster sizes do not make the realized item ratio swing arbitrarily, and the
    largest clusters are placed first (matching core.assign_clusters_to_splits).
    """
    splits = list(fractions.keys())
    n_total = sum(sizes.values())
    targets = {s: fractions[s] * n_total for s in splits}
    counts = {s: 0 for s in splits}
    out: dict[int, str] = {}
    # Largest-first, with the seed perturbing only ties (deterministic, no global RNG).
    ordered = sorted(sizes, key=lambda c: (-sizes[c], hash((seed, c)) & 0xFFFFFFFF, c))
    for c in ordered:
        best = max(splits, key=lambda s: (targets[s] - counts[s], -splits.index(s)))
        out[c] = best
        counts[best] += sizes[c]
    return out


def leakage_across_axes(
    protein_cluster: Sequence[int],
    ligand_cluster: Sequence[int],
    assignment: Sequence[str | None],
) -> dict:
    """Verify double-cold integrity: no protein or ligand cluster shared train<->test."""
    prot_to_splits = defaultdict(set)
    lig_to_splits = defaultdict(set)
    for p, l, s in zip(protein_cluster, ligand_cluster, assignment):
        if s is None:
            continue
        prot_to_splits[p].add(s)
        lig_to_splits[l].add(s)
    prot_leaks = {p: sp for p, sp in prot_to_splits.items() if len(sp) > 1}
    lig_leaks = {l: sp for l, sp in lig_to_splits.items() if len(sp) > 1}
    return {
        "protein_axis_clean": not prot_leaks,
        "ligand_axis_clean": not lig_leaks,
        "protein_leaks": len(prot_leaks),
        "ligand_leaks": len(lig_leaks),
    }
