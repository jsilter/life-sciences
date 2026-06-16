#!/usr/bin/env python3
"""Orchestrator: turn a biological dataset into a leakage-aware train/val/test split.

This is the entrypoint the skill drives. It selects a modality strategy, produces
cluster labels, assigns whole clusters to splits via the shared core, then reports
the honest diagnostics every run must surface: realized split fractions, the
train<->test nearest-neighbour similarity distribution, and a provenance manifest.

CLI-wired modalities: `sequence` (MMseqs2 with a k-mer fallback), `small_molecule`
(RDKit scaffold), `structure` (Foldseek), `temporal` (chronological split of sequences by
date), and `metadata` (the group / metadata axis alone, for histology-style
tile/slide/patient data). The clustering modalities route through `_cluster_for_modality`.
`protein_ligand` is library-only (it needs bespoke per-pair cluster inputs); drive it via
`split_protein_ligand.py` (see references/). The group axis composes with the clustering
modalities via `group_keys`, and `--stratify-col` (a metadata column) class-balances them;
neither composes with `temporal`.

CLI:
    # sequence split
    python -m scripts.split --modality sequence --fasta data.fasta \
        --train 0.8 --val 0.1 --test 0.1 --min-seq-id 0.3 --out splits/

    # small-molecule (scaffold) split
    python -m scripts.split --modality small_molecule --smiles mols.smi --out splits/

    # structure (Foldseek) split
    python -m scripts.split --modality structure --structures pdbs/ --out splits/

    # temporal split (sequences by deposition date; flags residual train<->test homology)
    python -m scripts.split --modality temporal --fasta data.fasta \
        --metadata meta.csv --id-col id --date-col deposited --out splits/

    # group / metadata split (propose-then-confirm)
    python -m scripts.split --modality metadata --metadata histo.csv \
        --id-col tile_id --auto-detect-groups            # prints proposal, exits
    python -m scripts.split --modality metadata --metadata histo.csv \
        --id-col tile_id --group-col patient_id --out splits/   # splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from . import cluster_sequence, group_detect, leakage_metrics
from .core import assign_clusters_to_splits, clusters_from_pairs


def _hash_items(items: Sequence) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(repr(it).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def read_fasta(path: str) -> tuple[list[str], list[str]]:
    ids, seqs = [], []
    cur_id, cur = None, []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur))
                parts = line[1:].split()
                if not parts:  # a bare ">" (or ">" + only whitespace) has no identifier
                    raise ValueError(f"malformed FASTA at {path}:{lineno}: empty header '>'")
                cur_id, cur = parts[0], []
            elif line:
                cur.append(line)
    if cur_id is not None:
        ids.append(cur_id)
        seqs.append("".join(cur))
    return ids, seqs


def read_smiles(path: str) -> tuple[list[str], list[str]]:
    """Read a SMILES file for the small_molecule modality.

    One entry per line, ``SMILES [id]`` (whitespace-separated, the RDKit ``.smi``
    convention). Blank lines and lines starting with ``#`` are skipped; the id defaults to
    ``mol<line-index>`` when a line carries no second field.
    """
    ids, smis = [], []
    with open(path) as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smis.append(parts[0])
            ids.append(parts[1] if len(parts) > 1 else f"mol{n}")
    return ids, smis


def read_structures(path: str) -> tuple[list[str], list[str]]:
    """Resolve structure inputs to (ids, paths) for the structure modality.

    ``path`` is either a directory of .pdb/.cif/.ent/.mmcif files or a text file listing one
    structure path per line. ids are the file stems (the clustering itself is positional, so
    duplicate stems from different directories are handled downstream by foldseek_cluster).
    """
    p = Path(path)
    exts = {".pdb", ".cif", ".ent", ".mmcif"}
    if p.is_dir():
        paths = sorted(str(q) for q in p.iterdir() if q.suffix.lower() in exts)
    else:
        paths = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    ids = [Path(q).stem for q in paths]
    return ids, paths


def _tool_backend(name: str, actual_version, validated_version, extra: dict) -> dict:
    """Backend manifest entry recording the ACTUAL tool version, warning on drift from
    the validated one (clustering shifts across releases, so the pin is not assumed)."""
    backend = {"backend": name, "version": actual_version,
               "version_validated": validated_version, **extra}
    if actual_version and validated_version and actual_version != validated_version:
        backend["warning"] = (
            f"installed {name} {actual_version} differs from the validated "
            f"{validated_version}; clustering can shift across releases - re-validate or "
            "install the validated version (recorded so the split stays auditable)")
    return backend


def _cluster_for_modality(modality: str, ids, items, *, threshold: float, prefer_tool: bool) -> tuple[list[int], dict]:
    """Return (cluster_labels, backend_info) for a clustering-based modality.

    Handles `sequence`, `small_molecule`, and `structure`. Temporal and
    protein_ligand do not go through this path (see make_split).
    """
    if modality == "sequence":
        if prefer_tool and cluster_sequence.mmseqs_available():
            labels = cluster_sequence.mmseqs_cluster(ids, items, min_seq_id=threshold)
            return labels, _tool_backend(
                "mmseqs2", cluster_sequence.mmseqs_version(), cluster_sequence.PINNED_MMSEQS_VERSION,
                {"min_seq_id": threshold, "cluster_mode": 1})
        pairs = cluster_sequence.kmer_similar_pairs(items, threshold=threshold)
        labels = clusters_from_pairs(len(items), pairs)
        return labels, {"backend": "kmer_jaccard_fallback", "threshold": threshold, "k": 3,
                        "warning": "MMseqs2 not available; used k-mer Jaccard approximation. "
                                   "Install MMseqs2 for production splits."}

    if modality == "small_molecule":
        from . import split_small_molecule as sm
        if not sm.rdkit_available():
            raise RuntimeError("RDKit required for small_molecule modality; `pip install rdkit`")
        # Scaffold split is the baseline; Butina at the configured Tanimoto is harder.
        labels = sm.scaffold_labels(items)
        return labels, {"backend": "rdkit_bemis_murcko_scaffold", "note": sm.PINNED_RDKIT_NOTE}

    if modality == "structure":
        from . import cluster_structure as cs
        if not cs.foldseek_available():
            raise RuntimeError(
                "Foldseek required for structure modality and no pure-Python fallback exists "
                "(needs 3D coordinates). Install Foldseek, or supply precomputed cluster labels, "
                "or fall back to a sequence split and disclose that structural leakage is unchecked."
            )
        labels = cs.foldseek_cluster(items, ids=ids, tmscore_threshold=threshold)
        return labels, _tool_backend(
            "foldseek", cs.foldseek_version(), cs.PINNED_FOLDSEEK_VERSION,
            {"tmscore_threshold": threshold})

    raise NotImplementedError(f"modality '{modality}' not a clustering modality")


def _sequence_similarity(threshold_k: int = 3):
    return lambda a, b: cluster_sequence.kmer_jaccard(a, b, k=threshold_k)


def _similarity_for_modality(modality: str):
    """Return (similarity_callable, metric_name) for scoring cross-split leakage, or
    (None, None) when this modality is not scorable in pure Python here."""
    if modality == "sequence":
        return _sequence_similarity(), "kmer_jaccard_k3"
    if modality == "small_molecule":
        from . import split_small_molecule as sm
        if sm.rdkit_available():
            return sm.tanimoto, "ecfp4_tanimoto"
    return None, None


# Leakage nearest-neighbour thresholds appropriate to the SCORING metric. These are kept
# distinct from the clustering threshold (--min-seq-id): for the MMseqs2 sequence backend
# the clustering threshold is a sequence-identity value, and for the scaffold backend it is
# unused entirely, so reusing it as a Tanimoto cutoff would compare incomparable scales.
_LEAKAGE_THRESHOLD_BY_METRIC = {
    "ecfp4_tanimoto": 0.4,  # Tanimoto convention (see references/thresholds.md)
}


_METADATA_MODALITIES = ("metadata", "grouped")


def _grouping_block(group_keys, final_labels, modality, group_provenance) -> dict:
    """The honest 'group axis' report: sizes, nulls, and whether similarity glued
    distinct groups together (clusters_merged_across_groups)."""
    dist = group_detect.group_size_distribution(group_keys)
    keys_in_cluster: dict[int, set] = {}
    for lab, k in zip(final_labels, group_keys):
        if k is not None:
            keys_in_cluster.setdefault(lab, set()).add(k)
    merged = sum(1 for ks in keys_in_cluster.values() if len(ks) > 1)
    block = {
        "applied": True,
        "n_groups": dist["n_groups"],
        "group_size_distribution": {k: dist[k] for k in ("min", "max", "median", "p95", "n_singletons")},
        "n_null_group_keys": dist["n_null_group_keys"],
        "composed_with_similarity": modality not in _METADATA_MODALITIES,
        "clusters_merged_across_groups": merged,
    }
    if group_provenance:
        # Carry user-facing context (which column, who confirmed it, the proposal).
        for k in ("group_col", "id_col", "confirmed_by", "proposal"):
            if k in group_provenance:
                block[k] = group_provenance[k]
    return block


def make_split(
    ids: Sequence[str],
    items: Sequence,
    modality: str = "sequence",
    fractions: dict[str, float] | None = None,
    threshold: float = 0.3,
    seed: int = 0,
    stratify_labels: Sequence | None = None,
    prefer_tool: bool = True,
    group_keys: Sequence | None = None,
    group_provenance: dict | None = None,
    dates: Sequence | None = None,
) -> dict:
    """Produce a leakage-aware split and a full diagnostic report.

    Returns a dict with: per-item assignment, realized fractions, invariant
    reports, leakage stats (nearest-neighbour distribution), and a provenance
    manifest. The orchestrator never silently hides a failed invariant; callers
    inspect `report["invariants"]`.

    Two leakage axes, composed in a *single* union-find pass:

    * **similarity** - the per-modality clustering (sequence identity, scaffold, ...).
    * **group / metadata** - when ``group_keys`` is given (item-aligned), replicate
      units that share a source entity (patient, site, batch) are forced into one
      split. ``modality="metadata"`` runs the group axis ALONE (no similarity tool,
      no similarity leakage scored) for cases like histology where no biological
      similarity is available.

    ``modality="temporal"`` is a third, orthogonal axis: ``dates`` (item-aligned) drives a
    chronological split (train=past, test=future) instead of clustering. Items are scored
    for residual train<->test homology (a temporal cut alone does not guarantee novelty), so
    the temporal path expects sequence ``items``; the group axis does not compose with it.
    """
    if fractions is None:
        fractions = {"train": 0.8, "val": 0.1, "test": 0.1}

    # A duplicate id is itself a leakage finding: two distinct items claiming the same id
    # would otherwise be free to land in different splits (train/test contamination) while
    # every invariant still passed. Reject it here so the whole API - not just the metadata
    # CLI path - is guarded.
    if len(set(ids)) != len(ids):
        from collections import Counter
        dups = [i for i, k in Counter(ids).items() if k > 1]
        raise ValueError(
            f"duplicate id(s) in input (e.g. {dups[:5]}); a duplicate id is itself a leakage "
            "finding - resolve before splitting")

    sim = None
    sim_metric = None
    temporal_block = None

    if modality == "temporal":
        # Chronological axis: order by date instead of clustering. The group axis does not
        # compose with a date cut (a group can straddle the cutoff), so it is not accepted.
        if dates is None:
            raise ValueError("modality 'temporal' requires dates (item-aligned to items)")
        if len(dates) != len(items):
            raise ValueError("dates must align with items")
        if group_keys is not None:
            raise ValueError("the group axis does not compose with a temporal split; "
                             "run a group split or a temporal split, not both")
        from . import split_temporal as st
        assignment = st.temporal_split(dates, fractions)
        cluster_labels = list(range(len(items)))  # temporal has no similarity clusters
        n_clusters = 0
        realized_fractions = {s: (assignment.count(s) / len(items) if items else 0.0)
                              for s in fractions}
        backend = {"backend": "temporal_chronological",
                   "note": "chronological split (train=past, test=future); no similarity "
                           "clustering. A temporal cut does NOT by itself guarantee novelty - "
                           "post-cutoff items can still be near-duplicates of pre-cutoff ones, "
                           "so residual train<->test homology is scored and flagged."}
        invariants = [
            leakage_metrics.check_partition(assignment, len(items)),
            leakage_metrics.check_ratio(realized_fractions, fractions),
            leakage_metrics.check_no_empty_splits(assignment, fractions),
            leakage_metrics.check_temporal_monotonicity(assignment, dates),
        ]
        # Temporal items are sequences (see the CLI), so score residual homology with k-mer
        # Jaccard: test items that still have a near-duplicate in train get flagged so the
        # caller can additionally filter them.
        sim, sim_metric = _similarity_for_modality("sequence")
        flagged = st.residual_homology_in_temporal_split(items, assignment, sim, threshold)
        temporal_block = {
            "residual_homology_threshold": threshold,
            "n_test_with_residual_homology": len(flagged),
            "residual_homology_test_ids": [ids[i] for i in flagged],
        }
    else:
        if modality in _METADATA_MODALITIES:
            if group_keys is None:
                raise ValueError("modality 'metadata' requires group_keys (the confirmed grouping column)")
            cluster_labels = clusters_from_pairs(len(items), group_detect.pairs_from_groups(group_keys))
            backend = {"backend": "group_metadata_only",
                       "note": "group axis only; no biological-similarity tool was run and no "
                               "similarity leakage is scored (there is no sequence/structure/molecule "
                               "to compare). Group integrity is the guarantee here."}
            if group_provenance:
                backend["group_provenance"] = {k: v for k, v in group_provenance.items() if k != "proposal"}
        else:
            cluster_labels, backend = _cluster_for_modality(
                modality, ids, items, threshold=threshold, prefer_tool=prefer_tool
            )
            if group_keys is not None:
                # Compose: lift similarity labels back to edges, add group edges, re-cluster
                # ONCE. Whole-component assignment then respects both axes simultaneously.
                sim_pairs = group_detect.pairs_from_labels(cluster_labels)
                grp_pairs = group_detect.pairs_from_groups(group_keys)
                cluster_labels = clusters_from_pairs(len(items), list(sim_pairs) + grp_pairs)
                backend = {**backend, "composed_with_groups": True}
            sim, sim_metric = _similarity_for_modality(modality)

        result = assign_clusters_to_splits(cluster_labels, fractions, seed=seed, stratify_labels=stratify_labels)
        assignment = result.assignment
        realized_fractions = result.realized_fractions
        n_clusters = result.n_clusters

        invariants = [
            leakage_metrics.check_partition(assignment, len(items)),
            leakage_metrics.check_cluster_integrity(assignment, cluster_labels),
            leakage_metrics.check_ratio(realized_fractions, fractions),
            leakage_metrics.check_no_empty_splits(assignment, fractions),
        ]
        if group_keys is not None:
            invariants.append(leakage_metrics.check_group_integrity(assignment, group_keys))

        # Scaffold disjointness is a meaningful extra invariant for molecules.
        if modality == "small_molecule":
            from . import split_small_molecule as sm
            if sm.rdkit_available():
                scaffolds = [sm.bemis_murcko_scaffold(s) for s in items]
                invariants.append(leakage_metrics.check_scaffold_disjoint(scaffolds, assignment))

    # Leakage stats for any modality we can score with a similarity callable. The leakage
    # threshold tracks the SCORING metric (not the clustering threshold), and the block
    # records which metric was used and whether it matches the clustering basis, so the
    # reported pass/fail is interpretable rather than a cross-scale comparison.
    stats = None
    metric_matches_clustering = None
    if sim is not None:
        leak_threshold = _LEAKAGE_THRESHOLD_BY_METRIC.get(sim_metric, threshold)
        stats = leakage_metrics.compute_leakage_stats(items, assignment, sim, leak_threshold)
        backend_name = backend.get("backend") if isinstance(backend, dict) else None
        # k-mer Jaccard scoring only matches the clustering basis when the clustering also
        # used k-mer Jaccard (the fallback). For the MMseqs2 (identity), scaffold, and
        # temporal backends the scored metric is intentionally a different, complementary lens.
        metric_matches_clustering = (sim_metric == "kmer_jaccard_k3"
                                     and backend_name == "kmer_jaccard_fallback")

    # Channel for advisories that are NOT hard invariant failures (empty splits are now
    # the no_empty_splits invariant; this stays for future soft notices, e.g. large
    # discard fractions). Kept so the report shape is stable for callers.
    warnings_out: list[str] = []

    manifest = {
        "modality": modality,
        "n_items": len(items),
        "n_clusters": n_clusters,
        "fractions_requested": fractions,
        "fractions_realized": realized_fractions,
        "threshold": threshold,
        "seed": seed,
        "backend": backend,
        "input_hash": _hash_items(items),
        "split_hash": _hash_items(assignment),
    }

    report = {
        "assignment": assignment,
        "ids": list(ids),
        "cluster_labels": cluster_labels,
        "realized_fractions": realized_fractions,
        "invariants": [asdict(r) for r in invariants],
        "all_invariants_pass": all(bool(r) for r in invariants),
        "leakage": None if stats is None else {
            "metric": sim_metric,
            "metric_matches_clustering": metric_matches_clustering,
            "max_cross_similarity": stats.max_cross_similarity,
            "threshold": stats.threshold,
            "n_test": stats.n_test,
            "n_train": stats.n_train,
            "n_violations": stats.n_violations,
            "passes_threshold": stats.passes_threshold,
            "nn_similarity": stats.nn_similarity,
        },
        "provenance": manifest,
        "warnings": warnings_out,
    }
    if temporal_block is not None:
        report["temporal"] = temporal_block
    if group_keys is not None:
        report["grouping"] = _grouping_block(group_keys, cluster_labels, modality, group_provenance)
    return report


def _print_proposal(proposal) -> None:
    """Render a ranked grouping proposal for the user to confirm (propose-then-confirm)."""
    print("Grouping-key proposal (propose-then-confirm: nothing is split yet)\n")
    rec = proposal.recommended
    if rec is not None:
        print(f"  RECOMMENDED: --group-col {rec.column}  "
              f"({rec.n_groups} groups, sizes {rec.group_size_min}-{rec.group_size_max}, "
              f"median {rec.group_size_median:g}; canonical='{rec.canonical_key}')")
    print("\n  Ranked candidates:")
    for c in proposal.candidates:
        tag = "DISQUALIFIED" if c.disqualified else f"score={c.rank_score:.2f}"
        sat = "satisfiable" if c.satisfiable_for else "NOT satisfiable for requested ratio"
        print(f"   - {c.column:<20} {tag:<14} groups={c.n_groups:<6} {sat}")
        for w in c.warnings:
            print(f"       ! {w}")
    for note in proposal.notes:
        print(f"\n  note: {note}")
    print("\nRe-run with --group-col <column> once you have confirmed the unit of generalization.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Leakage-aware biological train/val/test split")
    # The CLI wires the clustering modalities plus temporal. protein_ligand is library-only
    # (it needs bespoke per-pair cluster inputs); drive it via split_protein_ligand.py as
    # documented in references/.
    p.add_argument("--modality", default="sequence",
                   choices=["sequence", "structure", "small_molecule", "temporal", "metadata"])
    p.add_argument("--fasta", help="FASTA input for sequence and temporal modalities")
    p.add_argument("--smiles", help="SMILES file for small_molecule modality "
                                     "('SMILES [id]' per line)")
    p.add_argument("--structures", help="structure input for structure modality (a directory "
                                         "of .pdb/.cif files, or a text file listing paths)")
    p.add_argument("--date-col", help="metadata column of per-item dates for temporal modality "
                                       "(ISO 'YYYY-MM-DD' recommended; sorts chronologically)")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--min-seq-id", type=float, default=0.3, dest="threshold",
                   help="identity/similarity threshold (sequence: MMseqs2 --min-seq-id)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="splits", help="output directory")
    # Group / metadata axis (composes with any modality; metadata = group axis only).
    p.add_argument("--metadata", help="CSV/TSV of per-item metadata for group-aware splitting")
    p.add_argument("--id-col", help="metadata column matching item ids (auto-guessed if omitted)")
    p.add_argument("--group-col", help="confirmed grouping column; no group key spans two splits")
    p.add_argument("--stratify-col", help="metadata column to stratify on (also disqualified as a group key)")
    p.add_argument("--auto-detect-groups", action="store_true",
                   help="print the ranked grouping proposal and EXIT without splitting "
                        "unless --group-col is also given (propose-then-confirm)")
    p.add_argument("--drop-null-groups", action="store_true",
                   help="drop items whose group key is null instead of treating them as singletons")
    p.add_argument("--no-tools", action="store_false", dest="prefer_tool",
                   help="force the hermetic k-mer / pure-Python fallback instead of MMseqs2/Foldseek "
                        "(deterministic; for reproducible tests and demos)")
    args = p.parse_args(argv)

    fractions = {k: v for k, v in {"train": args.train, "val": args.val, "test": args.test}.items() if v > 0}

    # ----- load items per modality -----
    aligned = None
    if args.modality == "metadata":
        if not args.metadata:
            p.error("--metadata is required for modality metadata")
        table = group_detect.load_metadata(args.metadata)
        id_col = args.id_col or group_detect.guess_id_column(table)
        if id_col is None:
            p.error("could not guess an id column; pass --id-col")
        ids = [str(v) for v in table.column(id_col)]
        if len(set(ids)) != len(ids):
            p.error(f"id column '{id_col}' has duplicate values; a duplicate id is itself a finding")
        items = list(ids)  # no biological content in metadata-only mode
        aligned = table
    elif args.modality == "sequence":
        if not args.fasta:
            p.error("--fasta is required for sequence modality")
        ids, items = read_fasta(args.fasta)
    elif args.modality == "small_molecule":
        if not args.smiles:
            p.error("--smiles is required for small_molecule modality ('SMILES [id]' per line)")
        ids, items = read_smiles(args.smiles)
    elif args.modality == "structure":
        if not args.structures:
            p.error("--structures is required for structure modality (a directory of .pdb/.cif "
                    "files, or a text file listing structure paths)")
        ids, items = read_structures(args.structures)
    elif args.modality == "temporal":
        # Sequence content + a per-item date. Sequences make the residual-homology check
        # (the temporal-cut caveat) meaningful; dates come from a --metadata --date-col.
        if not args.fasta:
            p.error("--fasta is required for temporal modality (sequences to split by date)")
        if not args.metadata or not args.date_col:
            p.error("temporal modality needs --metadata with --date-col (the per-item date)")
        ids, items = read_fasta(args.fasta)
    else:  # defensive: argparse choices should already exclude anything else
        p.error(f"modality '{args.modality}' is library-only; drive it via the Python functions "
                "in scripts/ (see references/)")
        return 2

    # A duplicate id is itself a leakage finding regardless of modality.
    if args.modality != "metadata" and len(set(ids)) != len(ids):
        from collections import Counter
        dups = [i for i, k in Counter(ids).items() if k > 1]
        p.error(f"duplicate id(s) in input (e.g. {dups[:5]}); a duplicate id is itself a finding - "
                "resolve before splitting")

    # Metadata join (group and/or stratify axes) composes with any clustering modality.
    if args.modality != "metadata" and args.metadata:
        table = group_detect.load_metadata(args.metadata)
        id_col = args.id_col or group_detect.guess_id_column(table)
        if id_col is None:
            p.error("could not guess a metadata id column; pass --id-col")
        aligned = table.align_to(ids, id_col)

    # ----- temporal axis: per-item dates (does not compose with the group/stratify axes) -----
    dates = None
    if args.modality == "temporal":
        if args.group_col or args.stratify_col or args.auto_detect_groups:
            p.error("temporal modality does not compose with the group/stratify axes "
                    "(a group or class can straddle a date cutoff); run them separately")
        dates = list(aligned.column(args.date_col))
        keep = [i for i, d in enumerate(dates) if d is not None]
        if len(keep) < len(dates):
            print(f"WARNING: dropped {len(dates) - len(keep)} item(s) with a null '{args.date_col}' "
                  "(an item with no date cannot be placed chronologically)", file=sys.stderr)
            ids = [ids[i] for i in keep]
            items = [items[i] for i in keep]
            dates = [dates[i] for i in keep]

    # ----- stratify axis (independent of grouping) -----
    stratify_labels = None
    if args.stratify_col:
        if aligned is None:
            p.error("--stratify-col requires --metadata")
        stratify_labels = list(aligned.column(args.stratify_col))

    # ----- group axis: propose-then-confirm (not for temporal, which has no group axis) -----
    group_keys = None
    group_provenance = None
    if args.modality != "temporal" and (args.modality == "metadata" or args.metadata):
        if args.auto_detect_groups and not args.group_col:
            proposal = group_detect.propose_grouping(
                aligned, ids=ids, fractions=fractions, stratify_col=args.stratify_col)
            _print_proposal(proposal)
            return 0
        if not args.group_col:
            if args.modality == "metadata":
                p.error("modality metadata needs a grouping column: run with --auto-detect-groups "
                        "to see the proposal, then re-run with --group-col <column>")
            # clustering modality + metadata but no group column and no detection requested:
            # the metadata is still used for --stratify-col (above) if given.
        else:
            keys = list(aligned.column(args.group_col))
            if args.drop_null_groups:
                keep = [i for i, k in enumerate(keys) if k is not None]
                ids = [ids[i] for i in keep]
                items = [items[i] for i in keep]
                keys = [keys[i] for i in keep]
                if stratify_labels is not None:
                    stratify_labels = [stratify_labels[i] for i in keep]
            group_keys = keys
            group_provenance = {"group_col": args.group_col,
                                "id_col": args.id_col or group_detect.guess_id_column(aligned),
                                "confirmed_by": "cli_flag"}

    # Never silently swallow a metadata file: if it was loaded but no axis consumed it,
    # say so rather than running a similarity-only split that looks group-aware. (Temporal
    # uses the metadata for --date-col, so it is exempt.)
    if (args.metadata and args.modality not in ("metadata", "temporal")
            and group_keys is None and stratify_labels is None):
        print("WARNING: --metadata was provided but neither --group-col, --stratify-col, nor "
              "--auto-detect-groups used it; running a similarity-only split and ignoring the "
              "metadata. Add --group-col <column> (or --auto-detect-groups) to use the group axis.",
              file=sys.stderr)

    report = make_split(ids, items, modality=args.modality, fractions=fractions,
                        threshold=args.threshold, seed=args.seed, prefer_tool=args.prefer_tool,
                        group_keys=group_keys, group_provenance=group_provenance,
                        stratify_labels=stratify_labels, dates=dates)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in set(report["assignment"]):
        sel = [ids[i] for i, s in enumerate(report["assignment"]) if s == split]
        (out / f"{split}.ids.txt").write_text("\n".join(sel) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2))

    pr = report["provenance"]
    print(f"modality={pr['modality']} n={pr['n_items']} clusters={pr['n_clusters']} "
          f"backend={pr['backend'].get('backend')}")
    print(f"realized fractions: {report['realized_fractions']}")
    print(f"all invariants pass: {report['all_invariants_pass']}")
    if report["leakage"]:
        lk = report["leakage"]
        print(f"max train<->test similarity: {lk['max_cross_similarity']:.3f} "
              f"(threshold {lk['threshold']}, violations {lk['n_violations']})")
    if "temporal" in report:
        t = report["temporal"]
        print(f"temporal: {t['n_test_with_residual_homology']} test item(s) still have a "
              f">={t['residual_homology_threshold']} train neighbour (residual homology to "
              "filter/disclose)")
    if "grouping" in report:
        g = report["grouping"]
        print(f"grouping: col={g.get('group_col')} n_groups={g['n_groups']} "
              f"sizes(min/med/max)={g['group_size_distribution']['min']}/"
              f"{g['group_size_distribution']['median']:g}/{g['group_size_distribution']['max']} "
              f"composed_with_similarity={g['composed_with_similarity']} "
              f"clusters_merged_across_groups={g['clusters_merged_across_groups']}")
    if isinstance(pr["backend"], dict) and pr["backend"].get("warning"):
        print(f"WARNING: {pr['backend']['warning']}", file=sys.stderr)
    for w in report.get("warnings", []):
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
