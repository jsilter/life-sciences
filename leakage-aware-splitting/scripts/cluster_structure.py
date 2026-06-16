#!/usr/bin/env python3
"""Structure-modality clustering with Foldseek.

When remote homologs share a fold but not detectable sequence identity, a sequence
split leaks. Foldseek converts structures to a 3Di structural alphabet and searches
via MMseqs2, making all-vs-all structural comparison tractable (TM-align would take
millennia at that scale). Cluster structures, then assign whole structural clusters
to one split. Standard fold-equivalence threshold: TM-score >= 0.5; Foldseek also
uses E-value < 0.01.

Requires the Foldseek binary on PATH. There is no pure-Python fallback for real
structural similarity (it needs 3D coordinates), so when Foldseek is absent the
caller must either supply precomputed cluster labels or fall back to a sequence
split with an explicit warning that structural leakage is unchecked.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

PINNED_FOLDSEEK_NOTE = "pin your Foldseek version in the manifest; clustering shifts across releases"
# Foldseek version the test suite was validated against (see environment.yml).
PINNED_FOLDSEEK_VERSION = "10.941cd33"


def foldseek_available() -> bool:
    return shutil.which("foldseek") is not None


def foldseek_version() -> str | None:
    """Actual version of the foldseek binary on PATH, or None. Recorded in the
    provenance manifest so the structural split is auditable against the exact binary."""
    if not foldseek_available():
        return None
    try:
        out = subprocess.run(["foldseek", "version"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def foldseek_cluster(
    pdb_paths: Sequence[str],
    ids: Sequence[str] | None = None,
    min_seq_id: float = 0.0,
    coverage: float = 0.8,
    tmscore_threshold: float = 0.5,
    tmp_dir: str | None = None,
) -> list[int]:
    """Cluster structures with `foldseek easy-cluster`; return per-item cluster labels.

    Parameters mirror the structural-leakage recipe: cluster at TM-score >= 0.5 with
    80% coverage. `pdb_paths` are paths to .pdb/.cif files in item order; `ids`
    default to the file stems.

    Raises RuntimeError if Foldseek is missing so the caller can choose an explicit
    fallback rather than silently skipping structural leakage control.
    """
    if not foldseek_available():
        raise RuntimeError("foldseek not found on PATH; install Foldseek for structural splits")

    explicit_tmp = tmp_dir is not None
    work = Path(tmp_dir) if explicit_tmp else Path(tempfile.mkdtemp(prefix="bioevalsplit_foldseek_"))
    try:
        indir = work / "structures"
        indir.mkdir(parents=True, exist_ok=True)
        # Foldseek easy-cluster takes a directory of structures and keys results on each
        # file's stem. Stage every input under a unique, collision-proof name (a zero-padded
        # index prefix) so two paths that share a basename/stem (e.g. run1/model.pdb and
        # run2/model.pdb, or 1abc.pdb / 1abc.cif) are NOT silently overwritten or merged into
        # one cluster. The index prefix is also how we map results back, so cluster labels
        # stay strictly positional and `ids` is irrelevant to correctness.
        staged_stems = []
        for i, p in enumerate(pdb_paths):
            src = Path(p)
            dst = indir / f"{i:06d}_{src.name}"
            shutil.copyfile(src, dst)  # stream rather than read whole file into memory
            staged_stems.append(f"{i:06d}_{src.stem}")

        out_prefix = work / "clu"
        cmd = [
            "foldseek", "easy-cluster", str(indir), str(out_prefix), str(work / "tmp"),
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--tmscore-threshold", str(tmscore_threshold),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"foldseek easy-cluster failed (exit {e.returncode}): "
                f"{(e.stderr or e.stdout or '').strip()}") from e

        rep_of: dict[str, str] = {}
        with (Path(str(out_prefix) + "_cluster.tsv")).open() as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                rep, member = parts[0], parts[1]
                rep_of[member] = rep

        rep_to_label: dict[str, int] = {}
        labels = []
        for stem in staged_stems:
            rep = rep_of.get(stem, stem)
            if rep not in rep_to_label:
                rep_to_label[rep] = len(rep_to_label)
            labels.append(rep_to_label[rep])
        return labels
    finally:
        if not explicit_tmp:
            shutil.rmtree(work, ignore_errors=True)
