# Protein-ligand (drug-target) splitting: control both axes

## Why one axis is not enough

Under a random split, deep DTI models reach AUROC > 0.98 - but this is hidden ligand
bias: correct predictions from drug features alone, not from interaction patterns. The
honest split must keep leakage low on **both** the protein side and the ligand side.

PDBbind -> CASF is the canonical leak: >700 training complexes share near-duplicate
similarity with a CASF complex, involving ~45% of CASF test complexes; models score
well even with all protein or all ligand information removed, proving memorization.
De-leaking (CleanSplit: remove train complexes with protein TM-score > 0.8, ligand
Tanimoto > 0.9, or similar poses) drops top models toward the level of trivial search
baselines.

## Split types

| Split | Test pairs have... | Tests generalization to... |
|-------|--------------------|----------------------------|
| Warm / random | seen proteins and seen ligands | nothing (memorization) |
| Cold-drug | unseen ligands, seen proteins | new chemistry on known targets |
| Cold-target | unseen proteins, seen ligands | known chemistry on new targets |
| **Double-cold (cold-pair)** | unseen proteins AND unseen ligands | genuinely new pairs (the honest estimate) |

## Recipe

1. Cluster proteins (sequence: `cluster_sequence`; or structure: `cluster_structure`)
   and ligands (scaffold/Tanimoto: `split_small_molecule`) **separately**.
2. Run `scripts/split_protein_ligand.py:double_cold_split` with the per-pair protein and
   ligand cluster labels. It partitions protein clusters and ligand clusters
   independently and keeps only pairs whose protein and ligand land in the same split;
   off-diagonal pairs are **discarded** (keeping them would leak one axis). It reports
   `discarded_fraction` so you disclose how much data was dropped.
3. Verify with `leakage_across_axes` that no protein or ligand cluster spans train/test.

The greedy version discards more data than necessary. **DataSAIL** formulates the 2D
assignment as a combinatorial optimization that minimizes leakage while retaining more
data - prefer it for production. PLINDER and LP-PDBBind provide multi-similarity
protein+ligand splits and are good golden references.

## A subtlety on negatives

A non-reported protein-ligand pair is *untested*, not a confirmed negative. Naive
negative sampling bakes in assumptions; state them explicitly. This splitter only
partitions the pairs you provide.

## Task-dependent leakage (PLINDER)

For rigid docking, a shared protein is not leakage if the pocket/conformation differs;
for co-folding it is. Let the user declare the task and choose the similarity axes
accordingly.

## Citations

- Bai, P., Miljković, F., John, B. & Lu, H. (2023). Interpretable bilinear attention
  network with domain adaptation improves drug-target prediction. *Nature Machine
  Intelligence*, 5(2), 126-136. https://doi.org/10.1038/s42256-022-00605-1. Documents
  hidden ligand bias and motivates cold-pair (cold-drug + cold-target) evaluation.
- Graber, D., Stockinger, P., Meyer, F., Mishra, S., Horn, C. & Buller, R. (2025).
  Resolving data bias improves generalization in binding affinity prediction.
  *Nature Machine Intelligence*, 7(10), 1713-1725. https://doi.org/10.1038/s42256-025-01124-5.
  Introduces the PDBbind "CleanSplit" benchmark (re-splitting PDBbind to remove train/test
  similarity leakage), the reference approach used here for protein-ligand de-leaking.
- Joeres, R., Blumenthal, D.B. & Kalinina, O.V. Data splitting to avoid information leakage
  with DataSAIL. *Nature Communications* (2025). https://doi.org/10.1038/s41467-025-58606-8.
  Formulates train/test assignment as an optimization - the production tool for 2D
  cold-drug + cold-target splits (cf. PLINDER, LP-PDBBind).
