# Experiment 082 — CCT5 + UMAP-50 + BIRCH (k=7, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_082`
- **Status:** executed — K=7 counterpart of exp_061/068/075; see methods note below
- **Varies vs. exp_075:** n_clusters [6] → [7] (threshold sweep kept)

## Setup

CCT5 diff embeddings (66 × 768-dim, Kaggle run), 62 students, UMAP-50 cosine +
L2-normalize, BIRCH thresholds [0.005, 0.025] step 0.0025 × nc=[7], cosine
silhouette, equal-depth bins. Sandbox: `/tmp/opencode/s03run_fx7/i82/`.

Artifacts: `../models/exp_082_best_clustering.pkl`, `../figures/exp_082_umap_birch_k7.png`.

## 2. Results — and a methods finding

Winner: (thr=0.0225, nc=7), sil 0.5347 — but only **6 unique labels**: with ≤7
CF-subclusters, Birch skips the global agglomerative step and returns subcluster
labels directly, so `n_clusters` is an **upper bound**, not an exact count.
Verification: ARI(i82 labels, exp_075 labels) = **1.0** (identical partition up
to label permutation) → NMI 0.2038, purity 0.3065, 6 bins — this run reproduces
exp_075 exactly.

| k (requested → actual) | NMI | Purity | Source |
|---|---|---|---|
| 4 | 0.1197 | 0.4194 | exp_061 |
| 5 | 0.1332 | 0.3548 | exp_068 |
| 6 | 0.2038 | 0.3065 | exp_075 |
| 7 → **6** | 0.2038 | 0.3065 | exp_082 (= exp_075) |

## Reading

Two lessons: (1) for the k-effect table, CCT5's k=7 cell equals its k=6 cell —
monotone climb stands, no new information; (2) methodologically, BIRCH nc must
be validated by counting *actual* unique labels (as all fixed-K runs in this
thesis do when storing best_k), never trusted from the parameter alone. Same
upper-bound behavior would bite any future nc larger than the subcluster count.
