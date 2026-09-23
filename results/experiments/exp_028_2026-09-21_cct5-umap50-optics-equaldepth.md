# Experiment 028 — CCT5 + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-21
- **ID:** `exp_028`
- **Status:** partial — Kaggle-generated CCT5 embeddings (see exp_025 §1)
- **Varies vs. exp_004/008/012/016/020/024:** embedding model only (→ CCT5); all else identical

## 1. Setup

Same embeddings (66 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_025. Clustering: **OPTICS** (`cluster_method='xi'`) grid
(min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2], cosine silhouette) via
`clustering/optics_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_cct5/op/`. Artifacts: `../models/exp_028_best_clustering.pkl`,
`../figures/exp_028_umap_optics.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (min_samples, xi, max_eps)** | **(9, 0.01, inf)** |
| **Silhouette (cosine)** | **0.4631** |
| **Clusters** | **3 + 18 noise** (script stores best_k=3, counting noise); sizes `[9, 12, 18(noise), 23]` — 29% noise |
| **NMI** (vs. 3 equal-depth grade bins) | **0.0551** |
| **Purity** | **0.4839** |

- Grade-bin borders (3 bins): `[55.39, 95.0, 103.33]` — identical to exp_012/020 (same k).
- NMI/purity treat noise as an ordinary label; purity 0.48 on 3 bins is a coarseness
  artifact (cf. NMI ≈ 0.06), same as exp_012/020.

## 3. Component observations (for the conclusion)

- NMI 0.0551 ≈ floor (with CCBERT 0.09–0.16, CC2Vec 0.02–0.11): all three
  change-oriented models agree — change input alone does not yield grade structure.
- OPTICS stays the most embedding-sensitive algorithm (NMI 0.04–0.55); CCT5 sits at
  its low end with CodeBERT/CCBERT/CC2Vec, opposite e5 (0.55).

## 4.–5. Blocked steps / reproduce

As exp_025 (replace `km/kmeans_cc5.py` with `op/optics_cc5.py`).
