# Experiment 012 — CodeBERT + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_012`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_004/exp_008:** embedding model only (CodeT5/e5 → CodeBERT); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **CodeBERT** (768-dim), same artifact as exp_009 | `student_embeddings/user_CodeBert_embeddings.jsonl` (67 users) |
| Students evaluated | **62** (5 ungraded excluded) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized — aligned with exp_001 | `clustering/optics_clustering.py` + alignment patch |
| Clustering | **OPTICS** (`cluster_method='xi'`) grid: min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2] × max_eps [inf]; all-noise or <4-cluster configs skipped; cosine silhouette | same script; silhouette metric set to cosine |
| Grade discretization | **Equal-depth** with k = stored best_k (=3, noise counted) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_cb/op/`. Artifacts: `../models/exp_012_best_clustering.pkl`,
`../figures/exp_012_umap_optics.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (min_samples, xi, max_eps)** | **(5, 0.2, inf)** |
| **Silhouette (cosine)** | **0.5728** |
| **Clusters** | **2 + 28 noise** (script stores best_k=3, counting noise); sizes `[6, 7, 21, 28(noise)]` — 45% of students labeled noise |
| **NMI** (vs. 3 equal-depth grade bins) | **0.1539** |
| **Purity** | **0.5000** |

- Grade-bin borders (3 bins): `[55.39, 95.0, 103.33]`.
- NMI/purity treat noise (`-1`) as an ordinary label. Purity 0.50 on 3 bins with a
  28-student noise "cluster" is a coarseness artifact, not agreement.
- Unlike exp_004/008 (winners at xi=0.01/0.02-scale), the winner here needs xi=0.2 —
  CodeBERT space requires a much steeper density drop to cut clusters.

## 3. Component observations (for the conclusion)

- **Embedding (CodeBERT vs. CodeT5/e5):** NMI 0.1539 sits between exp_004 (0.1937) and
  exp_008 (0.5485) — OPTICS is the most embedding-sensitive algorithm in the series
  (NMI range 0.15–0.55 across embeddings, vs. ≤0.03 for KMeans/BIRCH).
- **Degenerate grouping:** 2 real clusters + 45% noise means OPTICS-xi finds almost no
  grade-meaningful density structure under CodeBERT; report as a negative result.
- Same k-confounding caveat (3 vs. 6/17 bins); fixed-bin variant still open.

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine).

## 5. Reproduce

```bash
mkdir -p /tmp/opencode/s03run_cb/op && cd /tmp/opencode/s03run_cb/op
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# optics_cb.py = clustering/optics_clustering.py with embeddings → .../user_CodeBert_embeddings.jsonl,
#   UMAP(metric="cosine") + normalize(), silhouette metric="cosine"
# evaluate_cb.py = src/evaluation/evaluation.py with embeddings → .../user_CodeBert_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python optics_cb.py  # → best params, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_cb.py               # → NMI, purity
```
