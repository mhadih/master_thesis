# Experiment 002 — CodeT5 + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-15
- **ID:** `exp_002`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_001:** clustering algorithm only (KMeans → DBSCAN); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **CodeT5-base** (768-dim), same artifact as exp_001 | `student_embeddings/user_CodeT5_file_embeddings.jsonl` (68 users) |
| Students evaluated | **62** (same 6 ungraded users excluded as exp_001) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized — input space aligned with exp_001 | `clustering/dbscan_clustering.py` + alignment patch (repo default is euclidean, unnormalized) |
| Clustering | **DBSCAN** grid: 20 eps values from the k-distance curve (min_samples=4) × min_samples 2…9; configs with <4 clusters skipped; cosine silhouette | same script; silhouette metric set to cosine to match exp_001 |
| Grade discretization | **Equal-depth** with k = stored best_k | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_dbscan/`. Artifacts: `../models/exp_002_best_clustering.pkl`,
`../figures/exp_002_umap_dbscan.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (eps, min_samples)** | **(0.0230, 7)** |
| **Silhouette (cosine)** | **0.4586** |
| **Clusters** | **6 + 7 noise points** (script stores best_k=7, counting noise as a cluster) |
| **NMI** (vs. 7 equal-depth grade bins) | **0.2231** |
| **Purity** | **0.3387** |

### Grid excerpt (cosine silhouette; <4-cluster configs skipped by the script)

| eps | ms=2 | ms=3 | ms=4 | ms=5 | ms=6 | ms=7 | ms=8 | ms=9 |
|---|---|---|---|---|---|---|---|---|
| 0.019 | 0.2199 | 0.2950 | 0.3966 | 0.3148 | 0.2008 | — | — | — |
| 0.020 | 0.1800 | 0.2920 | 0.4121 | 0.3962 | 0.3409 | — | — | — |
| 0.021 | — | — | 0.1162 | 0.4235 | 0.4049 | 0.1717 | — | — |
| 0.022 | — | — | 0.1162 | 0.2754 | 0.4403 | 0.3206 | — | — |
| **0.023** | — | — | — | 0.2375 | 0.2715 | **0.4586** ✅ | 0.2462 | — |
| 0.024 | — | — | — | — | 0.0499 | 0.3990 | 0.4206 | — |
| 0.025 | — | — | — | — | — | 0.2578 | 0.4112 | 0.3718 |
| 0.026 | — | — | — | — | — | — | — | 0.1935 |

(— = skipped: all-noise / single cluster / <4 clusters. Full log in sandbox.)

- Winning partition sizes: `12, 9, 9, 9, 8, 8` + 7 noise (`-1`).
- Grade-bin borders (7 bins): `[29.04, 50.58, 60.05, 84.57, 95.0, 100.0, 103.33]`.
- NMI/purity treat noise (`-1`) as an ordinary label.

## 3. Component observations (for the conclusion)

- **Clustering (DBSCAN vs. KMeans/exp_001):** silhouette 0.4586 < 0.5179 (KMeans k=17);
  NMI 0.2231 ≪ 0.5356, while purity is a wash (0.3387 vs. 0.3548). Caveat for a fair
  comparison: the two winners use different k (6+noise vs. 17), so NMI is not
  apples-to-apples — the k=7 KMeans runner-up (silhouette 0.5094) is the fairer
  baseline for DBSCAN. Re-evaluate KMeans@k=7 NMI/purity before concluding.
- **Density structure:** DBSCAN finds 6 uneven clusters plus 7 noise students on the
  same UMAP-50 space where KMeans sees ~17 fine groups — suggests the embedding space
  has a few dense cores with sparsely scattered students rather than cleanly
  separated fine clusters. The 7 noise students are worth case-studying (outlier
  editing trajectories?).
- **Discretization coupling:** evaluation bins = stored best_k, so the algorithm's
  cluster count dictates the grade resolution (7 bins here vs. 17 in exp_001) — this
  confounds cross-algorithm NMI/purity comparison. Consider a fixed-bin variant
  (e.g. always 7 bins) for the conclusion.
- Unchanged from exp_001: CodeT5 embeddings, UMAP-50 input space, equal-depth caveats
  (see exp_001 §3).

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine; `codet5-base` not cached).
Step 6 plots were produced for exp_001 and not re-run (identical input embeddings).

## 5. Reproduce

```bash
# from repo root
mkdir -p /tmp/opencode/s03run_dbscan && cd /tmp/opencode/s03run_dbscan
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
cp /tmp/opencode/s03run/evaluate_codet5.py .
# dbscan_codet5.py = clustering/dbscan_clustering.py with:
#   embeddings_filename → ./student_embeddings/user_CodeT5_file_embeddings.jsonl
#   grades_filename     → ./gradesheets/user_grades.csv
#   UMAP(metric="cosine") + normalize()  [align with exp_001]
#   silhouette metric="cosine"           [align with exp_001]
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python dbscan_codet5.py  # → best params, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_codet5.py               # → NMI, purity
```
