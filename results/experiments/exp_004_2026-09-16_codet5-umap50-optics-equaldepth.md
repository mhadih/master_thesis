# Experiment 004 — CodeT5 + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_004`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_001/002/003:** clustering algorithm only (KMeans → OPTICS); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **CodeT5-base** (768-dim), same artifact as exp_001–003 | `student_embeddings/user_CodeT5_file_embeddings.jsonl` (68 users) |
| Students evaluated | **62** (same 6 ungraded users excluded) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized — input space aligned with exp_001–003 | `clustering/optics_clustering.py` + alignment patch (repo default is euclidean, unnormalized; repo script already points at the CodeT5 file) |
| Clustering | **OPTICS** (`cluster_method='xi'`) grid: min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2] × max_eps [inf]; all-noise or <4-cluster configs skipped; cosine silhouette | same script; silhouette metric set to cosine to match exp_001–003 |
| Grade discretization | **Equal-depth** with k = stored best_k (=6, noise excluded from the count) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_optics/`. Artifacts: `../models/exp_004_best_clustering.pkl`,
`../figures/exp_004_umap_optics.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (min_samples, xi, max_eps)** | **(5, 0.01, inf)** |
| **Silhouette (cosine)** | **0.4663** |
| **Clusters** | **6 + 12 noise** (script stores best_k=6, excluding noise); cluster sizes `[6, 7, 8, 9, 9, 11]`, noise 12 |
| **NMI** (vs. 6 equal-depth grade bins) | **0.1937** |
| **Purity** | **0.3871** |

### Grid excerpt (cosine silhouette; skipped = all-noise or <4 clusters)

| ms | xi=0.01 | xi=0.05 | xi=0.1 | xi=0.2 |
|---|---|---|---|---|
| 2 | 0.3313 | 0.2778 | 0.1823 | −0.0586 |
| 3 | 0.2196 | 0.1862 | 0.1836 | skipped |
| 4 | 0.4189 | 0.4064 | 0.1027 | skipped |
| **5** | **0.4663** ✅ | 0.4107 | skipped | skipped |
| 6 | 0.3701 | 0.3420 | skipped | skipped |
| 7 | 0.4006 | 0.3119 | skipped | skipped |
| 8 | 0.3770 | 0.3733 | skipped | skipped |

- Larger xi (steeper density-drop requirement) collapses to <4 clusters on this data —
  only xi ∈ {0.01, 0.05} survive the ≥4-cluster filter (plus two xi=0.1 rows).
- Grade-bin borders (6 bins): `[34.66, 55.39, 78.83, 95.0, 101.17, 103.33]`.
- NMI/purity treat noise (`-1`) as an ordinary label.

## 3. Component observations (for the conclusion)

- **Clustering (OPTICS vs. rest):** silhouette 0.4663 — below KMeans (0.5179), BIRCH
  (0.5172) and DBSCAN (0.4586, near-tie). NMI 0.1937 is the lowest so far
  (0.54 / 0.31 / 0.22), but **purity 0.3871 is the highest** (0.35 / 0.29 / 0.34) —
  with only 6 coarse bins, majority-vote purity is easier. Same k-confounding caveat
  as exp_002/003: winners use different k (6 vs. 17/10/7), so keep the fixed-bin
  variant on the to-do list.
- **Noise heaviness:** 12/62 students (19%) labeled noise — highest noise rate yet
  (DBSCAN: 7; BIRCH/KMeans: 0). Xi-method OPTICS is the most conservative grouper
  here; those 12 plus DBSCAN's 7 are the outlier case-study pool (check overlap).
- **Xi sensitivity:** usable range is narrow (xi ≤ 0.05); xi ≥ 0.1 yields <4 clusters.
  One methods-section sentence, mirroring the BIRCH threshold note in exp_003.
- Unchanged from exp_001–003: CodeT5 embeddings, UMAP-50 space, equal-depth caveats.

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine; `codet5-base` not cached).
Step 6 plots were produced for exp_001 and not re-run (identical input embeddings).

## 5. Reproduce

```bash
# from repo root
mkdir -p /tmp/opencode/s03run_optics && cd /tmp/opencode/s03run_optics
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# evaluate_codet5.py = src/evaluation/evaluation.py with embeddings → .../user_CodeT5_file_embeddings.jsonl
# optics_codet5.py = clustering/optics_clustering.py with:
#   embeddings_filename → ./student_embeddings/user_CodeT5_file_embeddings.jsonl
#   grades_filename     → ./gradesheets/user_grades.csv
#   UMAP(metric="cosine") + normalize()  [align with exp_001]
#   silhouette metric="cosine"           [align with exp_001]
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python optics_codet5.py  # → best params, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_codet5.py               # → NMI, purity
```
