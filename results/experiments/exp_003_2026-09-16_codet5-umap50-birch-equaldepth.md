# Experiment 003 — CodeT5 + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_003`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_001/exp_002:** clustering algorithm only (KMeans → BIRCH); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **CodeT5-base** (768-dim), same artifact as exp_001/002 | `student_embeddings/user_CodeT5_file_embeddings.jsonl` (68 users) |
| Students evaluated | **62** (same 6 ungraded users excluded) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized — input space aligned with exp_001/002 | `clustering/birch_clustering.py` + alignment patch (repo default is euclidean, unnormalized) |
| Clustering | **BIRCH** grid: thresholds × n_clusters [4, 5, 6, 8, 10]; configs with <4 clusters skipped; cosine silhouette | same script; silhouette metric set to cosine to match exp_001/002 |
| Grade discretization | **Equal-depth** with k = stored best_k (=10) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_birch/`. Artifacts: `../models/exp_003_best_clustering.pkl`,
`../figures/exp_003_umap_birch.png`.

### Threshold-grid rescaling (deviation from repo defaults, documented)

The repo default grid `threshold = linspace(0.3, 1.5, 10)` collapses to a **single
subcluster** on the L2-normalized UMAP-50 space (all 50 configs skipped, best=None) —
it was tuned for unnormalized embeddings. A probe (`probe.py` in sandbox) on this
space gave: 0.005→54, 0.010→27, 0.015→17, 0.020→11, 0.030→2, ≥0.05→1 subclusters.
Grid rerun with `threshold = linspace(0.005, 0.025, 9)`. Threshold here is a search
range, not a modeling choice — the input space is unchanged.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (threshold, n_clusters)** | **(0.01, 10)** |
| **Silhouette (cosine)** | **0.5172** |
| **Clusters** | **10, no noise**; sizes `[2, 3, 4, 5, 6, 7, 8, 8, 9, 10]` |
| **NMI** (vs. 10 equal-depth grade bins) | **0.3073** |
| **Purity** | **0.2903** |

### Grid excerpt (cosine silhouette)

| thr. | nc=4 | nc=5 | nc=6 | nc=8 | nc=10 |
|---|---|---|---|---|---|
| 0.005 | 0.4193 | 0.4476 | 0.4417 | 0.4872 | 0.4969 |
| 0.007 | 0.4403 | 0.4400 | 0.4408 | 0.4792 | 0.5012 |
| **0.010** | 0.4570 | 0.4826 | 0.4929 | 0.4777 | **0.5172** ✅ |
| 0.013 | 0.3735 | 0.3809 | 0.4153 | 0.4528 | 0.4948 |
| 0.015 | 0.4063 | 0.4472 | 0.4374 | 0.4137 | 0.4810 |
| 0.018 | 0.3970 | 0.4435 | 0.4543 | 0.3933 | 0.4263 |
| 0.020 | 0.3823 | 0.4268 | 0.4147 | 0.4428 | 0.4237 |
| 0.023 | 0.4250 | 0.4992 | 0.4499 | 0.4484 | 0.4564 |
| 0.025 | 0.4363 | 0.3894 | 0.3894 | 0.3894 | 0.3894 |

- Grade-bin borders (10 bins): `[27.12, 40.38, 52.75, 60.05, 78.83, 86.17, 96.7,
  100.0, 101.5, 103.33]`.

## 3. Component observations (for the conclusion)

- **Clustering (BIRCH vs. KMeans/DBSCAN):** silhouette 0.5172 ≈ KMeans k=17 (0.5179) —
  a tie on cluster *compactness*, but NMI tells another story: 0.3073 (BIRCH@k=10)
  vs. 0.5356 (KMeans@k=17) vs. 0.2231 (DBSCAN@6+noise). Compactness ≠ grade
  relevance on this data. Purity is again flat across algorithms (0.29 vs. 0.35/0.34).
- **Same k-confounding caveat as exp_002:** winners use different k (10 vs. 17 vs. 7),
  so cross-algorithm NMI is not apples-to-apples; keep the fixed-bin variant on the
  to-do list for the conclusion.
- **Hyperparameter sensitivity:** BIRCH is unusable on normalized embeddings with the
  repo's default threshold range — any future normalized-space run must rescale the
  grid (§1 probe). Worth one sentence in the thesis methods section.
- **Hierarchical structure:** BIRCH's CF-tree + global clustering yields a skewed size
  profile (2…10) vs. KMeans' balanced one — the small clusters (2–4 students) are
  candidates for outlier case studies alongside DBSCAN's 7 noise students.
- Unchanged from exp_001/002: CodeT5 embeddings, UMAP-50 space, equal-depth caveats.

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine; `codet5-base` not cached).
Step 6 plots were produced for exp_001 and not re-run (identical input embeddings).

## 5. Reproduce

```bash
# from repo root
mkdir -p /tmp/opencode/s03run_birch && cd /tmp/opencode/s03run_birch
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# evaluate_codet5.py = src/evaluation/evaluation.py with embeddings → .../user_CodeT5_file_embeddings.jsonl
# birch_codet5.py = clustering/birch_clustering.py with:
#   embeddings_filename → ./student_embeddings/user_CodeT5_file_embeddings.jsonl
#   grades_filename     → ./gradesheets/user_grades.csv
#   UMAP(metric="cosine") + normalize()  [align with exp_001]
#   silhouette metric="cosine"           [align with exp_001]
#   threshold grid linspace(0.005, 0.025, 9)  [rescaled, see §1]
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python birch_codet5.py  # → best params, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_codet5.py              # → NMI, purity
```
