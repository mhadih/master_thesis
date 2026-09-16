# Experiment 007 — multilingual-e5 + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_007`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_003:** embedding model only (CodeT5 → multilingual-e5); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **multilingual-e5-base** (768-dim), same artifact as exp_005 | `student_embeddings/user_e5_embeddings.jsonl` (68 users) |
| Students evaluated | **62** (same 6 ungraded users excluded) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized — aligned with exp_001 | `clustering/birch_clustering.py` (post-fix defaults: same space + threshold grid) |
| Clustering | **BIRCH** grid: thresholds [0.005, 0.025] step 0.0025 × n_clusters [4, 5, 6, 8, 10]; <4-cluster configs skipped; cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=10) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_e5/bi/`. Artifacts: `../models/exp_007_best_clustering.pkl`,
`../figures/exp_007_umap_birch.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (threshold, n_clusters)** | **(0.0075, 10)** |
| **Silhouette (cosine)** | **0.5197** |
| **Clusters** | **10, no noise**; sizes `[4, 4, 6, 6, 6, 6, 7, 7, 7, 9]` |
| **NMI** (vs. 10 equal-depth grade bins) | **0.3273** |
| **Purity** | **0.3226** |

- Grade-bin borders (10 bins): `[27.12, 40.38, 52.75, 60.05, 78.83, 86.17, 96.7,
  100.0, 101.5, 103.33]` — identical to exp_003 (same k, same grades).

## 3. Component observations (for the conclusion)

- **Embedding (e5 vs. CodeT5/exp_003):** silhouette 0.5197 vs. 0.5172, NMI 0.3273 vs.
  0.3073, purity 0.3226 vs. 0.2903 — e5 ahead on all three, but gaps are small
  (≤0.033). Same winning n_clusters=10 at a nearby threshold (0.0075 vs. 0.01):
  BIRCH behaves nearly identically under both embeddings.
- Same k-confounding caveat as exp_003 (10 vs. 17 bins); fixed-bin variant still open.

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine).

## 5. Reproduce

```bash
mkdir -p /tmp/opencode/s03run_e5/bi && cd /tmp/opencode/s03run_e5/bi
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# birch_e5.py = clustering/birch_clustering.py with embeddings → .../user_e5_embeddings.jsonl
# evaluate_e5.py = src/evaluation/evaluation.py with embeddings → .../user_e5_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python birch_e5.py  # → best params, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_e5.py              # → NMI, purity
```
