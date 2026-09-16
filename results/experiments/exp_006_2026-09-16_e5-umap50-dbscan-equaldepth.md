# Experiment 006 — multilingual-e5 + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_006`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_002:** embedding model only (CodeT5 → multilingual-e5); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **multilingual-e5-base** (768-dim), same artifact as exp_005 | `student_embeddings/user_e5_embeddings.jsonl` (68 users) |
| Students evaluated | **62** (same 6 ungraded users excluded) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized — aligned with exp_001 | `clustering/dbscan_clustering.py` + alignment patch |
| Clustering | **DBSCAN** grid: 20 eps values from the k-distance curve × min_samples 2…9; <4-cluster configs skipped; cosine silhouette | same script; silhouette metric set to cosine |
| Grade discretization | **Equal-depth** with k = stored best_k (=5, noise counted) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_e5/db/`. Artifacts: `../models/exp_006_best_clustering.pkl`,
`../figures/exp_006_umap_dbscan.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (eps, min_samples)** | **(0.0206, 5)** |
| **Silhouette (cosine)** | **0.4688** |
| **Clusters** | **4 + 8 noise** (script stores best_k=5, counting noise); sizes `[8(noise), 12, 13, 14, 15]` |
| **NMI** (vs. 5 equal-depth grade bins) | **0.1354** |
| **Purity** | **0.3871** |

- Grade-bin borders (5 bins): `[40.38, 60.05, 86.17, 100.0, 103.33]`.
- NMI/purity treat noise (`-1`) as an ordinary label.

## 3. Component observations (for the conclusion)

- **Embedding (e5 vs. CodeT5/exp_002):** silhouette 0.4688 vs. 0.4586, NMI 0.1354 vs.
  0.2231, purity 0.3871 vs. 0.3387 — mixed: e5 edges silhouette/purity, CodeT5 edges
  NMI, but both NMI values are weak (≤0.22). DBSCAN remains the weakest algorithm on
  NMI under either embedding.
- **Noise heaviness** (8/62, 13%) mirrors exp_002 (7/62) — density-based grouping
  leaves a similar outlier fringe under both embeddings.
- Same k-confounding caveat as exp_002 (5 vs. 17 bins); fixed-bin variant still open.

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine).

## 5. Reproduce

```bash
mkdir -p /tmp/opencode/s03run_e5/db && cd /tmp/opencode/s03run_e5/db
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# dbscan_e5.py = clustering/dbscan_clustering.py with embeddings → .../user_e5_embeddings.jsonl,
#   UMAP(metric="cosine") + normalize(), silhouette metric="cosine"
# evaluate_e5.py = src/evaluation/evaluation.py with embeddings → .../user_e5_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python dbscan_e5.py  # → best params, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_e5.py               # → NMI, purity
```
