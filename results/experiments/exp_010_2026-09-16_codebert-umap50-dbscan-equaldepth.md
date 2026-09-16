# Experiment 010 — CodeBERT + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_010`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_002/exp_006:** embedding model only (CodeT5/e5 → CodeBERT); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **CodeBERT** (768-dim), same artifact as exp_009 | `student_embeddings/user_CodeBert_embeddings.jsonl` (67 users) |
| Students evaluated | **62** (5 ungraded excluded) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized — aligned with exp_001 | `clustering/dbscan_clustering.py` + alignment patch |
| Clustering | **DBSCAN** grid: 20 eps values from the k-distance curve × min_samples 2…9; <4-cluster configs skipped; cosine silhouette | same script; silhouette metric set to cosine |
| Grade discretization | **Equal-depth** with k = stored best_k (=7, noise counted) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_cb/db/`. Artifacts: `../models/exp_010_best_clustering.pkl`,
`../figures/exp_010_umap_dbscan.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (eps, min_samples)** | **(0.0190, 5)** |
| **Silhouette (cosine)** | **0.6479** |
| **Clusters** | **6 + 3 noise** (script stores best_k=7, counting noise); sizes `[3(noise), 4, 6, 6, 7, 15, 21]` |
| **NMI** (vs. 7 equal-depth grade bins) | **0.2085** |
| **Purity** | **0.3387** |

- Grade-bin borders (7 bins): `[29.04, 50.58, 60.05, 84.57, 95.0, 100.0, 103.33]` —
  identical to exp_002 (same k).
- NMI/purity treat noise (`-1`) as an ordinary label.

## 3. Component observations (for the conclusion)

- **Embedding (CodeBERT vs. CodeT5/e5):** silhouette jumps to 0.6479 (vs. 0.4586 /
  0.4688) — CodeBERT space is denser at fine scale — but NMI stays weak at 0.2085
  (vs. 0.2231 / 0.1354). Same moral as exp_009: density ≠ grade relevance.
- Noise fringe is thinnest here (3/62); density cores are tight under CodeBERT.

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine).

## 5. Reproduce

```bash
mkdir -p /tmp/opencode/s03run_cb/db && cd /tmp/opencode/s03run_cb/db
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# dbscan_cb.py = clustering/dbscan_clustering.py with embeddings → .../user_CodeBert_embeddings.jsonl,
#   UMAP(metric="cosine") + normalize(), silhouette metric="cosine"
# evaluate_cb.py = src/evaluation/evaluation.py with embeddings → .../user_CodeBert_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python dbscan_cb.py  # → best params, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_cb.py               # → NMI, purity
```
