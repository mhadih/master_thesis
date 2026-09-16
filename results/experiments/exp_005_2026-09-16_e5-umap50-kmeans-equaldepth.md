# Experiment 005 — multilingual-e5 + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_005`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_001:** embedding model only (CodeT5 → multilingual-e5); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **multilingual-e5-base** (`intfloat/multilingual-e5-base`, 768-dim) | `student_embeddings/user_e5_embeddings.jsonl` (68 users, same user set as CodeT5) |
| Students evaluated | **62** (same 6 ungraded users excluded as exp_001) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=17) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_e5/km/`. Artifacts: `../models/exp_005_best_clustering.pkl`,
`../figures/exp_005_umap_kmeans_k17.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best k** | **17** |
| **Silhouette (cosine)** | **0.5252** |
| **NMI** (vs. 17 equal-depth grade bins) | **0.5350** |
| **Purity** | **0.3387** |

- Cluster sizes at k=17: `[2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6, 7]`.
- Grade-bin borders (17 bins): `[22.33, 27.12, 31.17, 40.38, 50.17, 52.75, 55.67,
  60.05, 66.33, 78.83, 84.93, 86.17, 94.17, 96.7, 98.92, 100.0, 103.33]` —
  identical to exp_001 (same k, same grades).

## 3. Component observations (for the conclusion)

- **Embedding (e5 vs. CodeT5/exp_001):** silhouette 0.5252 vs. 0.5179, NMI 0.5350 vs.
  0.5356, purity 0.3387 vs. 0.3548 — effectively a tie on all three metrics, with the
  same winning k=17. On this task the general-purpose multilingual-e5 matches the
  code-specific CodeT5; the embedding-model effect is negligible under KMeans.
- Unchanged from exp_001: UMAP-50 space, KMeans behavior, equal-depth caveats.

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine). e5 embeddings are a
pre-existing artifact; Step 6 plots were produced for exp_001 and not re-run.

## 5. Reproduce

```bash
mkdir -p /tmp/opencode/s03run_e5/km && cd /tmp/opencode/s03run_e5/km
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# kmeans_e5.py = clustering/kmeans_clustering.py (already points at user_e5_embeddings.jsonl)
# evaluate_e5.py = src/evaluation/evaluation.py with embeddings → .../user_e5_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python kmeans_e5.py  # → best k, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_e5.py               # → NMI, purity
```
