# Experiment 011 — CodeBERT + UMAP-50 + BIRCH + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_011`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_003/exp_007:** embedding model only (CodeT5/e5 → CodeBERT); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **CodeBERT** (768-dim), same artifact as exp_009 | `student_embeddings/user_CodeBert_embeddings.jsonl` (67 users) |
| Students evaluated | **62** (5 ungraded excluded) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized — aligned with exp_001 | `clustering/birch_clustering.py` (post-fix defaults) |
| Clustering | **BIRCH** grid: thresholds [0.005, 0.025] step 0.0025 × n_clusters [4, 5, 6, 8, 10]; <4-cluster configs skipped; cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=4) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_cb/bi/`. Artifacts: `../models/exp_011_best_clustering.pkl`,
`../figures/exp_011_umap_birch.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (threshold, n_clusters)** | **(0.0075, 4)** |
| **Silhouette (cosine)** | **0.6828** |
| **Clusters** | **4, no noise**; sizes `[6, 12, 21, 23]` |
| **NMI** (vs. 4 equal-depth grade bins) | **0.0943** |
| **Purity** | **0.3871** |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]` — identical to exp_009 (same k).

## 3. Component observations (for the conclusion)

- **Embedding (CodeBERT vs. CodeT5/e5):** silhouette 0.6828 (vs. 0.5172 / 0.5197) with
  a coarse 4-cluster winner — yet NMI 0.0943 is the lowest of all 12 experiments.
  The recurring pattern hardens: CodeBERT yields the most compact partitions and the
  least grade-relevant ones.
- Purity 0.3871 on 4 bins is a coarseness artifact, not agreement (cf. NMI ≈ 0.09).

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine).

## 5. Reproduce

```bash
mkdir -p /tmp/opencode/s03run_cb/bi && cd /tmp/opencode/s03run_cb/bi
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# birch_cb.py = clustering/birch_clustering.py with embeddings → .../user_CodeBert_embeddings.jsonl
# evaluate_cb.py = src/evaluation/evaluation.py with embeddings → .../user_CodeBert_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python birch_cb.py  # → best params, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_cb.py              # → NMI, purity
```
