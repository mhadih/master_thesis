# Experiment 008 — multilingual-e5 + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-16
- **ID:** `exp_008`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (same as exp_001, §4)
- **Varies vs. exp_004:** embedding model only (CodeT5 → multilingual-e5); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **multilingual-e5-base** (768-dim), same artifact as exp_005 | `student_embeddings/user_e5_embeddings.jsonl` (68 users) |
| Students evaluated | **62** (same 6 ungraded users excluded) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized — aligned with exp_001 | `clustering/optics_clustering.py` + alignment patch |
| Clustering | **OPTICS** (`cluster_method='xi'`) grid: min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2] × max_eps [inf]; all-noise or <4-cluster configs skipped; cosine silhouette | same script; silhouette metric set to cosine |
| Grade discretization | **Equal-depth** with k = stored best_k (=17, noise excluded from the count) | `src/evaluation/evaluation.py` (unchanged) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run_e5/op/`. Artifacts: `../models/exp_008_best_clustering.pkl`,
`../figures/exp_008_umap_optics.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (min_samples, xi, max_eps)** | **(2, 0.01, inf)** |
| **Silhouette (cosine)** | **0.4603** |
| **Clusters** | **17 + 4 noise** (script stores best_k=17, excluding noise); sizes `[2×8, 3×2, 4×4, 5, 6×2, 7]`, noise 4 |
| **NMI** (vs. 17 equal-depth grade bins) | **0.5485** |
| **Purity** | **0.3710** |

- Grade-bin borders (17 bins): same as exp_001/exp_005 (same k, same grades).
- NMI/purity treat noise (`-1`) as an ordinary label.

## 3. Component observations (for the conclusion)

- **Embedding (e5 vs. CodeT5/exp_004):** NMI 0.5485 vs. 0.1937 — the largest
  embedding-driven swing in the whole series, and the best NMI of all 8 experiments.
  Silhouette barely moved (0.4603 vs. 0.4663) and the winner sits at the same xi
  (0.01) with finer granularity (17 clusters + 4 noise vs. 6 + 12 noise). OPTICS-xi
  structure aligns with grades far better under e5 than CodeT5.
- **Algorithm ranking flips with the embedding:** under CodeT5, KMeans led NMI
  (0.5356); under e5, OPTICS leads (0.5485). No single "best algorithm" — report the
  interaction, not a winner.
- Purity 0.3710 ≈ exp_004's 0.3871 (both coarse/fine-bin effects as before);
  k-confounding caveat applies (17 vs. 6 bins across the pair).

## 4. Blocked steps (same as exp_001)

Steps 0/1/3 blocked (no `code_recorder` DB on this machine).

## 5. Reproduce

```bash
mkdir -p /tmp/opencode/s03run_e5/op && cd /tmp/opencode/s03run_e5/op
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# optics_e5.py = clustering/optics_clustering.py with embeddings → .../user_e5_embeddings.jsonl,
#   UMAP(metric="cosine") + normalize(), silhouette metric="cosine"
# evaluate_e5.py = src/evaluation/evaluation.py with embeddings → .../user_e5_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python optics_e5.py  # → best params, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_e5.py               # → NMI, purity
```
