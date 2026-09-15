# Experiment 001 — CodeT5 + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-15
- **ID:** `exp_001`
- **Status:** partial — clustering/evaluation executed; DB-backed steps blocked (see §4)

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03, assignment `AP-Spring03-CA6-phase3` | `code_recorder` DB (unavailable at run time; see §4) |
| Embedding model | **CodeT5-base** (`Salesforce/codet5-base`, 768-dim, mean-pooled final-file-per-user) | `student_embeddings/user_CodeT5_file_embeddings.jsonl` (pre-existing artifact, 68 users) |
| Students evaluated | **62** (68 embedded − 6 without grades: `7, 142, 143, 145, 146, 148`) | `gradesheets/user_grades.csv` (141 graded users, range 0.0–103.33) |
| Dimension reduction | **UMAP, n_components=50**, `metric=cosine`, `random_state=42`, then L2-normalize | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** (sorted grades → k equal-count bins) | `src/evaluation/evaluation.py` (defaults) |
| Steps skipped | edit-distance (Step 2), distance-matrix (Step 4) | per workflow note |

Run sandbox: `/tmp/opencode/s03run/` (repo scripts + symlinked inputs, CodeT5-configured copies;
`MPLBACKEND=Agg`). Artifacts: `../models/exp_001_best_clustering.pkl`,
`../figures/exp_001_umap_kmeans_k17.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best k** | **17** |
| **Silhouette (cosine)** | **0.5179** |
| **NMI** (vs. 17 equal-depth grade bins) | **0.5356** |
| **Purity** | **0.3548** |

### Silhouette sweep (k = 4…20)

| k | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | **17** | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sil. | 0.3889 | 0.4996 | 0.4660 | **0.5094** | 0.4905 | 0.4818 | 0.4730 | 0.4776 | 0.4862 | **0.5140** | 0.5023 | 0.5005 | 0.4982 | **0.5179** | 0.4719 | 0.4919 | 0.4297 |

- Curve is flat-topped (0.47–0.52 for k = 5…17); k=17 wins but k=7 (0.5094) and k=13 (0.5140)
  are near-ties and more interpretable.
- Cluster sizes at k=17: `8, 5, 4×6, 3×4, 2×3` — balanced, no degenerate singletons.
- Grade-bin borders (17 bins): `[22.33, 27.12, 31.17, 40.38, 50.17, 52.75, 55.67,
  60.05, 66.33, 78.83, 84.93, 86.17, 94.17, 96.7, 98.92, 100.0, 103.33]`.

## 3. Component observations (for the conclusion)

- **Embedding (CodeT5):** 768-dim file-level embeddings separate students well enough for
  silhouette ≈ 0.52 after UMAP-50. Compare later vs. CodeBERT / E5 / Jina / CCBert
  runs (same UMAP+KMeans harness) to isolate model effect.
- **Dimension reduction (UMAP-50, cosine):** fixed in this experiment; ablation candidate —
  vary `n_components` (e.g. 10/50/100) and metric (cosine vs. euclidean) while holding
  model + clustering fixed.
- **Clustering (KMeans):** k-selection is weakly determined (flat sweep); decision between
  best-silhouette k=17 and interpretable k=7 should be revisited once other embeddings
  are tested. Compare later vs. DBSCAN / BIRCH / Agglomerative / OPTICS artifacts
  already in `results/figures/`.
- **Grade discretization (equal-depth):** with k=17 each bin holds ≈3.6 students, which
  mechanically depresses purity (0.35) while NMI (0.54) stays moderate. Candidate for
  conclusion: purity-vs-k sensitivity analysis, or fixed-bin-count comparison across runs.
- **NMI vs. purity gap (0.54 vs. 0.35):** suggests clusters capture grade *structure*
  better than exact bin membership — worth one paragraph in the conclusion.

## 4. Blocked steps (not executed)

| Step | Script | Outcome |
|---|---|---|
| 0 | `src/data_collection/retrieve_data.py` | `OperationalError: password authentication failed for user "hadi"` — port 5432 belongs to the LMS docker Postgres (no `code_recorder` DB); host PG15 cluster is down |
| 1 | `src/data_collection/convert_studentId_to_userId.py` | `FileNotFoundError: 'APS03_CA6_grades.csv'` (CWD-relative path bug; file is in `gradesheets/`), then same DB wall |
| 3 | `generate_emb/codeT5_inference.py` | nothing to embed without DB; `codet5-base` not in HF cache (GPU here: 2 GB MX150); HF Hub reachable so a fresh run is possible once the DB is restored |
| 6 | `src/visualization/plot_embeddings.py` | OK — `pca/tsne/umap_plot.png` regenerated from CodeT5 embeddings |

## 5. Reproduce

```bash
# from repo root; inputs via symlinks, outputs land in CWD
mkdir -p /tmp/opencode/s03run && cd /tmp/opencode/s03run
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
sed -e 's|user_e5_embeddings.jsonl|user_CodeT5_file_embeddings.jsonl|' \
  /home/hadi/thesis/clustering/kmeans_clustering.py > kmeans_codet5.py
sed -e 's|user_jina_embeddings.jsonl|user_CodeT5_file_embeddings.jsonl|' \
  /home/hadi/thesis/src/evaluation/evaluation.py > evaluate_codet5.py
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python kmeans_codet5.py   # → best k, silhouette, best_clustering.pkl
/home/hadi/thesis/venv/bin/python evaluate_codet5.py                # → NMI, purity
```
