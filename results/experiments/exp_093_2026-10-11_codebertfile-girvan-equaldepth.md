# Experiment 093 — CodeBERT whole-file distance + Girvan–Newman + equal-depth grades (S03)

- **Date (UTC):** 2026-10-11
- **ID:** `exp_093`
- **Status:** executed — first Girvan–Newman run in the series
- **Varies vs. all prior exps:** clustering algorithm only (community detection on
  the student distance graph instead of UMAP-space clustering); embeddings are
  the max-NMI CodeBERT whole-file vectors (exp_038 setup)

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CodeBERT whole-file** (68 × 768-dim), 62 graded students | `student_embeddings/user_CodeBert_file_embeddings.jsonl` (rewrote `src/distance_analysis/calculate_distance.py` to this file) |
| Distance | cosine → [0,1] → dataset-normalized matrix + heatmap | sandbox `/tmp/opencode/s03run_girvan/` (repo script + symlinks; seaborn installed for the heatmap) |
| Clustering | **Girvan–Newman** on the complete weighted student graph (weight = distance), k sweep 4…20, best precomputed-distance silhouette | `clustering/girvan_clustering.py` + pickle dump of the winner |
| Eval | equal-depth, bins = winner's k (=19), NMI + purity | same `src/evaluation/evaluation.py` as before |

Artifacts: `../models/exp_093_best_clustering.pkl`, `../figures/exp_093_distance_heatmap.png`.

## 2. Results

| Metric | exp_093 (Girvan–Newman) | exp_038 (OPTICS, same embeddings) |
|---|---|---|
| Best k | **19** | 20 |
| Silhouette (precomputed) | **−0.3796** | 0.3903 |
| Partition | **18 singletons + 1 mega-cluster of 44** | 20 mixed groups |
| NMI (19 bins) | **0.4235** | 0.6641 |
| Purity | 0.3871 | 0.4194 |

## 3. Component observations (for the conclusion)

- **Degenerate winner:** negative silhouette at every k (least-bad −0.38 at k=19);
  Girvan–Newman on a *complete distance-weighted* graph peels singletons instead
  of splitting communities — edge-betweenness on dense distance weights isolates
  outliers first. The algorithm is mismatched to this graph construction, not to
  the data.
- **NMI 0.42/Purity 0.39 are inflated by degeneracy:** 18 singleton "clusters"
  are trivially pure and nearly self-matching against 19 bins. Do not rank this
  number alongside exp_001–092 — it measures the metric's failure mode, which is
  itself the finding: NMI/purity without a silhouette/degeneracy check mislead.
- **Fix for a follow-up (not run):** sparsify first (k-NN graph on the distance
  matrix) so betweenness can find bottlenecks instead of peeling leaves; then
  Girvan–Newman becomes a legitimate community baseline.

## 4. Reproduce

```bash
mkdir -p /tmp/opencode/s03run_girvan && cd /tmp/opencode/s03run_girvan
ln -sfn /home/hadi/thesis/student_embeddings/user_CodeBert_file_embeddings.jsonl user_CodeBert_file_embeddings.jsonl
ln -sfn /home/hadi/thesis/gradesheets/user_grades.csv user_grades.csv
cp /home/hadi/thesis/src/distance_analysis/calculate_distance.py .
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python calculate_distance.py  # needs seaborn
# girvan_clustering.py + pickle dump, then standard evaluation (see §1)
```
