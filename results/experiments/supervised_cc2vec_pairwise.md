# Supervised file aggregation — pairwise-ranking LSTM vs. mean baseline (S03)

- **Date (UTC):** 2026-10-07
- **ID:** supervised (not an exp_0xx clustering run — different task and evaluation)
- **Status:** executed — 5-fold CV over students, 3 seeds per model
- **Varies:** aggregator only (frozen random LSTM last-hidden vs. plain mean-pool);
  CC2Vec change vectors frozen, Linear grade head trained in both arms

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Data | CC2Vec per-file change matrices, 62 graded students | `data/cc2v_file_changes.pt` (same store as exp_029–034) |
| Models | LSTM(196→16) + Dropout(0.2) + Linear(16→1) vs. mean-pool + Linear(196→1); Adam 1e-3, wd 1e-4, 20 epochs | `src/aggregation/cc2vec_supervised.py` |
| Task | Pairwise ranking: P(score_a > score_b) = σ(pred_a − pred_b), ~1.8k training pairs/fold (ties skipped) | — |
| Protocol | 5-fold CV over students (LOOCV planned; infeasible on CPU with dead GPU — same leakage rule: pairs touching held-out students never train their fold) | seeds 0, 1, 2, reported jointly |
| Metrics | Pairwise accuracy on held-out pairs + Spearman(pred, grade) pooled over folds | `supervised_cc2vec_5fold.json` |

## 2. Results

| Model | Pairwise acc (s0/s1/s2) | Mean±std | Spearman (s0/s1/s2) |
|---|---|---|---|
| LSTM aggregator | 0.592 / 0.588 / 0.562 | **0.581±0.013** | 0.020 / 0.153 / 0.086 |
| Mean-pool + head | 0.527 / 0.481 / 0.505 | 0.504±0.019 | 0.122 / 0.085 / 0.020 |

## 3. Component observations (for the conclusion)

- **Order-sensitivity buys ~8 points of ranking accuracy** (0.58 vs. chance-level
  0.50, consistent across seeds): the LSTM extracts a weak but real ordering
  signal that mean-pooling misses — the supervised counterpart to exp_033's
  compactness finding.
- **Absolute prediction is noise** (Spearman ≈ 0.08 both arms): better-than-chance
  at *who outscores whom*, useless at *by how much*.
- **Do not place these numbers beside exp_021–034 NMI.** Different task (ranking
  vs. clustering), different evaluation (held-out prediction vs. unsupervised
  agreement), and any in-sample NMI of a grade-trained aggregator would be
  circular. The valid cross-reference is qualitative only: supervised and
  unsupervised evidence agree that order carries weak signal.
- Limits: n=62, 5-fold not LOOCV (compute), single hidden size (16), fixed 20
  epochs without early stopping — treat as a pilot, not a production result.

## 4. Reproduce

```bash
/home/hadi/thesis/venv/bin/python src/aggregation/cc2vec_supervised.py
# env: ABL_STORE, ABL_HIDDEN=16, ABL_EPOCHS=20, ABL_LR, ABL_SEEDS, ABL_MAX_LEN, ABL_FOLDS
```
