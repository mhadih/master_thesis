# Supervised LSTM aggregation — temporal-order shuffle control (S03)

- **Date (UTC):** 2026-10-17
- **ID:** `shuffle-control` (follow-up to `supervised_cc2vec_pairwise.md`)
- **Status:** executed — 3 order modes × 3 seeds, CPU (determinism); 5-fold CV
- **Varies:** only the order of change vectors inside each file. Everything else
  (LSTM 196→16, dropout 0.2, head, pairwise-ranking loss, folds, seeds) identical
  to `src/aggregation/cc2vec_supervised.py`.

## 1. Motivation

The supervised ranking result gave the LSTM aggregator ~8 points over mean-pooling
(0.581 vs. 0.504). That gap is ambiguous: it could reflect genuine *temporal*
modelling of code development, or merely the extra nonlinearity/parameters of an
LSTM regardless of order. This control separates the two by destroying the
temporal structure while keeping the exact multiset of change vectors.

| Mode | What changes | What is held fixed |
|---|---|---|
| `chrono` | nothing (original order) | — |
| `shuffled` | random permutation of the last-200 changes, fixed per (file, seed) | same vectors, same file length, same truncation |
| `reversed` | chronological flip | same vectors, adjacency preserved, direction reversed |

Implementation: order is transformed *after* truncation to last `MAX_LEN`, so the
two arms differ only in order, never in which changes are present. Sanity check:
991/1058 files changed under shuffle, 1035/1058 under reverse (the rest have <2
changes). The mean-pool arm is permutation-invariant by construction and is
therefore not rerun — it is 0.504 in every mode.

## 2. Results (pairwise accuracy on held-out pairs, ~3064 pairs/seed)

| Mode | acc s0/s1/s2 | Mean ± std | Spearman (mean) |
|---|---|---|---|
| `chrono` (original) | 0.592 / 0.588 / 0.562 | **0.581 ± 0.013** | 0.087 |
| `shuffled` (order destroyed) | 0.592 / 0.589 / 0.564 | **0.582 ± 0.013** | 0.087 |
| `reversed` (order reversed) | 0.593 / 0.588 / 0.558 | **0.580 ± 0.015** | 0.086 |

`chrono` reproduces `supervised_cc2vec_5fold.json` **bit-identically**
(0.592037/0.020020, 0.588·/0.153·, 0.562·/0.086·), confirming the only variable
is order. Artifacts: `supervised_cc2vec_shuffle_control.json`.

## 3. Descriptive analysis (for the thesis)

**The LSTM's advantage is order-invariant.** Shuffling the temporal sequence
changes pairwise accuracy by ≤0.001 on every seed — far inside the seed-to-seed
noise (±0.013). Reversing it likewise. The three modes are statistically
indistinguishable, so the ~8-point edge over mean-pooling cannot be attributed
to modelling code evolution over time.

**What the advantage actually is.** The gain is real and seed-consistent
(0.58 vs. 0.50, ~9 binomial SE on 3064 pairs, though pairs are correlated across
shared students so the effective margin is smaller). But its source is the
*nonlinear, parameterized* file encoder — an LSTM applied to an unordered set of
change vectors — not temporal reasoning. The recurrent architecture buys a
better pooling function than the arithmetic mean; the recurrence itself is idle.

**Practical implication.** The honest title for this component is "learned
nonlinear aggregation," not "temporal aggregation." For the thesis conclusion,
this is a negative result about the *method's claim*, and reporting it strengthens
credibility: the model is competitive, but the temporal framing is not supported
by the control. If temporal modelling were the goal, it would need to be forced
(e.g. position/recency-encoded inputs, an explicit timestamp feature, or an
order-sensitivity penalty), because a standard LSTM trained on pairwise grade
ranking does not exploit order even when the signal is available.

**Why order isn't exploited here (hypothesis).** With only 62 students, a 16-d
hidden state and 20 epochs, the ranking objective is satisfied by a nearly
order-invariant function; nothing in the LOS pushes it to read the final hidden
state as "latest changes." The frozen-LSTM ablation (exp_033/034) showed the same
theme from the other side: order-aware recurrence maximized *compactness* while
leaving grade signal unchanged.

**Limits.** Single hidden size, fixed 20 epochs, one task (pairwise ranking), n=62;
"order-invariant" is asserted at these hyper-parameters, not universally. A larger
model or loss that rewards order-sensitivity could behave differently. Pairs are
exchangeable within a fold, so the reported SE is optimistic; the null, however,
is robust across all three seeds and both perturbations.

## 4. Reproduce

```bash
./venv/bin/python src/aggregation/cc2vec_shuffle_control.py            # all modes
ABL_MODES=shuffled ABL_SEEDS=0 ./venv/bin/python src/aggregation/cc2vec_shuffle_control.py
# env: ABL_STORE, ABL_MODES, ABL_SEEDS, ABL_HIDDEN, ABL_EPOCHS, ABL_LR,
#      ABL_FOLDS, ABL_MAX_LEN, ABL_DEVICE(cpu), ABL_OUT
```
