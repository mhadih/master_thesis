"""Supervised file aggregator: pairwise-ranking LSTM over CC2Vec change vectors.

Protocol A (prediction, no clustering): 5-fold CV over students (LOOCV was
planned but is infeasible on CPU with the dead GPU; same leakage rule).
For each held-out fold, train on ordered pairs (i, j) drawn only from the
other 4 folds (label 1 iff grade_i > grade_j, ties skipped), then score
held-out pairs (exactly one endpoint in the fold). Reports pairwise accuracy
on held-out pairs + Spearman(pred, grade) pooled over folds. Compares LSTM
aggregation vs. a mean-pool + linear baseline trained identically.

Method: per-file change matrices -> file embeddings (LSTM last-hidden with
dropout, or plain mean) -> length-weighted mean -> student vector -> Linear
-> scalar score. CC2Vec change vectors stay frozen; only aggregator + head train.

Leakage rule: pairs touching held-out students never appear in their training
folds. NMI-against-grades on trained students would be circular and is NOT
computed here (see exp_021-034 for the unsupervised NMI series).

Env: ABL_STORE (default data/cc2v_file_changes.pt), HIDDEN (default 16),
EPOCHS (default 20), LR (default 1e-3), SEEDS (default "0,1,2"), MAX_LEN (200),
FOLDS (default 5).

Writes results/experiments/supervised_cc2vec_5fold.json (metrics + predictions).
"""

import json
import os
from collections import defaultdict

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.environ.get("ABL_STORE", os.path.join(REPO, "data", "cc2v_file_changes.pt"))
GRADES = os.path.join(REPO, "gradesheets", "user_grades.csv")
HIDDEN = int(os.environ.get("ABL_HIDDEN", "16"))
EPOCHS = int(os.environ.get("ABL_EPOCHS", "20"))
LR = float(os.environ.get("ABL_LR", "1e-3"))
FOLDS = int(os.environ.get("ABL_FOLDS", "5"))
SEEDS = [int(s) for s in os.environ.get("ABL_SEEDS", "0,1,2").split(",") if s.strip() != ""]
MAX_LEN = int(os.environ.get("ABL_MAX_LEN", "200"))
OUT = os.path.join(REPO, "results", "experiments", "supervised_cc2vec_5fold.json")


class Aggregator(nn.Module):
    """LSTM file encoder (or mean-pool baseline) + linear grade head."""

    def __init__(self, use_lstm=True, dropout=0.2):
        super().__init__()
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(input_size=196, hidden_size=HIDDEN, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(HIDDEN if use_lstm else 196, 1)

    def file_emb(self, vecs):
        if len(vecs) > MAX_LEN:
            vecs = vecs[-MAX_LEN:]
        if self.use_lstm:
            h, _ = self.lstm(vecs.unsqueeze(0))
            return h.squeeze(0)[-1]
        return vecs.mean(dim=0)

    def student_score(self, files, flens, train=True):
        embs = torch.stack([self.file_emb(v) for v in files])
        if train:
            embs = self.drop(embs)
        w = torch.tensor(flens, dtype=torch.float32, device=embs.device)
        stu = (embs * (w / w.sum()).unsqueeze(1)).sum(dim=0)
        return self.head(self.drop(stu) if train else stu).squeeze(-1)


def load_data(device):
    import csv

    store = torch.load(STORE, map_location="cpu", weights_only=False)
    grades = {}
    with open(GRADES) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            grades[int(row[0])] = float(row[1])
    per_user = defaultdict(list)  # uid -> [(file_vecs, flen)]
    for (uid, fn), d in store.items():
        if int(uid) not in grades:
            continue
        per_user[int(uid)].append((d["vecs"].float().to(device), d["flen"]))
    uids = sorted(per_user)
    print("graded users with files:", len(uids), flush=True)
    return per_user, grades, uids


def student_scores(model, per_user, order, device, train=True):
    out = []
    for u in order.tolist():
        feats = per_user[int(u)]
        out.append(model.student_score(
            [v for v, _ in feats], [l for _, l in feats], train=train))
    return torch.stack(out)


def run_seed(seed, per_user, grades, uids, device):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(uids), generator=rng).tolist()
    folds = [sorted(uids[i] for i in perm[f::FOLDS]) for f in range(FOLDS)]
    results = {}
    for use_lstm in (True, False):
        torch.manual_seed(seed)
        held_preds, held_ok, held_n = {}, 0, 0
        for fold in folds:
            test = set(fold)
            train = [u for u in uids if u not in test]
            order = torch.tensor(train)
            g = torch.tensor([grades[u] for u in train])
            model = Aggregator(use_lstm=use_lstm).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            model.train()
            for _ in range(EPOCHS):
                scores = student_scores(model, per_user, order, device, train=True)
                i, j = torch.randint(len(train), (2, 512))
                a, b = order[i], order[j]
                keep = g[i] != g[j]
                if not keep.any():
                    continue
                ya = (g[i[keep]] > g[j[keep]]).float().to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    scores[i[keep]] - scores[j[keep]], ya)
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                full = torch.tensor(uids)
                s2s = dict(zip(uids, student_scores(
                    model, per_user, full, device, train=False).tolist()))
            for s in fold:
                held_preds[s] = s2s[s]
                for t in uids:
                    if t in test or grades[s] == grades[t]:
                        continue
                    held_n += 1
                    if (s2s[s] > s2s[t]) == (grades[s] > grades[t]):
                        held_ok += 1
        key = "lstm" if use_lstm else "mean"
        results[key] = {"preds": held_preds, "pair_acc": held_ok / held_n,
                        "pairs": held_n}
    return results


def spearman(xs, ys):
    import statistics as _st  # noqa
    rx = {v: i for i, v in enumerate(sorted(set(xs)))}
    ry = {v: i for i, v in enumerate(sorted(set(ys)))}
    n = len(xs)
    d2 = sum((rx[x] - ry[y]) ** 2 for x, y in zip(xs, ys))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_user, grades, uids = load_data(device)
    out = {"seeds": {}, "config": {"hidden": HIDDEN, "epochs": EPOCHS, "lr": LR,
                                   "n_users": len(uids), "device": str(device)}}
    for seed in SEEDS:
        r = run_seed(seed, per_user, grades, uids, device)
        for key in ("lstm", "mean"):
            pr = r[key]["preds"]
            xs = [pr[u] for u in uids]
            ys = [grades[u] for u in uids]
            r[key]["spearman"] = spearman(xs, ys)
        out["seeds"][str(seed)] = r
        print("seed %d: lstm acc=%.3f rho=%.3f | mean acc=%.3f rho=%.3f" % (
            seed, r["lstm"]["pair_acc"], r["lstm"]["spearman"],
            r["mean"]["pair_acc"], r["mean"]["spearman"]), flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
