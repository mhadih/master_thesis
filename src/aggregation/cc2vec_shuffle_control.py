"""Order-shuffle control for the supervised LSTM aggregator.

Same protocol as src/aggregation/cc2vec_supervised.py (5-fold CV over students,
pairwise-ranking loss, LSTM 196->16 + Dropout(0.2) + Linear(16->1), Adam 1e-3,
wd 1e-4, 20 epochs, seeds 0,1,2, CPU for determinism). The only change is the
ORDER of change vectors inside each file, applied before training:

  chrono   : original temporal order (reproduces supervised_cc2vec_5fold.json)
  shuffled : fixed random permutation per (file, seed). Truncate to last
             MAX_LEN first, then permute, so chrono vs shuffled differ ONLY in
             order — the same multiset of change vectors, same file length.
  reversed : deterministic flip (preserves adjacency, reverses direction).

Rationale: if the LSTM's advantage over mean-pooling survives destroying
temporal order, that advantage cannot be attributed to modelling the code
development process. The mean-pool arm is NOT rerun: it is permutation
invariant by construction, so it is identical in all three modes (reference
0.504 from supervised_cc2vec_5fold.json).

Writes results/experiments/supervised_cc2vec_shuffle_control.json, merging into
any existing file so modes can be run in separate invocations.

Env: ABL_STORE, ABL_HIDDEN(16), ABL_EPOCHS(20), ABL_LR(1e-3), ABL_FOLDS(5),
     ABL_SEEDS("0,1,2"), ABL_MAX_LEN(200), ABL_MODES("chrono,shuffled,reversed"),
     ABL_DEVICE("cpu"), ABL_SHUFFLE_SEED_BASE(1000), ABL_OUT.
"""

import csv
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
MODES = [m.strip() for m in os.environ.get(
    "ABL_MODES", "chrono,shuffled,reversed").split(",") if m.strip()]
SHUFFLE_SEED_BASE = int(os.environ.get("ABL_SHUFFLE_SEED_BASE", "1000"))
OUT = os.environ.get(
    "ABL_OUT",
    os.path.join(REPO, "results", "experiments", "supervised_cc2vec_shuffle_control.json"))


class Aggregator(nn.Module):
    """LSTM file encoder + linear grade head (identical to cc2vec_supervised.py)."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=196, hidden_size=HIDDEN, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(HIDDEN, 1)

    def file_emb(self, vecs):
        if len(vecs) > MAX_LEN:
            vecs = vecs[-MAX_LEN:]
        h, _ = self.lstm(vecs.unsqueeze(0))
        return h.squeeze(0)[-1]

    def student_score(self, files, flens, train=True):
        embs = torch.stack([self.file_emb(v) for v in files])
        if train:
            embs = self.drop(embs)
        w = torch.tensor(flens, dtype=torch.float32, device=embs.device)
        stu = (embs * (w / w.sum()).unsqueeze(1)).sum(dim=0)
        return self.head(self.drop(stu) if train else stu).squeeze(-1)


def load_data(device):
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
    return per_user, grades, sorted(per_user)


def transform(per_user, mode, shuffle_seed, device):
    """Return a per_user view with file orders transformed (fixed per mode+seed)."""
    if mode == "chrono":
        return per_user
    g = torch.Generator().manual_seed(shuffle_seed)
    out = defaultdict(list)
    for uid, feats in per_user.items():
        for vecs, flen in feats:
            v = vecs[-MAX_LEN:] if len(vecs) > MAX_LEN else vecs
            if mode == "shuffled":
                v = v[torch.randperm(len(v), generator=g)]
            elif mode == "reversed":
                v = v.flip(0)
            else:
                raise ValueError(f"unknown mode: {mode}")
            out[uid].append((v.clone(), flen))
    return out


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
    torch.manual_seed(seed)  # once per arm; matches cc2vec_supervised.py
    held_preds, held_ok, held_n = {}, 0, 0
    for fold in folds:
        test = set(fold)
        train = [u for u in uids if u not in test]
        order = torch.tensor(train)
        g = torch.tensor([grades[u] for u in train])
        model = Aggregator().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        model.train()
        for _ in range(EPOCHS):
            scores = student_scores(model, per_user, order, device, train=True)
            i, j = torch.randint(len(train), (2, 512))
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
    return {"preds": held_preds, "pair_acc": held_ok / held_n, "pairs": held_n}


def spearman(xs, ys):
    rx = {v: i for i, v in enumerate(sorted(set(xs)))}
    ry = {v: i for i, v in enumerate(sorted(set(ys)))}
    n = len(xs)
    d2 = sum((rx[x] - ry[y]) ** 2 for x, y in zip(xs, ys))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    device = torch.device(os.environ.get("ABL_DEVICE", "cpu"))
    per_user, grades, uids = load_data(device)
    print("graded users with files:", len(uids), "device:", device, flush=True)
    out = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            out = json.load(f)
    for mode in MODES:
        out.setdefault(mode, {})
        for seed in SEEDS:
            view = transform(per_user, mode, SHUFFLE_SEED_BASE + seed, device)
            r = run_seed(seed, view, grades, uids, device)
            r["spearman"] = spearman([r["preds"][u] for u in uids],
                                     [grades[u] for u in uids])
            del r["preds"]
            out[mode][str(seed)] = r
            print(f"{mode} seed {seed}: acc={r['pair_acc']:.3f} "
                  f"rho={r['spearman']:.3f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
