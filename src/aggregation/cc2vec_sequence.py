"""Sequence-aware file aggregation for CC2Vec change vectors (order-invariance study).

Reads the per-file change store (see generate_emb/cc2vec_changes_csv.py) and
builds per-user embeddings under three sequence-aware schemes (user level stays
length-weighted, as in the base setup):

  decay : file_emb = sum_i gamma^(C-1-i) * v_i / Z, gamma via ABL_DECAY_GAMMA
          (default 0.9). Closed-form recency, zero parameters.
  lstm-last / lstm-mean : frozen randomly-initialized LSTM (input 196,
          hidden via ABL_LSTM_HIDDEN default 64) over the chronological change
          sequence; file_emb = last hidden state / mean hidden state.
          Untrained by design (unsupervised setting, 62 students) — a fixed
          nonlinear reservoir. Run ABL_SEEDS times (default "0,1,2") and report
          mean+-std; never cherry-pick one seed.

Long files are truncated to the last ABL_MAX_LEN changes (default 200) so
monster files (max 3523 changes) don't dominate runtime.

Env vars:
  ABL_STORE   input .pt store (default: <repo>/data/cc2v_file_changes.pt)
  ABL_OUTDIR  output dir (default: student_embeddings/ in repo)
  ABL_DECAY_GAMMA, ABL_LSTM_HIDDEN, ABL_MAX_LEN, ABL_SEEDS (comma-separated)

Outputs: user_CC2Vec_abl_decay.jsonl,
         user_CC2Vec_abl_lstm_last_s{seed}.jsonl,
         user_CC2Vec_abl_lstm_mean_s{seed}.jsonl
"""

import json
import os
from collections import defaultdict

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.environ.get("ABL_STORE", os.path.join(REPO, "data", "cc2v_file_changes.pt"))
OUTDIR = os.environ.get("ABL_OUTDIR", os.path.join(REPO, "student_embeddings"))
GAMMA = float(os.environ.get("ABL_DECAY_GAMMA", "0.9"))
HIDDEN = int(os.environ.get("ABL_LSTM_HIDDEN", "64"))
MAX_LEN = int(os.environ.get("ABL_MAX_LEN", "200"))
SEEDS = [int(s) for s in os.environ.get("ABL_SEEDS", "0,1,2").split(",") if s.strip() != ""]


def build_lstm(seed):
    torch.manual_seed(seed)  # seeds the default uniform init below
    lstm = nn.LSTM(input_size=196, hidden_size=HIDDEN, batch_first=True)
    for p in lstm.parameters():
        p.detach_()
        p.requires_grad_(False)
    lstm.eval()
    return lstm


def file_sequences(store):
    """Yield (uid, fn, seq[C,196] truncated to last MAX_LEN, flen)."""
    for (uid, fn), d in store.items():
        vecs = d["vecs"].float()
        if len(vecs) > MAX_LEN:
            vecs = vecs[-MAX_LEN:]
        yield uid, fn, vecs, d["flen"]


def write_user_embeddings(per_user, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for uid in sorted(per_user, key=int):
            items = per_user[uid]
            embs = torch.stack([e for e, _ in items])
            w = torch.tensor([fl for _, fl in items], dtype=torch.float32)
            w = w / w.sum()
            emb = (embs * w.unsqueeze(1)).sum(dim=0)
            f.write(json.dumps({"user_id": str(uid), "embedding": [float(x) for x in emb.tolist()]}) + "\n")
    print("users:", len(per_user), "->", out, flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = torch.load(STORE, map_location="cpu", weights_only=False)
    print("files:", len(store), "device:", device, flush=True)
    seqs = list(file_sequences(store))

    # decay weighting
    per_user = defaultdict(list)
    for uid, fn, vecs, flen in seqs:
        c = len(vecs)
        w = torch.tensor([GAMMA ** (c - 1 - i) for i in range(c)], dtype=torch.float32)
        per_user[uid].append(((vecs * (w / w.sum()).unsqueeze(1)).sum(dim=0), flen))
    write_user_embeddings(per_user, os.path.join(OUTDIR, "user_CC2Vec_abl_decay.jsonl"))

    # frozen LSTM variants
    for seed in SEEDS:
        lstm = build_lstm(seed).to(device)
        last, mean = defaultdict(list), defaultdict(list)
        with torch.no_grad():
            for uid, fn, vecs, flen in seqs:
                h, _ = lstm(vecs.unsqueeze(0).to(device))
                h = h.squeeze(0).cpu()
                last[uid].append((h[-1], flen))
                mean[uid].append((h.mean(dim=0), flen))
        write_user_embeddings(last, os.path.join(OUTDIR, f"user_CC2Vec_abl_lstm_last_s{seed}.jsonl"))
        write_user_embeddings(mean, os.path.join(OUTDIR, f"user_CC2Vec_abl_lstm_mean_s{seed}.jsonl"))


if __name__ == "__main__":
    main()
