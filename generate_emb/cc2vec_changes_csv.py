"""CC2Vec per-file change vectors from a local CSV (no database needed).

Same change encoding as cc2vec_inference.py (difflib added/removed lines per
consecutive snapshot pair -> upstream 2x10x64 padding/mapping ->
forward_commit_embeds_diff, 196-dim per change), but reads snapshots from a CSV
file with columns [user_id, filename, content, date] and saves every file's
full change matrix for offline aggregation (see src/aggregation/cc2vec_ablations.py):
  {(user_id, filename): {'vecs': Tensor[C,196] (CPU), 'added': [int],
                         'removed': [int], 'flen': int}}

Designed for cloud runners (Kaggle / Colab) where the DB is unavailable.
Incremental .pt saves + resume support: aborts lose nothing.

Env vars:
  CPP_TRACES_CSV   snapshots CSV (default: <repo>/my_ccbert/cpp_traces.csv)
  CC2VEC_STORE     output .pt store (default: <repo>/data/cc2v_file_changes.pt;
                   90MB, gitignored)
  CC2VEC_LIMIT_USERS  process only first N users (smoke tests)
"""

import difflib
import json
import os
import pickle
import sys
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC2VEC_DIR = os.path.join(REPO, "cc2vec")
sys.path.insert(0, CC2VEC_DIR)
os.chdir(CC2VEC_DIR)
from jit_cc2ftr_model import HierachicalRNN

CSV_PATH = os.environ.get("CPP_TRACES_CSV", os.path.join(REPO, "my_ccbert", "cpp_traces.csv"))
OUT = os.environ.get("CC2VEC_STORE", os.path.join(REPO, "data", "cc2v_file_changes.pt"))
LIMIT = int(os.environ.get("CC2VEC_LIMIT_USERS", "0"))
# S03 target users (= user_ids in student_embeddings/user_CodeT5_file_embeddings.jsonl,
# inlined here so other code can reuse them without reading that file).
TARGET_USERS = {
    "7", "49", "75", "77", "83", "142", "143", "145", "146", "148", "150", "154",
    "159", "160", "162", "163", "164", "165", "166", "167", "169", "175", "179",
    "180", "181", "182", "183", "184", "189", "191", "195", "198", "199", "204",
    "205", "212", "218", "219", "222", "223", "227", "229", "230", "231", "232",
    "233", "234", "238", "240", "242", "243", "245", "247", "248", "249", "250",
    "252", "255", "259", "261", "262", "267", "269", "273", "279", "281", "283",
    "284",
}
BATCH = 32
MAX_FILE, MAX_LINE, MAX_LEN = 2, 10, 64

with open(os.path.join(CC2VEC_DIR, "openstack_dict.pkl"), "rb") as f:
    _, dict_code = pickle.load(f)
NULL_ID = dict_code["<NULL>"]


class Args:
    vocab_code = len(dict_code)
    batch_size = BATCH
    embed_size = 64
    hidden_size = 32
    class_num = 54445
    dropout_keep_prob = 0.5


model = HierachicalRNN(args=Args())
model.load_state_dict(torch.load(os.path.join(CC2VEC_DIR, "openstack_cc2ftr.pt"), map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model on", device, flush=True)


def split_change(old_code, new_code):
    old_lines, new_lines = old_code.splitlines(), new_code.splitlines()
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    added, removed = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            removed.extend(old_lines[i1:i2])
            added.extend(new_lines[j1:j2])
        elif tag == "delete":
            removed.extend(old_lines[i1:i2])
        elif tag == "insert":
            added.extend(new_lines[j1:j2])
    return [l for l in added if l.strip()], [l for l in removed if l.strip()]


def pad_line(line, max_length=MAX_LEN):
    toks = line.split()
    if len(toks) < max_length:
        return " ".join(toks + ["<NULL>"] * (max_length - len(toks)))
    return " ".join(toks[:max_length])


def null_file():
    return [(" <NULL>" * MAX_LEN).strip()] * MAX_LINE


def pad_pair(added, removed):
    def pad_file(lines):
        lines = [pad_line(l) for l in lines[:MAX_LINE]]
        while len(lines) < MAX_LINE:
            lines.append((" <NULL>" * MAX_LEN).strip())
        return lines

    def pad_files(files):
        files = files[:MAX_FILE]
        while len(files) < MAX_FILE:
            files.append(null_file())
        return files

    return pad_files([pad_file(added)]), pad_files([pad_file(removed)])


def map_code(padded):
    return [[[dict_code.get(t.lower(), NULL_ID) for t in line.split(" ")] for line in f] for f in padded]


@torch.no_grad()
def encode_pairs(pairs):
    import numpy as _np
    vecs = []
    for j in range(0, len(pairs), BATCH):
        chunk = pairs[j:j + BATCH]
        b = len(chunk)
        add = _np.array([p[0] for p in chunk])
        rem = _np.array([p[1] for p in chunk])
        if b < BATCH:
            pad = _np.full((BATCH - b, MAX_FILE, MAX_LINE, MAX_LEN), NULL_ID)
            add = _np.concatenate([add, pad])
            rem = _np.concatenate([rem, pad])
        hs = (
            torch.zeros(2, BATCH, 32, device=device),
            torch.zeros(2, BATCH, 32, device=device),
            torch.zeros(2, BATCH, 32, device=device),
        )
        vecs.append(model.forward_commit_embeds_diff(add, rem, *hs)[:b].cpu())
    return torch.cat(vecs, dim=0)


store = {}
if os.path.exists(OUT):
    store = torch.load(OUT, map_location="cpu", weights_only=False)
    print("resume: %d files already saved" % len(store), flush=True)

user_dfs = {}
for ch in pd.read_csv(
    CSV_PATH,
    usecols=["user_id", "filename", "content", "date"],
    chunksize=200000,
    dtype={"user_id": str},
    low_memory=False,
):
    ch = ch[ch["user_id"].isin(TARGET_USERS)]
    if ch.empty:
        continue
    for uid, grp in ch.groupby("user_id", sort=False):
        user_dfs.setdefault(uid, []).append(grp)
print("users found:", len(user_dfs), flush=True)

uids = sorted(user_dfs, key=int)
if LIMIT:
    uids = uids[:LIMIT]
n_files = 0
for uid in tqdm(uids, desc="Users"):
    df = pd.concat(user_dfs[uid])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    for fn, fdf in df.groupby("filename", sort=False):
        if (uid, fn) in store:
            continue
        fdf = fdf.reset_index(drop=True)
        contents = [c if isinstance(c, str) else "" for c in fdf["content"].tolist()]
        pairs, added_n, removed_n = [], [], []
        for i in range(1, len(contents)):
            if not contents[i - 1].strip() or not contents[i].strip():
                continue
            added, removed = split_change(contents[i - 1], contents[i])
            if not added and not removed:
                continue
            a, r = pad_pair(added, removed)
            pairs.append((map_code(a), map_code(r)))
            added_n.append(len(added))
            removed_n.append(len(removed))
        if not pairs:
            continue
        vecs = encode_pairs(pairs)
        store[(uid, fn)] = {
            "vecs": vecs,
            "added": added_n,
            "removed": removed_n,
            "flen": len(contents[-1].split()) if contents[-1].strip() else 0,
        }
        n_files += 1
        if n_files % 50 == 0:
            torch.save(store, OUT)
torch.save(store, OUT)
print("files saved: %d total, output: %s" % (len(store), OUT), flush=True)
