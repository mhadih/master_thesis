import psycopg2
import pandas as pd
import os
import torch
import difflib
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cc2vec"))
from src.db_config import get_code_recorder_config
from jit_cc2ftr_model import HierachicalRNN

conn = psycopg2.connect(**get_code_recorder_config())

query = """
    SELECT user_id, filename, content, date
    FROM codetrace
    WHERE identifier = 'Text Change'
"""

traces_df = pd.read_sql_query(query, conn)

sorted_df = traces_df.sort_values(by='date')

# Assume df is already in correct chronological order
# Columns: user_id, filename, content
print("### Structured data (dataframe) is ready!")


# ---- CC2Vec setup (Hierarchical Attention Network over added/removed code) ----
# Pretrained on OpenStack patches (embed_size=64, hidden_size=32).
# Patch representation = forward_commit_embeds_diff -> 196-dim vector.
CC2VEC_DIR = os.path.join(os.path.dirname(__file__), "..", "cc2vec")
BATCH = 32
MAX_FILE, MAX_LINE, MAX_LEN = 2, 10, 64  # upstream padding defaults

with open(os.path.join(CC2VEC_DIR, "openstack_dict.pkl"), "rb") as f:
    _, dict_code = pickle.load(f)


class Args:
    vocab_code = len(dict_code)
    batch_size = BATCH
    embed_size = 64
    hidden_size = 32
    class_num = 54445  # must match checkpoint fc2 (log-message vocab); unused by embedding path
    dropout_keep_prob = 0.5


model = HierachicalRNN(args=Args())
model.load_state_dict(torch.load(os.path.join(CC2VEC_DIR, "openstack_cc2ftr.pt"), map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # inference mode
print("### Load CC2Vec model is completed!")


def split_change(old_code, new_code):
    """Split a consecutive snapshot pair into added / removed line lists."""
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()
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
    added = [l for l in added if l.strip()]
    removed = [l for l in removed if l.strip()]
    return added, removed


def pad_line(line, max_length=MAX_LEN):
    toks = line.split()  # any whitespace (tabs/multiple spaces collapse)
    if len(toks) < max_length:
        return " ".join(toks + ["<NULL>"] * (max_length - len(toks)))
    return " ".join(toks[:max_length])


def null_file():
    return [(" <NULL>" * MAX_LEN).strip()] * MAX_LINE


def pad_pair(added, removed):
    """One change pair -> padded ([files, lines, words]) structures."""
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
    """Lowercase whitespace-token mapping with dict_code (OOV -> <NULL>)."""
    null_id = dict_code["<NULL>"]
    return [[[dict_code.get(t.lower(), null_id) for t in line.split(" ")] for line in f] for f in padded]


@torch.no_grad()
def encode_pairs(pairs):
    """pairs: list of (added_ids, removed_ids); returns [N, 196] tensor on CPU."""
    vecs = []
    for j in range(0, len(pairs), BATCH):
        chunk = pairs[j:j + BATCH]
        b = len(chunk)
        # NOTE: upstream forward_code expects numpy (it .cuda()-converts internally)
        add = np.array([p[0] for p in chunk])
        rem = np.array([p[1] for p in chunk])
        # pad batch with NULL-only pairs if short (keeps model.batch_size views valid)
        if b < BATCH:
            pad = np.full((BATCH - b, MAX_FILE, MAX_LINE, MAX_LEN), dict_code["<NULL>"])
            add = np.concatenate([add, pad])
            rem = np.concatenate([rem, pad])
        hs = (
            torch.zeros(2, BATCH, 32, device=device),
            torch.zeros(2, BATCH, 32, device=device),
            torch.zeros(2, BATCH, 32, device=device),
        )
        out = model.forward_commit_embeds_diff(add, rem, *hs)[:b]
        vecs.append(out.cpu())
    return torch.cat(vecs, dim=0)


# Storage for user embeddings
user_embeddings = defaultdict(list)
lengths = defaultdict(list)

# Generate embeddings (final snapshot per (user_id, filename), consecutive pairs)
grouped = sorted_df.groupby(['user_id', 'filename'], sort=False)

for (user_id, filename), group in tqdm(grouped, desc="Processing groups", unit="group"):
    if not filename.endswith("cpp"):
        continue
    group = group.sort_values(by='date').reset_index(drop=True)
    contents = [c if isinstance(c, str) else "" for c in group["content"].tolist()]
    pairs = []
    for i in range(1, len(contents)):
        if not contents[i - 1].strip() or not contents[i].strip():
            continue
        added, removed = split_change(contents[i - 1], contents[i])
        if not added and not removed:
            continue  # identical snapshots carry no change signal
        a, r = pad_pair(added, removed)
        pairs.append((map_code(a), map_code(r)))
    if not pairs:
        continue
    file_reprs = encode_pairs(pairs)  # [num_changes, 196]
    user_embeddings[user_id].append(file_reprs.mean(dim=0))
    final_code = contents[-1]
    lengths[user_id].append(len(final_code.split()) if final_code.strip() else 0)

print("### Generate embeddings is completed!")


# Aggregate per user (word-count-weighted average of their file embeddings)
final_user_embeddings = {}
for user_id, emb_list in user_embeddings.items():
    weights = torch.tensor(lengths[user_id], dtype=torch.float32)
    weights = weights / weights.sum()
    final_user_embeddings[user_id] = torch.sum(
        weights[:, None] * torch.stack(emb_list), dim=0
    )  # shape: (196,)

# Path to save embeddings
output_file = "user_CC2Vec_embeddings.jsonl"

with open(output_file, 'w') as f:
    for user_id, emb in final_user_embeddings.items():
        emb_list = emb.tolist()  # Convert tensor to native Python list
        json_line = {
            "user_id": str(user_id),   # Convert to string to avoid int64 issues
            "embedding": [float(x) for x in emb_list]
        }
        f.write(json.dumps(json_line) + '\n')

print(f"### Saved user embeddings to {output_file}")
