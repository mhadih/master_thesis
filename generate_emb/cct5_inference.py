import psycopg2
import pandas as pd
import os
import torch
import difflib
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from collections import defaultdict
from tqdm import tqdm
import json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.db_config import get_code_recorder_config

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


# ---- CCT5 setup (CodeT5-based code-change model, ESEC/FSE 2023) ----
# A code change is encoded as a unified diff with special markers:
#   '+' lines -> "<add>", '-' lines -> "<del>", context -> "<keep>"
# (mirrors the input construction in the CCT5 reproduction repo, minus DFG edges).
# Representation = T5 encoder mean-pool over the diff string (768-dim),
# same protocol as codeT5_inference.py in this thesis.
CCT5_DIR = os.environ.get("CCT5_DIR", "/home/hadi/thesis/cct5")
MODEL_NAME = "Salesforce/codet5-base"
MAX_LENGTH = 512  # Truncate long diffs


def build_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    base = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<keep>", "<add>", "<del>",
            "<start>", "<end>", "<issue_id>", "<version_id>", "<commit_id>"]
    extra = ["<extra_id_{}>".format(i) for i in range(99, -1, -1)]
    extra += ["<e{}>".format(i) for i in range(99, -1, -1)]
    extra += ["<msg>"]
    add_tokens = base + extra
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [t for t in add_tokens if t not in tokenizer.get_vocab()]})
    return tokenizer


def encode_change(old_code, new_code):
    """One consecutive snapshot pair -> CCT5 diff string (faithful to upstream)."""
    diff = list(difflib.unified_diff(old_code.splitlines(), new_code.splitlines()))[2:]
    parts = []
    for line in diff[1:]:
        if not line:
            continue
        if line[0] == '+':
            parts.append("<add>" + line[1:])
        elif line[0] == '-':
            parts.append("<del>" + line[1:])
        else:
            parts.append("<keep>" + line[1:])
    return "".join(parts)


tokenizer = build_tokenizer()
# NOTE: use_safetensors=True avoids torch.load, which transformers>=4.57 blocks
# for torch<2.6 (CVE-2025-32434). Place a safetensors conversion of the
# CCT5 pre-training checkpoint next to pytorch_model.bin in CCT5_DIR first.
model = T5ForConditionalGeneration.from_pretrained(CCT5_DIR, use_safetensors=True)
model.eval()  # inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("### Load CCT5 model is completed!")


def get_cct5_embedding(diff_str):
    tokens = tokenizer(diff_str, return_tensors='pt', truncation=True,
                       padding='max_length', max_length=MAX_LENGTH)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        encoder_out = model.encoder(input_ids=tokens["input_ids"],
                                    attention_mask=tokens["attention_mask"])[0]
        embedding = encoder_out.mean(dim=1).squeeze(0)  # shape: (hidden_dim,)
    return embedding.cpu()


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
    change_reprs = []
    for i in range(1, len(contents)):
        if not contents[i - 1].strip() or not contents[i].strip():
            continue
        diff_str = encode_change(contents[i - 1], contents[i])
        if not diff_str.strip():
            continue  # identical snapshots carry no change signal
        change_reprs.append(get_cct5_embedding(diff_str))
    if not change_reprs:
        continue
    file_embedding = torch.stack(change_reprs).mean(dim=0)  # [hidden_dim]
    user_embeddings[user_id].append(file_embedding)
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
    )  # shape: (hidden_dim=768,)

# Path to save embeddings
output_file = "user_CCT5_embeddings.jsonl"

with open(output_file, 'w') as f:
    for user_id, emb in final_user_embeddings.items():
        emb_list = emb.tolist()  # Convert tensor to native Python list
        json_line = {
            "user_id": str(user_id),   # Convert to string to avoid int64 issues
            "embedding": [float(x) for x in emb_list]
        }
        f.write(json.dumps(json_line) + '\n')

print(f"### Saved user embeddings to {output_file}")
