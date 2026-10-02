import psycopg2
import pandas as pd
import os
import torch
import re
from transformers import RobertaTokenizerFast, RobertaModel
from huggingface_hub import hf_hub_download
from collections import defaultdict
from tqdm import tqdm
import json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cct5"))
from src.db_config import get_code_recorder_config
from dfg_cpp import extract_dataflow

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


# ---- GraphCodeBERT with data flow (structural-rich embeddings) ----
# Per snapshot: [CLS] + code BPE tokens + [SEP] + DFG nodes, with
#   * position ids: standard for code, NODE_POSITION_ID (shared) for all DFG
#     nodes so the model treats them as an unordered set;
#   * graph-guided mask: code<->code full, node<->its code tokens, node<->node
#     along data-flow edges (symmetric + self);
#   * node embeddings initialized as the mean of their mapped code-token
#     embeddings (official GraphCodeBERT practice).
# Snapshot vector = concat([CLS], mean code-token states, mean DFG-node
# states) = 3*768-dim. Consecutive snapshot vectors are differenced
# (e_t - e_{t-1}) to represent change; diffs average per file, files average
# per user by final length.
MODEL_NAME = "microsoft/graphcodebert-base"
MAX_LENGTH = 512
MAX_NODES = 128
NODE_POSITION_ID = 0
VALIDATE_N = 5  # first snapshots also embedded without DFG; mean cosine shift printed
# Experiment population (same 68 S03 users as the other tracks; the DB holds 284).
# Full-corpus runs are possible by emptying this set, at ~4x the cost.
TARGET_USERS = {
    "7", "49", "75", "77", "83", "142", "143", "145", "146", "148", "150", "154",
    "159", "160", "162", "163", "164", "165", "166", "167", "169", "175", "179",
    "180", "181", "182", "183", "184", "189", "191", "195", "198", "199", "204",
    "205", "212", "218", "219", "222", "223", "227", "229", "230", "231", "232",
    "233", "234", "238", "240", "242", "243", "245", "247", "248", "249", "250",
    "252", "255", "259", "261", "262", "267", "269", "273", "279", "281", "283",
    "284",
}

# NOTE: the fast tokenizer is required (not the slow RobertaTokenizer):
# only it returns offset_mapping, which aligns tree-sitter tokens to BPE
# spans. It is built from the same vocab.json/merges.txt, so ids are
# byte-identical to the slow tokenizer (verified).
tokenizer = RobertaTokenizerFast(
    vocab_file=hf_hub_download(MODEL_NAME, "vocab.json"),
    merges_file=hf_hub_download(MODEL_NAME, "merges.txt"))
# NOTE: use_safetensors=True avoids torch.load, which transformers>=4.57 blocks
# for torch<2.6 (CVE-2025-32434). The official repo ships only pytorch_model.bin,
# so place a safetensors conversion next to it in the HF cache first.
model = RobertaModel.from_pretrained(MODEL_NAME, use_safetensors=True)
model.eval()  # inference mode

# Choose your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def build_graph_inputs(code):
    """Snapshot code -> dict with input_ids, position_ids, attn_mask, splits,
    or None if the snapshot yields no DFG nodes (falls back to plain encoding
    by the caller). All index math in final-sequence coordinates."""
    clean = code  # NOTE: must be the exact string fed to extract_dataflow
    tokens, dfg, index_to_code = extract_dataflow(clean)
    if not dfg:
        return None
    enc = tokenizer(clean, return_offsets_mapping=True, add_special_tokens=False)
    bpe_spans = enc["offset_mapping"]
    # (line, col) -> char offset
    line_starts, pos = [0], 0
    for m in re.finditer("\n", clean):
        line_starts.append(m.end())
    pos_to_off = lambda p: line_starts[p[0]] + p[1]  # noqa: E731

    # unique DFG node occurrences, ordered by token idx
    node_keys, seen = [], set()
    for edge in dfg:
        for key in [(edge[0], edge[1])] + [(c, i) for c, i in zip(edge[3], edge[4])]:
            if key not in seen:
                seen.add(key)
                node_keys.append(key)
    # node -> BPE span (in code-only coordinates)
    idx_to_pos = {i: pos for pos, (i, _) in index_to_code.items()}
    node_spans = []
    for _, idx in node_keys:
        start = pos_to_off(idx_to_pos[idx][0])
        end = pos_to_off(idx_to_pos[idx][1])
        span = [j for j, (a, b) in enumerate(bpe_spans) if not (b <= start or a >= end)]
        node_spans.append(span)
    keep = [k for k, s in enumerate(node_spans) if s][:MAX_NODES]
    if not keep:
        return None
    node_keys = [node_keys[k] for k in keep]
    node_spans = [node_spans[k] for k in keep]
    # edge set restricted to kept nodes, as index pairs
    key_to_node = {k: i for i, k in enumerate(node_keys)}
    edge_pairs = set()
    for edge in dfg:
        src = (edge[0], edge[1])
        if src not in key_to_node:
            continue
        for c, i in zip(edge[3], edge[4]):
            if (c, i) in key_to_node:
                a, b = key_to_node[src], key_to_node[(c, i)]
                edge_pairs.add((a, b))
                edge_pairs.add((b, a))

    code_ids = enc["input_ids"]
    n_keep_code = MAX_LENGTH - 2 - len(node_keys)
    code_ids = code_ids[:n_keep_code]
    # drop nodes whose span was truncated away
    alive = [i for i, s in enumerate(node_spans) if s and max(s) < len(code_ids)]
    if not alive:
        return None
    node_keys = [node_keys[i] for i in alive]
    node_spans = [[p for p in node_spans[i] if p < len(code_ids)] for i in alive]
    node_spans = [s for s in node_spans if s]
    if not node_spans:
        return None
    # re-index edges after drops
    old_to_new = {o: n for n, o in enumerate(alive)}
    edge_pairs = {(old_to_new[a], old_to_new[b]) for a, b in edge_pairs
                  if a in old_to_new and b in old_to_new}

    n_code, n_node = len(code_ids), len(node_spans)
    L = 1 + n_code + 1 + n_node
    input_ids = [tokenizer.cls_token_id] + code_ids + [tokenizer.sep_token_id] + \
        [tokenizer.mask_token_id] * n_node
    position_ids = [0] + list(range(1, n_code + 1)) + [n_code + 1] + [NODE_POSITION_ID] * n_node
    allowed = torch.zeros(L, L, dtype=torch.bool)
    allowed[0:1 + n_code, 0:1 + n_code] = True  # CLS+code <-> CLS+code (SEP fenced below)
    base = 1 + n_code + 1
    for i, span in enumerate(node_spans):
        for p in span:
            allowed[base + i, 1 + p] = True
            allowed[1 + p, base + i] = True
    for a, b in edge_pairs:
        allowed[base + a, base + b] = True
    allowed.fill_diagonal_(True)
    extended = (1.0 - allowed.float()).unsqueeze(0).unsqueeze(0) * -10000.0
    return {"input_ids": input_ids, "position_ids": position_ids,
            "extended_mask": extended, "n_code": n_code, "n_node": n_node,
            "spans": node_spans}


def _pos_of(index_to_code, idx):
    for pos, (i, _) in index_to_code.items():
        if i == idx:
            return pos
    raise KeyError(idx)


@torch.no_grad()
def encode_snapshot(code):
    """Full forward; returns (concat_vec[2304], used_dfg: bool)."""
    g = build_graph_inputs(code)
    if g is None:
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        h = model(**inputs)[0].squeeze(0)
        cls = h[0]
        return torch.cat([cls, cls, cls]).cpu(), False
    input_ids = torch.tensor([g["input_ids"]], dtype=torch.long, device=device)
    position_ids = torch.tensor([g["position_ids"]], dtype=torch.long, device=device)
    w = model.embeddings.word_embeddings(input_ids)
    p = model.embeddings.position_embeddings(position_ids)
    t = model.embeddings.token_type_embeddings(torch.zeros_like(input_ids))
    emb = w + p + t
    # init DFG node embeddings from mapped code-token embeddings
    code_states = w[0, 1:1 + g["n_code"]]
    for i, span in enumerate(g["spans"]):
        emb[0, 1 + g["n_code"] + 1 + i] = code_states[
            torch.tensor(span, device=device)].mean(dim=0)
    emb = model.embeddings.LayerNorm(emb)
    h = model.encoder(emb, attention_mask=g["extended_mask"].to(device))[0].squeeze(0)
    cls = h[0]
    code_mean = h[1:1 + g["n_code"]].mean(dim=0)
    node_mean = h[1 + g["n_code"] + 1:].mean(dim=0)
    return torch.cat([cls, code_mean, node_mean]).cpu(), True


def _node_span(g, i):
    return g["spans"][i]


@torch.no_grad()
def encode_snapshot_plain(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    h = model(**inputs)[0].squeeze(0)
    cls = h[0]
    return torch.cat([cls, cls, cls]).cpu(), False


# Path to save embeddings (separate file: DFG-guided 2304-d vectors;
# user_GraphCodeBert_embeddings.jsonl holds the earlier plain 768-d run)
output_file = "user_GraphCodeBert_dfg_embeddings.jsonl"

# Resume support: already-embedded users are skipped (incremental append below),
# so killing the run (or shutting down) loses at most the in-flight user.
DONE = set()
if os.path.exists(output_file):
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if line:
                DONE.add(json.loads(line)["user_id"])
    print("resume: %d users already embedded, skipping" % len(DONE), flush=True)


# Storage for user embeddings
user_embeddings = defaultdict(list)
lengths = defaultdict(list)

# Per-snapshot embeddings, then consecutive differences e_t - e_{t-1}
grouped = sorted_df[sorted_df["user_id"].astype(str).isin(TARGET_USERS)].groupby(
    ['user_id', 'filename'], sort=False)

validated = 0
for (user_id, filename), group in tqdm(grouped, desc="Processing groups", unit="group"):
    if str(user_id) in DONE:
        continue  # resume: skip users embedded in a previous run
    if not(filename.endswith("cpp") or filename.endswith("hpp")):
        continue
    group = group.sort_values(by='date').reset_index(drop=True)
    snaps = []
    for _, row in group.iterrows():
        content = row["content"]
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            vec, used = encode_snapshot(content)
        except RuntimeError as e:
            # transient CUDA failure (2GB card): clear state, log, skip snapshot
            print("snapshot failed (%s/%s): %s" % (user_id, filename, str(e).splitlines()[0][:120]), flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            with open("skipped_snapshots.log", "a") as sf:
                sf.write("%s\t%s\t%s\n" % (user_id, filename, row.get("date", "")))
            continue
        snaps.append(vec)
        if validated < VALIDATE_N and used:
            plain, _ = encode_snapshot_plain(content)
            from torch.nn.functional import cosine_similarity as cos
            print("validate: DFG-vs-plain cosine distance = %.4f"
                  % (1 - cos(vec, plain, dim=0).item()), flush=True)
            validated += 1
    if len(snaps) < 2:
        continue  # need at least one transition
    diffs = torch.stack([(snaps[i] - snaps[i - 1]) for i in range(1, len(snaps))])
    file_emb = diffs.mean(dim=0)  # [2304]
    user_embeddings[user_id].append(file_emb)
    final_code = group.iloc[-1]["content"]
    lengths[user_id].append(len(final_code.split()) if isinstance(final_code, str) and final_code.strip() else 0)

print("### Generate embeddings is completed!")


# Aggregate per user (weighted sum of their file embeddings)
final_user_embeddings = {}
for user_id, emb_list in user_embeddings.items():
    weights = torch.tensor(lengths[user_id], dtype=torch.float32)
    weights = weights / weights.sum()
    final_user_embeddings[user_id] = torch.sum(
        weights[:, None] * torch.stack(emb_list), dim=0
    )  # shape: (2304,)

with open(output_file, 'a') as f:  # append: each run adds newly finished users
    for user_id, emb in final_user_embeddings.items():
        emb_list = emb.tolist()  # Convert tensor to native Python list
        json_line = {
            "user_id": str(user_id),   # Convert to string to avoid int64 issues
            "embedding": [float(x) for x in emb_list]
        }
        f.write(json.dumps(json_line) + '\n')
        f.flush()

print(f"### Saved user embeddings to {output_file}")

conn.close()
