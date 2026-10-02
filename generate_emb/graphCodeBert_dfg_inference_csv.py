"""GraphCodeBERT DFG-guided embeddings from a local CSV (no database needed).

Same logic as graphCodeBert_inference.py (per snapshot: [CLS] + code BPE +
[SEP] + DFG nodes, shared node position, graph-guided mask, node embeddings
from mapped token means; snapshot vector = concat([CLS], mean code, mean nodes)
= 2304-d; consecutive differences e_t - e_{t-1} averaged per file, files
averaged per user by final length), but reads snapshots from a CSV file with
columns [user_id, filename, content, date] instead of PostgreSQL.

Designed for cloud runners (Kaggle / Colab) where the DB is unavailable.
C++ data flow comes from cct5/dfg_cpp.py (needs: pip install tree_sitter
tree_sitter_cpp==0.23.4 ... pinned versions in repo history).

Env vars:
  CPP_TRACES_CSV  snapshots CSV (default: <repo>/my_ccbert/cpp_traces.csv)
  GCB_BATCH       not used (one snapshot per forward; variable lengths)
  GCB_LIMIT_USERS embed only first N users (smoke tests)

Writes student_embeddings/user_GraphCodeBert_dfg_embeddings.jsonl under the
repo root. Incremental append + resume support.
"""

import json
import os
import re
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel
from huggingface_hub import hf_hub_download

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "cct5"))
from dfg_cpp import extract_dataflow

CSV_PATH = os.environ.get("CPP_TRACES_CSV", os.path.join(REPO, "my_ccbert", "cpp_traces.csv"))
TARGET_USERS = {
    "7", "49", "75", "77", "83", "142", "143", "145", "146", "148", "150", "154",
    "159", "160", "162", "163", "164", "165", "166", "167", "169", "175", "179",
    "180", "181", "182", "183", "184", "189", "191", "195", "198", "199", "204",
    "205", "212", "218", "219", "222", "223", "227", "229", "230", "231", "232",
    "233", "234", "238", "240", "242", "243", "245", "247", "248", "249", "250",
    "252", "255", "259", "261", "262", "267", "269", "273", "279", "281", "283",
    "284",
}
MODEL_NAME = "microsoft/graphcodebert-base"
MAX_LENGTH = 512
MAX_NODES = 128
NODE_POSITION_ID = 0
VALIDATE_N = 5
LIMIT = int(os.environ.get("GCB_LIMIT_USERS", "0"))
OUT = os.path.join(REPO, "student_embeddings", "user_GraphCodeBert_dfg_embeddings.jsonl")

# Fast tokenizer is required (offsets for tree-sitter->BPE alignment); ids are
# byte-identical to the slow tokenizer.
tokenizer = RobertaTokenizerFast(
    vocab_file=hf_hub_download(MODEL_NAME, "vocab.json"),
    merges_file=hf_hub_download(MODEL_NAME, "merges.txt"))
try:
    # Local safetensors copy if present (torch<2.6 cannot torch.load the .bin).
    model = RobertaModel.from_pretrained(MODEL_NAME, use_safetensors=True)
except Exception:
    model = RobertaModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("### Load GraphCodeBERT+DFG model is completed on", device, flush=True)


def build_graph_inputs(code):
    """Snapshot code -> model-ready dict, or None (falls back to plain)."""
    tokens, dfg, index_to_code = extract_dataflow(code)
    if not dfg:
        return None
    enc = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
    bpe_spans = enc["offset_mapping"]
    line_starts = [0]
    for m in re.finditer("\n", code):
        line_starts.append(m.end())
    pos_to_off = lambda p: line_starts[p[0]] + p[1]  # noqa: E731

    node_keys, seen = [], set()
    for edge in dfg:
        for key in [(edge[0], edge[1])] + [(c, i) for c, i in zip(edge[3], edge[4])]:
            if key not in seen:
                seen.add(key)
                node_keys.append(key)
    idx_to_pos = {i: pos for pos, (i, _) in index_to_code.items()}
    node_spans = []
    for _, idx in node_keys:
        start = pos_to_off(idx_to_pos[idx][0])
        end = pos_to_off(idx_to_pos[idx][1])
        node_spans.append([j for j, (a, b) in enumerate(bpe_spans) if not (b <= start or a >= end)])
    keep = [k for k, s in enumerate(node_spans) if s][:MAX_NODES]
    if not keep:
        return None
    key_to_node = {node_keys[k]: i for i, k in enumerate(keep)}
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

    code_ids = enc["input_ids"][:MAX_LENGTH - 2 - len(keep)]
    alive = [i for i, k in enumerate(keep)
             if node_spans[k] and max(node_spans[k]) < len(code_ids)]
    if not alive:
        return None
    spans = [[p for p in node_spans[keep[i]] if p < len(code_ids)] for i in alive]
    spans = [s for s in spans if s]
    if not spans:
        return None
    # remap kept-node positions first
    kept_order = [keep[i] for i in alive]
    key_to_node2 = {node_keys[k]: n for n, k in enumerate(kept_order)}
    edge_pairs = set()
    for edge in dfg:
        src = (edge[0], edge[1])
        if src not in key_to_node2:
            continue
        for c, i in zip(edge[3], edge[4]):
            if (c, i) in key_to_node2:
                a, b = key_to_node2[src], key_to_node2[(c, i)]
                edge_pairs.add((a, b))
                edge_pairs.add((b, a))

    n_code, n_node = len(code_ids), len(spans)
    L = 1 + n_code + 1 + n_node
    input_ids = [tokenizer.cls_token_id] + code_ids + [tokenizer.sep_token_id] + \
        [tokenizer.mask_token_id] * n_node
    position_ids = [0] + list(range(1, n_code + 1)) + [n_code + 1] + [NODE_POSITION_ID] * n_node
    allowed = torch.zeros(L, L, dtype=torch.bool)
    allowed[0:1 + n_code, 0:1 + n_code] = True
    base = 1 + n_code + 1
    for i, span in enumerate(spans):
        for p in span:
            allowed[base + i, 1 + p] = True
            allowed[1 + p, base + i] = True
    for a, b in edge_pairs:
        allowed[base + a, base + b] = True
    allowed.fill_diagonal_(True)
    extended = (1.0 - allowed.float()).unsqueeze(0).unsqueeze(0) * -10000.0
    return {"input_ids": input_ids, "position_ids": position_ids,
            "extended_mask": extended, "n_code": n_code, "n_node": n_node,
            "spans": spans}


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


@torch.no_grad()
def encode_snapshot_plain(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    h = model(**inputs)[0].squeeze(0)
    cls = h[0]
    return torch.cat([cls, cls, cls]).cpu(), False


DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        line = line.strip()
        if line:
            DONE.add(json.loads(line)["user_id"])
    print("resume: %d users already embedded, skipping" % len(DONE), flush=True)

user_dfs = {}
for ch in pd.read_csv(CSV_PATH, usecols=["user_id", "filename", "content", "date"],
                      chunksize=200000, dtype={"user_id": str}, low_memory=False):
    ch = ch[ch["user_id"].isin(TARGET_USERS)]
    if ch.empty:
        continue
    for uid, grp in ch.groupby("user_id", sort=False):
        user_dfs.setdefault(uid, []).append(grp)
print("users found:", len(user_dfs), flush=True)

uids = [u for u in sorted(user_dfs, key=int) if u not in DONE]
if LIMIT:
    uids = uids[:LIMIT]
validated = 0
for uid in tqdm(uids, desc="Users"):
    df = pd.concat(user_dfs[uid])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    file_reprs, file_lens = [], []
    for fn, fdf in df.groupby("filename", sort=False):
        if not (fn.endswith("cpp") or fn.endswith("hpp")):
            continue
        fdf = fdf.reset_index(drop=True)
        contents = [c if isinstance(c, str) else "" for c in fdf["content"].tolist()]
        snaps = []
        for content in contents:
            if not content.strip():
                continue
            try:
                vec, used = encode_snapshot(content)
            except RuntimeError as e:
                print("snapshot failed (%s/%s): %s" % (uid, fn, str(e).splitlines()[0][:120]), flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
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
        file_reprs.append(diffs.mean(dim=0))  # [2304]
        file_lens.append(len(contents[-1].split()) if contents[-1].strip() else 0)
    if not file_reprs:
        continue
    w = torch.tensor(file_lens, dtype=torch.float32)
    emb = (torch.stack(file_reprs) * (w / w.sum()).unsqueeze(1)).sum(dim=0)
    with open(OUT, "a") as af:  # incremental append: aborts lose nothing
        af.write(json.dumps({"user_id": str(uid), "embedding": [float(x) for x in emb.tolist()]}) + "\n")
        af.flush()

print("output:", OUT, flush=True)
