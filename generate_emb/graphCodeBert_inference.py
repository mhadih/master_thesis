import psycopg2
import pandas as pd
import os
import torch
import re
from transformers import RobertaTokenizer, RobertaModel
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


def get_graphcodebert_embedding(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding as the representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
        return cls_embedding.squeeze().cpu()  # return as 1D tensor

def extract_functions_from_file(content):
    FUNC_PATTERN = re.compile(
        r'([a-zA-Z_][a-zA-Z0-9_:<>~*&\s]*?)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*\{',
        re.MULTILINE
    )
    functions = []
    for match in FUNC_PATTERN.finditer(content):
        start = match.start()
        brace_count = 1
        i = content.find('{', start) + 1
        while i < len(content) and brace_count > 0:
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
            i += 1
        func_code = content[start:i]
        functions.append(func_code)
    return functions

# Load GraphCodeBERT (pretrained on code + data flow; RoBERTa architecture)
MODEL_NAME = "microsoft/graphcodebert-base"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
# NOTE: use_safetensors=True avoids torch.load, which transformers>=4.57 blocks
# for torch<2.6 (CVE-2025-32434). The official repo ships only pytorch_model.bin,
# so place a safetensors conversion next to it in the HF cache first.
model = RobertaModel.from_pretrained(MODEL_NAME, use_safetensors=True)
model.eval()  # inference mode

# Choose your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Storage for user embeddings
user_embeddings = defaultdict(list)
lengths = defaultdict(list)

# Generate embeddings
grouped = sorted_df.groupby(['user_id', 'filename'], sort=False)

for (user_id, filename), group in tqdm(grouped, desc="Processing groups", unit="group"):
    if not(filename.endswith("cpp") or filename.endswith("hpp")):
        continue
    final_code = group.iloc[-1]['content']
    if not final_code.strip():
        continue
    functions = extract_functions_from_file(final_code)
    for func in functions:
        embedding = get_graphcodebert_embedding(func)
        user_embeddings[user_id].append(embedding)
        word_count = func.count(' ') + 1
        lengths[user_id].append(word_count)

print("### Generate embeddings is completed!")


# Aggregate per user (word-count-weighted average of their function embeddings)
final_user_embeddings = {}
for user_id, emb_list in user_embeddings.items():
    weights = torch.tensor(lengths[user_id], dtype=torch.float32)
    weights = weights / weights.sum()
    final_user_embeddings[user_id] = torch.sum(
        weights[:, None] * torch.stack(emb_list), dim=0
    )  # shape: (hidden_dim=768,)

# Path to save embeddings
output_file = "user_GraphCodeBert_embeddings.jsonl"

with open(output_file, 'w') as f:
    for user_id, emb in final_user_embeddings.items():
        emb_list = emb.tolist()  # Convert tensor to native Python list
        json_line = {
            "user_id": str(user_id),   # Convert to string to avoid int64 issues
            "embedding": [float(x) for x in emb_list]
        }
        f.write(json.dumps(json_line) + '\n')

print(f"### Saved user embeddings to {output_file}")
