import psycopg2
import pandas as pd
import os
import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm
import json
from termcolor import colored

# Path to save embeddings
output_file = "../student_embeddings/user_jina_embeddings.jsonl"

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
print(colored("### Structured data (dataframe) is ready!", "green"))

# Load jina model
MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(colored("### Load 'jina embeddings' model is completed!", "green"))

# Parameters
MAX_LENGTH = 256  # Truncate long code files

def get_jina_embedding(code_snippet: str):
    # Tokenize input
    inputs = tokenizer(
        code_snippet,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH  # Jina supports long contexts
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # The Jina embedding model outputs embeddings directly
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

    return embedding.cpu()


# Storage for user embeddings
user_embeddings = defaultdict(list)
lengths = defaultdict(list)

# Generate embeddings
# Assuming sorted_df is already a DataFrame
grouped = sorted_df.groupby(['user_id', 'filename'], sort=False)

for (user_id, filename), group in tqdm(grouped, desc="Processing groups", unit="group"):
    if not(filename.endswith("cpp")):
        continue
    final_code = group.iloc[-1]['content']
    if not final_code.strip():
        continue
    embedding = get_jina_embedding(final_code)
    user_embeddings[user_id].append(embedding)
    word_count = final_code.count(' ') + 1
    lengths[user_id].append(word_count)

print(colored("### Generate embeddings is completed!", "green"))

# Aggregate per user (weighted sum of their code embeddings)
final_user_embeddings = {}
for user_id, emb_list in user_embeddings.items():
    weights = torch.tensor(lengths[user_id], dtype=torch.float32)
    weights = weights / weights.sum()
    final_user_embeddings[user_id] = torch.sum(weights[:, None] * torch.stack(emb_list), dim=0)  # shape: (hidden_dim=768,)

with open(output_file, 'w') as f:
    for user_id, emb in final_user_embeddings.items():
        emb_list = emb.tolist()  # Convert tensor to native Python list
        json_line = {
            "user_id": str(user_id),   # Convert to string to avoid int64 issues
            "embedding": [float(x) for x in emb_list]  # Ensure nahted sum of their code emtive float
        }
        f.write(json.dumps(json_line) + '\n')

print(colored(f"### Saved user embeddings to {output_file}", "green"))