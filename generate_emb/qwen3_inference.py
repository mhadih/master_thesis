import psycopg2
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm
import json
from termcolor import colored

# Path to save embeddings
output_file = "../student_embeddings/user_qwen_embeddings.jsonl"

# Load Qwen3 Embedding model
MODEL_NAME = "Qwen/Qwen2-Embedding-0.5B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,   # or float32 if CPU
    device_map="auto"            # important for low VRAM
)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(colored("### Load 'Qwen3-Embedding-8B' model is completed!", "green"))

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

# Parameters
MAX_LENGTH = 512  # Truncate long code files

def get_e5_embedding(code_snippet: str):
    """
    Converts a C++ code snippet into an E5 embedding.
    """
    # E5 models require a prefix: 'passage: ' for documents/storage 
    # or 'query: ' for search queries.
    input_text = f"passage: {code_snippet}"
    
    # Tokenize the input
    batch_dict = tokenizer(
        input_text, 
        max_length=MAX_LENGTH,
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )

    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**batch_dict)
        
        # Perform mean pooling to get a single vector for the entire snippet
        # We use the attention mask to ignore padding tokens during averaging
        mask = batch_dict['attention_mask']
        embeddings = outputs.last_hidden_state
        
        # Calculate mean pooling
        input_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        # Normalize embeddings (highly recommended for E5)
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze().cpu()


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
    embedding = get_e5_embedding(final_code)
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