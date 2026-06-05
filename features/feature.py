import torch
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_custom_fasta(fasta_path):
    records = []
    current = None
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current is not None: records.append(current)
                parts = line[1:].split("|")
                current = {"ID": parts[0], "Location": parts[1], "Type": parts[2], "Fold": int(parts[3]), "Sequence": ""}
            else:
                current["Sequence"] += line
        if current is not None: records.append(current)
    return pd.DataFrame(records)

print("🚀 加载 ESM-2 3B 模型 (FP16 模式)...")
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model = model.half().to(device).eval()
batch_converter = alphabet.get_batch_converter()

df = parse_custom_fasta("/home/zhaozhimiao/xs/moe-protein/deeppro_dataset/graphpart_set.fasta")

# 1. 映射 Location 到 Index
locations = sorted(df["Location"].unique().tolist())
loc_to_idx = {l: i for i, l in enumerate(locations)}
print(f"标签映射: {loc_to_idx}")

# 2. 映射 Type 到 Index
# 2. 映射 Type 到 Index
types = sorted(df["Type"].unique().tolist())
type_to_idx = {t: i for i, t in enumerate(types)}
print(f"物种映射: {type_to_idx}")

MAX_SEQ_LEN = 1022 
TER_LEN = 50 

global_features, n_features, c_features = [], [], []
labels, folds, seq_ids, type_indices = [], [], [], []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Matrix Features"):
    seq_id, seq = row["ID"], row["Sequence"]
    fold, loc_str, type_str = int(row["Fold"]), row["Location"], row["Type"]

    if len(seq) > MAX_SEQ_LEN:
        seq = seq[:511] + seq[-511:]

    data = [(seq_id, seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(batch_tokens, repr_layers=[36])
        valid_tokens = out["representations"][36][0][1:-1] # [SeqLen, 2560]
        seq_len = valid_tokens.size(0)
        
        g_feat = valid_tokens.mean(dim=0).cpu().numpy().astype(np.float16)
        
        n_mat = torch.zeros((TER_LEN, 2560), dtype=torch.float16, device=device)
        n_len = min(TER_LEN, seq_len)
        n_mat[:n_len] = valid_tokens[:n_len]
        
        c_mat = torch.zeros((TER_LEN, 2560), dtype=torch.float16, device=device)
        c_mat[-n_len:] = valid_tokens[-n_len:]
        
        global_features.append(g_feat)
        n_features.append(n_mat.cpu().numpy())
        c_features.append(c_mat.cpu().numpy())

    seq_ids.append(seq_id)
    folds.append(fold)
    labels.append(loc_to_idx[loc_str])
    type_indices.append(type_to_idx[type_str])

    if idx % 50 == 0:
        torch.cuda.empty_cache()

save_path = "esm_features_3B_with_type.npz"
np.savez(
    save_path,
    global_feat=np.stack(global_features),
    n_feat=np.stack(n_features),
    c_feat=np.stack(c_features),
    labels=np.array(labels, dtype=np.int64),
    folds=np.array(folds, dtype=np.int32),
    types=np.array(type_indices, dtype=np.int64) # 存入物种标签
)
print(f"\n✅ 特征提取完毕: {save_path}")