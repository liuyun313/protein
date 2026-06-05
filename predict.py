import argparse, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Model definitions
# ============================================================================

class SwiGLUExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MoeLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([SwiGLUExpert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.shared_expert = SwiGLUExpert(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x_norm = self.layer_norm(x)
        shared_out = self.shared_expert(x_norm)
        logits = self.router(x_norm)
        gates = torch.softmax(logits, dim=-1)
        topk_val, topk_idx = torch.topk(gates, self.top_k, dim=-1)
        mask = torch.zeros_like(gates).scatter_(1, topk_idx, 1.0)
        gates = gates * mask
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-9)
        expert_outputs = torch.stack([exp(x_norm) for exp in self.experts], dim=1)
        moe_out = torch.einsum("bed,be->bd", expert_outputs, gates)
        return residual + self.dropout(shared_out + moe_out), torch.tensor(0.0, device=x.device)


class MotifCNN_MoE(nn.Module):
    def __init__(self, num_classes, num_types=3, moe_hidden_dim=2048,
                 num_experts=4, top_k=2, type_emb_dim=64):
        super().__init__()
        self.g_proj = nn.Sequential(
            nn.Linear(2560, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2)
        )
        self.n_cnn = nn.Sequential(
            nn.Conv1d(2560, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.c_cnn = nn.Sequential(
            nn.Conv1d(2560, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.t_emb = nn.Embedding(num_types, type_emb_dim)
        fusion_dim = 512 + 256 + 256 + type_emb_dim
        self.fusion = nn.Linear(fusion_dim, 1024)
        self.moe = MoeLayer(1024, moe_hidden_dim, num_experts=num_experts, top_k=top_k)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, num_classes)
        )

    def forward(self, g, n, c, t):
        g_out = self.g_proj(g)
        n_out = self.n_cnn(n).squeeze(-1)
        c_out = self.c_cnn(c).squeeze(-1)
        t_out = self.t_emb(t)
        fused = torch.cat([g_out, n_out, c_out, t_out], dim=-1)
        h = self.fusion(fused)
        h, _ = self.moe(h)
        return self.classifier(h)


class BinaryMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Feature extraction (matches github/features/feature.py)
# ============================================================================

LABEL_NAMES = [
    "CytoplasmicMembrane", "Cellwall", "Cytoplasmic",
    "Extracellular", "OuterMembrane", "Periplasmic",
]
CELLWALL_IDX = 1
EXTRA_IDX = 3
OUTER_MEMB_IDX = 4
PERI_IDX = 5

MAX_SEQ_LEN = 1022
TERM_LEN = 50


def parse_fasta(fasta_path):
    records = []
    current_id = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            records.append((current_id, "".join(current_seq)))
    return records


def extract_features(sequences, device):
    """
    Same extraction logic as github/features/feature.py:
      - ESM-2 3B FP16 on GPU
      - Max 1022aa (511+511 split)
      - N-terminal 50 residues, C-terminal 50 residues as separate matrices
    """
    import esm

    print("Loading ESM-2 3B...")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.half().to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    g_list, n_list, c_list = [], [], []

    for seq_id, seq in sequences:
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:511] + seq[-511:]

        data = [(seq_id, seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch_tokens, repr_layers=[36])
            valid_tokens = out["representations"][36][0][1:-1]  # strip BOS/EOS
            seq_len = valid_tokens.size(0)

            g_feat = valid_tokens.mean(dim=0).cpu().numpy().astype(np.float16)

            n_mat = torch.zeros((TERM_LEN, 2560), dtype=torch.float16, device=device)
            n_len = min(TERM_LEN, seq_len)
            n_mat[:n_len] = valid_tokens[:n_len]

            c_mat = torch.zeros((TERM_LEN, 2560), dtype=torch.float16, device=device)
            c_mat[-n_len:] = valid_tokens[-n_len:]

        g_list.append(g_feat)
        n_list.append(n_mat.cpu().numpy())
        c_list.append(c_mat.cpu().numpy())

    g_feat = np.stack(g_list).astype(np.float32)
    n_feat = np.stack(n_list).astype(np.float32)
    c_feat = np.stack(c_list).astype(np.float32)

    return g_feat, n_feat, c_feat


def make_binary_features(g_feat, n_feat, c_feat, type_idx):
    n_mean = n_feat.mean(axis=1)
    c_mean = c_feat.mean(axis=1)
    type_oh = np.eye(3, dtype=np.float32)[int(type_idx)].reshape(1, -1).repeat(len(g_feat), axis=0)
    return np.concatenate([g_feat, n_mean, c_mean, type_oh], axis=1).astype(np.float32)


def remap_invalid(pred, t_idx_arr):
    for i, t in enumerate(t_idx_arr):
        if t in [0, 2] and pred[i] in [OUTER_MEMB_IDX, PERI_IDX]:
            pred[i] = EXTRA_IDX
    return pred


# ============================================================================
# Main prediction pipeline
# ============================================================================

def predict(sequences, models_dir, device, num_experts=4, top_k=2, type_emb_dim=64):
    weights_dir = os.path.join(models_dir, "models_weights")
    bin_dir = os.path.join(weights_dir, "cwe_binary")

    # Step 1: ESM feature extraction
    g_feat, n_feat, c_feat = extract_features(sequences, device)

    g_t = torch.tensor(g_feat, dtype=torch.float32).to(device)
    n_t = torch.tensor(n_feat, dtype=torch.float32).transpose(1, 2).to(device)
    c_t = torch.tensor(c_feat, dtype=torch.float32).transpose(1, 2).to(device)
    num_samples = len(sequences)

    # Step 2: 20-model ensemble
    all_probs = None
    loaded_count = 0
    for outer in range(5):
        dev_indices = [f for f in range(5) if f != outer]
        for val_f in dev_indices:
            ckpt_path = os.path.join(weights_dir, f"outer{outer}_val{val_f}.pt")
            if not os.path.exists(ckpt_path):
                continue
            ck = torch.load(ckpt_path, map_location=device)
            model = MotifCNN_MoE(
                int(ck["num_classes"]), num_types=3,
                num_experts=num_experts, top_k=top_k,
                type_emb_dim=type_emb_dim
            ).to(device)
            try:
                model.load_state_dict(ck["state_dict"])
            except RuntimeError:
                model = MotifCNN_MoE(
                    int(ck["num_classes"]), num_types=3,
                    num_experts=8, top_k=2, type_emb_dim=64
                ).to(device)
                model.load_state_dict(ck["state_dict"])
            model.eval()
            with torch.no_grad():
                # Predict with all 3 types, average the logits
                logits_avg = None
                for t_val in range(3):
                    t_tensor = torch.full((num_samples,), t_val, dtype=torch.long, device=device)
                    logits = model(g_t, n_t, c_t, t_tensor)
                    if logits_avg is None:
                        logits_avg = logits
                    else:
                        logits_avg += logits
                logits_avg /= 3.0
                probs = torch.softmax(logits_avg, dim=1).cpu().numpy()
            all_probs = probs if all_probs is None else all_probs + probs
            loaded_count += 1

    if all_probs is None:
        raise RuntimeError("No models found!")

    all_probs /= loaded_count
    pred_6class = np.argmax(all_probs, axis=1)

    # Step 3: Infer species type via majority vote & apply biological remap
    type_votes = {0: 0, 1: 0, 2: 0}
    type_preds = {}
    for t_val in range(3):
        tp = pred_6class.copy()
        tp = remap_invalid(tp, np.full(num_samples, t_val))
        type_preds[t_val] = tp
        conf = all_probs[np.arange(num_samples), tp].mean()
        type_votes[t_val] = conf

    best_type = max(type_votes, key=type_votes.get)
    pred = type_preds[best_type]

    # Step 4: Binary refinement for Cellwall/Extracellular
    cwe_mask = np.isin(pred, [CELLWALL_IDX, EXTRA_IDX])
    if cwe_mask.any():
        bin_probs = None
        bin_loaded = 0
        for outer in range(5):
            bin_path = os.path.join(bin_dir, f"cwe_outer{outer}.pt")
            if not os.path.exists(bin_path):
                continue
            bin_ck = torch.load(bin_path, map_location=device)
            x_bin = make_binary_features(
                g_feat[cwe_mask], n_feat[cwe_mask], c_feat[cwe_mask], best_type
            )
            bmodel = BinaryMLP(int(bin_ck["input_dim"])).to(device)
            bmodel.load_state_dict(bin_ck["state_dict"])
            bmodel.eval()
            with torch.no_grad():
                logits = bmodel(torch.tensor(x_bin, dtype=torch.float32, device=device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            bin_probs = probs if bin_probs is None else bin_probs + probs
            bin_loaded += 1

        if bin_probs is not None:
            bin_probs /= bin_loaded
            yb = np.argmax(bin_probs, axis=1)
            refined = np.where(yb == 0, CELLWALL_IDX, EXTRA_IDX)
            pred[cwe_mask] = refined

    return [LABEL_NAMES[int(p)] for p in pred]


def main():
    parser = argparse.ArgumentParser(description="Protein Subcellular Localization Prediction")
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences = parse_fasta(args.fasta)
    results = predict(sequences, args.models_dir, device,
                      num_experts=args.experts,
                      top_k=args.top_k,
                      type_emb_dim=args.emb_dim)

    lines = [f"{seq_id}\t{loc}" for (seq_id, _), loc in zip(sequences, results)]
    output_text = "\n".join(lines)
    print(output_text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)


if __name__ == "__main__":
    main()
