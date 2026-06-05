import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, matthews_corrcoef
from pathlib import Path
import random
import os


DATA_PATH = "esm_features_3B_with_type.npz"
SAVE_DIR = "stage2_models_remap/cwe_binary"
BATCH_SIZE = 128
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 3407
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CELLWALL_IDX = 1
EXTRA_IDX = 3


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_binary_features(g_feat, n_feat, c_feat, types_idx):
    # 轻量特征：global + N端均值 + C端均值 + type one-hot
    n_mean = n_feat.mean(axis=1)
    c_mean = c_feat.mean(axis=1)
    type_oh = np.eye(3, dtype=np.float32)[types_idx]
    return np.concatenate([g_feat, n_mean, c_mean, type_oh], axis=1).astype(np.float32)


class BinaryDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BinaryMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_one_outer(x_train, y_train, x_val, y_val, model_path):
    train_ds = BinaryDataset(x_train, y_train)
    val_ds = BinaryDataset(x_val, y_val)

    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    cls_weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = cls_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BinaryMLP(x_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    weight_t = torch.tensor(cls_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=0.03)

    best_state = None
    best_score = -1.0

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        pred_all, true_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(DEVICE))
                pred_all.append(torch.argmax(logits, dim=1).cpu().numpy())
                true_all.append(yb.numpy())
        y_true = np.concatenate(true_all)
        y_pred = np.concatenate(pred_all)
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        score = 0.5 * mcc + 0.5 * f1
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    torch.save(
        {
            "state_dict": best_state,
            "input_dim": x_train.shape[1],
            "cellwall_idx": CELLWALL_IDX,
            "extra_idx": EXTRA_IDX,
        },
        model_path,
    )
    return best_score


def main():
    seed_everything(SEED)
    data = np.load(DATA_PATH, allow_pickle=True)
    g_feat = data["global_feat"]
    n_feat = data["n_feat"]
    c_feat = data["c_feat"]
    labels = data["labels"]
    folds = data["folds"]
    types_idx = data["types"]

    mask = np.isin(labels, [CELLWALL_IDX, EXTRA_IDX])
    x = make_binary_features(g_feat[mask], n_feat[mask], c_feat[mask], types_idx[mask])
    y = (labels[mask] == EXTRA_IDX).astype(np.int64)  # 0: Cellwall, 1: Extracellular
    f = folds[mask]

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Binary dataset size: {len(y)} (Cellwall={int((y==0).sum())}, Extra={int((y==1).sum())})")
    print(f"Save dir: {save_dir.resolve()}")

    scores = []
    for outer in range(5):
        tr = f != outer
        va = f == outer
        model_path = save_dir / f"cwe_outer{outer}.pt"
        score = train_one_outer(x[tr], y[tr], x[va], y[va], model_path)
        scores.append(score)
        print(f"[Outer {outer}] saved {model_path.name} score={score:.4f} n_train={int(tr.sum())} n_val={int(va.sum())}")

    print(f"Done. mean_score={np.mean(scores):.4f}")


if __name__ == "__main__":
    main()

