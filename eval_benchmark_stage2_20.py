import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from train_0408_motifcnn_remap_save20 import (
    BATCH_SIZE,
    DEVICE,
    MatrixESMDataset,
    MotifCNN_MoE,
    evaluate_and_print,
)

BENCHMARK_NPZ = "esm_features_3B_benchmark.npz"
GRAPH_PART_FASTA = "/home/zhaozhimiao/xs/moe-protein/deeppro_dataset/graphpart_set.fasta"
STAGE2_MODEL_DIR = "stage2_models_remap"


def parse_id_to_fold(fasta_path):
    id2fold = {}
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith(">"):
                continue
            parts = line[1:].split("|")
            if len(parts) < 4:
                continue
            seq_id = parts[0]
            fold = int(parts[3])
            id2fold[seq_id] = fold
    return id2fold


def remap_invalid_predictions(y_pred, t_idx, outer_membrane_idx=4, periplasmic_idx=5, extracellular_idx=3):
    idx_to_type = {0: "archaea", 1: "negative", 2: "positive"}
    remap_count = 0
    for i, t_val in enumerate(t_idx):
        type_str = idx_to_type.get(int(t_val), "")
        if type_str in ["archaea", "positive"] and y_pred[i] in [outer_membrane_idx, periplasmic_idx]:
            y_pred[i] = extracellular_idx
            remap_count += 1
    return y_pred, remap_count


def main():
    print(f"Using device: {DEVICE}")
    data = np.load(BENCHMARK_NPZ, allow_pickle=True)
    g_feat = data["global_feat"]
    n_feat = data["n_feat"]
    c_feat = data["c_feat"]
    y_true = data["labels"]
    t_idx = data["types"]
    seq_ids = data["seq_ids"]

    id2fold = parse_id_to_fold(GRAPH_PART_FASTA)
    sample_folds = []
    missing = []
    for sid in seq_ids:
        sid = str(sid)
        if sid not in id2fold:
            missing.append(sid)
            sample_folds.append(-1)
        else:
            sample_folds.append(id2fold[sid])
    sample_folds = np.array(sample_folds)
    if missing:
        raise ValueError(f"有 {len(missing)} 条 benchmark ID 在 graphpart_set.fasta 中找不到 fold 信息。")

    model_dir = Path(STAGE2_MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir.resolve()}")

    all_pred = np.full_like(y_true, fill_value=-1)
    total_remap = 0

    for outer_fold in range(5):
        fold_mask = sample_folds == outer_fold
        if fold_mask.sum() == 0:
            continue

        fold_model_paths = sorted(model_dir.glob(f"outer{outer_fold}_val*.pt"))
        if len(fold_model_paths) != 4:
            raise ValueError(
                f"outer_fold={outer_fold} 期望4个模型，实际找到 {len(fold_model_paths)} 个: "
                + ", ".join([p.name for p in fold_model_paths])
            )

        ds = MatrixESMDataset(
            g_feat[fold_mask], n_feat[fold_mask], c_feat[fold_mask], y_true[fold_mask], t_idx[fold_mask]
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        avg_probs = None
        num_classes = None
        for mp in fold_model_paths:
            ckpt = torch.load(mp, map_location=DEVICE)
            if num_classes is None:
                num_classes = int(ckpt["num_classes"])
            model = MotifCNN_MoE(num_classes).to(DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            probs = []
            with torch.no_grad():
                for g, n, c, _, t in loader:
                    logits, _ = model(g.to(DEVICE), n.to(DEVICE), c.to(DEVICE), t.to(DEVICE))
                    probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            probs = np.concatenate(probs, axis=0)
            avg_probs = probs if avg_probs is None else avg_probs + probs

        avg_probs /= len(fold_model_paths)
        pred = np.argmax(avg_probs, axis=1)
        pred, remap_n = remap_invalid_predictions(pred, t_idx[fold_mask])
        total_remap += remap_n
        all_pred[fold_mask] = pred
        print(f"[Outer Fold {outer_fold}] n={fold_mask.sum()} models=4 remap={remap_n}")

    if np.any(all_pred < 0):
        raise RuntimeError("存在未被预测到的样本，请检查 fold 映射。")

    if total_remap > 0:
        print(f"🛠️ [Trick 1 生效] 全部外层折共重映射 {total_remap} 条预测到 Extracellular。")

    label_names = [
        "CYtoplasmicMembrane",
        "Cellwall",
        "Cytoplasmic",
        "Extracellular",
        "OuterMembrane",
        "Periplasmic",
    ]
    idx_to_type = {0: "archaea", 1: "negative", 2: "positive"}
    evaluate_and_print(y_true, all_pred, t_idx, label_names, idx_to_type)


if __name__ == "__main__":
    main()
