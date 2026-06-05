import numpy as np
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_0408_motifcnn_remap_save20 import (
    BATCH_SIZE,
    DEVICE,
    MatrixESMDataset,
    MotifCNN_MoE,
)
from train_cellwall_extra_binary_oof import BinaryMLP, make_binary_features

DATA_PATH = "esm_features_3B_with_type.npz"
MULTI_DIR = "stage2_models_remap"
BIN_DIR = "stage2_models_remap/cwe_binary"
OUTPUT_DIR = "all_process"

CELLWALL_IDX = 1
EXTRA_IDX = 3
OUTER_MEMB_IDX = 4
PERI_IDX = 5

LABEL_NAMES = [
    "CytoplasmicMembrane",
    "Cellwall",
    "Cytoplasmic",
    "Extracellular",
    "OuterMembrane",
    "Periplasmic",
]

TYPE_NAMES = {0: "Archaea", 1: "Gram-Negative", 2: "Gram-Positive"}


def remap_invalid_predictions(y_pred, t_idx):
    idx_to_type = {0: "archaea", 1: "negative", 2: "positive"}
    n = 0
    for i, t_val in enumerate(t_idx):
        t_str = idx_to_type.get(int(t_val), "")
        if t_str in ["archaea", "positive"] and y_pred[i] in [OUTER_MEMB_IDX, PERI_IDX]:
            y_pred[i] = EXTRA_IDX
            n += 1
    return y_pred, n


def compute_per_label_mcc(y_true, y_pred, num_classes=6):
    result = {}
    for i in range(num_classes):
        y_t = (y_true == i).astype(int)
        y_p = (y_pred == i).astype(int)
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            result[LABEL_NAMES[i]] = np.nan
        elif y_p.sum() == 0 or y_p.sum() == len(y_p):
            result[LABEL_NAMES[i]] = np.nan
        else:
            result[LABEL_NAMES[i]] = matthews_corrcoef(y_t, y_p)
    return result


def compute_group_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "per_label_mcc": compute_per_label_mcc(y_true, y_pred),
    }


def format_val(mean, std, all_nan):
    if all_nan:
        return "-"
    return "{:.4f} +/- {:.4f}".format(mean, std)


def aggregate_fold_metrics(fold_metrics_list):
    group_order = ["overall", "archaea", "negative", "positive"]
    metric_names = ["acc", "macro_f1", "mcc"]

    result = {}
    for group in group_order:
        group_result = {}
        for metric in metric_names:
            vals = []
            for f in fold_metrics_list:
                if f[group] is not None:
                    vals.append(f[group][metric])
            vals = np.array(vals, dtype=float)
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                group_result[metric] = (0.0, 0.0)
                group_result[metric + "_all_nan"] = True
            elif len(valid) == 1:
                group_result[metric] = (float(valid[0]), 0.0)
                group_result[metric + "_all_nan"] = False
            else:
                group_result[metric] = (float(np.mean(valid)), float(np.std(valid, ddof=1)))
                group_result[metric + "_all_nan"] = False

        group_result["per_label_mcc"] = {}
        for label_name in LABEL_NAMES:
            vals = []
            for f in fold_metrics_list:
                if f[group] is not None:
                    vals.append(f[group]["per_label_mcc"].get(label_name, np.nan))
            vals = np.array(vals, dtype=float)
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                group_result["per_label_mcc"][label_name] = (0.0, 0.0, True)
            elif len(valid) == 1:
                group_result["per_label_mcc"][label_name] = (float(valid[0]), 0.0, False)
            else:
                group_result["per_label_mcc"][label_name] = (float(np.mean(valid)), float(np.std(valid, ddof=1)), False)

        result[group] = group_result

    return result


def print_aggregated_table(agg, stage_name):
    print("")
    print("=" * 130)
    print("  " + stage_name)
    print("=" * 130)

    group_order = ["overall", "archaea", "negative", "positive"]
    group_labels = ["OVERALL", "ARCHAEA", "GRAM-NEGATIVE", "GRAM-POSITIVE"]

    header = "{:<30s}".format("")
    for gl in group_labels:
        header += "{:>25s}".format(gl)
    print(header)
    print("-" * 130)

    for metric, display_name in [("acc", "Acc"), ("macro_f1", "Macro-F1"), ("mcc", "Multiclass MCC")]:
        row = "  {:<28s}".format(display_name)
        for group in group_order:
            mean, std = agg[group][metric]
            all_nan = agg[group].get(metric + "_all_nan", False)
            row += "{:>25s}".format(format_val(mean, std, all_nan))
        print(row)

    print("-" * 130)

    for label_name in LABEL_NAMES:
        row = "  {:<28s}".format(label_name)
        for group in group_order:
            mean, std, all_nan = agg[group]["per_label_mcc"][label_name]
            row += "{:>25s}".format(format_val(mean, std, all_nan))
        print(row)

    print("=" * 130)


def agg_to_dict(agg):
    out = {}
    for group in ["overall", "archaea", "negative", "positive"]:
        out[group] = {}
        for metric in ["acc", "macro_f1", "mcc"]:
            mean, std = agg[group][metric]
            all_nan = agg[group].get(metric + "_all_nan", False)
            out[group][metric] = {"mean": mean, "std": std, "all_nan": all_nan}
        out[group]["per_label_mcc"] = {}
        for label_name in LABEL_NAMES:
            mean, std, all_nan = agg[group]["per_label_mcc"][label_name]
            out[group]["per_label_mcc"][label_name] = {"mean": mean, "std": std, "all_nan": all_nan}
    return out


def plot_cm(cm, labels, title, out_file, normalize=False):
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        mat = cm / row_sum
    else:
        mat = cm

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(mat, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = "{:.2f}".format(mat[i, j]) if normalize else str(int(mat[i, j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    plt.close(fig)


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data: " + DATA_PATH)
    data = np.load(DATA_PATH, allow_pickle=True)
    g_feat = data["global_feat"]
    n_feat = data["n_feat"]
    c_feat = data["c_feat"]
    y_true = np.asarray(data["labels"])
    t_idx = np.asarray(data["types"])
    folds = np.asarray(data["folds"])

    multi_dir = Path(MULTI_DIR)
    bin_dir = Path(BIN_DIR)

    all_pred_before = np.full_like(y_true, -1)
    all_pred_after = np.full_like(y_true, -1)
    remap_total = 0

    fold_metrics_before = []
    fold_metrics_after = []

    outer_pbar = tqdm(range(5), desc="Outer folds", dynamic_ncols=True)
    for outer in outer_pbar:
        mask = folds == outer
        ds = MatrixESMDataset(g_feat[mask], n_feat[mask], c_feat[mask], y_true[mask], t_idx[mask])
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        multi_paths = sorted(multi_dir.glob("outer{}_val*.pt".format(outer)))
        if len(multi_paths) != 4:
            raise RuntimeError("outer{} model count: {}, expected 4".format(outer, len(multi_paths)))

        avg_probs = None
        for p in multi_paths:
            ck = torch.load(p, map_location=DEVICE)
            model = MotifCNN_MoE(int(ck["num_classes"])).to(DEVICE)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            probs = []
            with torch.no_grad():
                for g, n, c, _, t in loader:
                    logits, _ = model(g.to(DEVICE), n.to(DEVICE), c.to(DEVICE), t.to(DEVICE))
                    probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            probs = np.concatenate(probs, axis=0)
            avg_probs = probs if avg_probs is None else avg_probs + probs
        avg_probs /= 4
        pred = np.argmax(avg_probs, axis=1)

        pred, remap_n = remap_invalid_predictions(pred, t_idx[mask])
        remap_total += remap_n

        pred_before = pred.copy()
        all_pred_before[mask] = pred_before

        fold_true = y_true[mask]
        fold_type = t_idx[mask]

        metrics_before = {}
        metrics_before["overall"] = compute_group_metrics(fold_true, pred_before)
        for t_val, t_key in [(0, "archaea"), (1, "negative"), (2, "positive")]:
            tm = fold_type == t_val
            if tm.sum() > 0:
                metrics_before[t_key] = compute_group_metrics(fold_true[tm], pred_before[tm])
            else:
                metrics_before[t_key] = None
        fold_metrics_before.append(metrics_before)

        cwe_mask = np.isin(pred, [CELLWALL_IDX, EXTRA_IDX])
        if cwe_mask.any():
            bin_ck = torch.load(bin_dir / "cwe_outer{}.pt".format(outer), map_location=DEVICE)
            x_bin = make_binary_features(
                g_feat[mask][cwe_mask],
                n_feat[mask][cwe_mask],
                c_feat[mask][cwe_mask],
                t_idx[mask][cwe_mask],
            )
            bmodel = BinaryMLP(int(bin_ck["input_dim"])).to(DEVICE)
            bmodel.load_state_dict(bin_ck["state_dict"])
            bmodel.eval()
            with torch.no_grad():
                logits = bmodel(torch.tensor(x_bin, dtype=torch.float32, device=DEVICE))
                yb = torch.argmax(logits, dim=1).cpu().numpy()
            refined = np.where(yb == 0, CELLWALL_IDX, EXTRA_IDX)
            pred[cwe_mask] = refined

        pred_after = pred.copy()
        all_pred_after[mask] = pred_after

        metrics_after = {}
        metrics_after["overall"] = compute_group_metrics(fold_true, pred_after)
        for t_val, t_key in [(0, "archaea"), (1, "negative"), (2, "positive")]:
            tm = fold_type == t_val
            if tm.sum() > 0:
                metrics_after[t_key] = compute_group_metrics(fold_true[tm], pred_after[tm])
            else:
                metrics_after[t_key] = None
        fold_metrics_after.append(metrics_after)

        outer_pbar.set_postfix(samples=int(mask.sum()), remap_total=remap_total)

    if np.any(all_pred_before < 0) or np.any(all_pred_after < 0):
        raise RuntimeError("Some samples not predicted")

    print("")
    print("Using device: {}".format(DEVICE))
    print("Biological remap count: {}".format(remap_total))

    agg_before = aggregate_fold_metrics(fold_metrics_before)
    agg_after = aggregate_fold_metrics(fold_metrics_after)

    print_aggregated_table(agg_before, "STAGE 1: BEFORE Binary Refinement (6-class + Biological Remap)")
    print_aggregated_table(agg_after, "STAGE 2: AFTER Binary Refinement (Full Cascade: 6-class + Remap + BinaryMLP)")

    np.savez(
        output_dir / "oof_predictions.npz",
        y_true=y_true,
        pred_before=all_pred_before,
        pred_after=all_pred_after,
        types=t_idx,
        folds=folds,
        label_names=np.array(LABEL_NAMES),
    )
    print("")
    print("[Saved] Predictions => {}".format(output_dir / "oof_predictions.npz"))

    with open(str(output_dir / "metrics_before_binary.json"), "w") as f:
        json.dump(agg_to_dict(agg_before), f, indent=2)
    with open(str(output_dir / "metrics_after_binary.json"), "w") as f:
        json.dump(agg_to_dict(agg_after), f, indent=2)
    print("[Saved] Metrics JSON => {}/".format(output_dir))

    cm_before = confusion_matrix(y_true, all_pred_before, labels=list(range(6)))
    cm_after = confusion_matrix(y_true, all_pred_after, labels=list(range(6)))

    plot_cm(cm_before, LABEL_NAMES,
            "OOF - BEFORE Binary (Count)",
            output_dir / "cm_before_count.png", normalize=False)
    plot_cm(cm_before, LABEL_NAMES,
            "OOF - BEFORE Binary (Row-Norm)",
            output_dir / "cm_before_norm.png", normalize=True)
    plot_cm(cm_after, LABEL_NAMES,
            "OOF - AFTER Binary (Count)",
            output_dir / "cm_after_count.png", normalize=False)
    plot_cm(cm_after, LABEL_NAMES,
            "OOF - AFTER Binary (Row-Norm)",
            output_dir / "cm_after_norm.png", normalize=True)
    print("[Saved] 4 Confusion Matrix PNGs => {}/".format(output_dir))

    for stage, cm in [("BEFORE Binary", cm_before), ("AFTER Binary", cm_after)]:
        cw_row = cm[CELLWALL_IDX]
        ex_row = cm[EXTRA_IDX]
        print("")
        print("[{} - Cellwall/Extracellular details]".format(stage))
        print("  Cellwall row: {}".format(cw_row.tolist()))
        print("  Extracellular row: {}".format(ex_row.tolist()))
        print("  Cellwall recall={:.4f}  Extracellular recall={:.4f}".format(
            cw_row[CELLWALL_IDX]/max(cw_row.sum(),1),
            ex_row[EXTRA_IDX]/max(ex_row.sum(),1)))
        print("  Cellwall->Extracellular={}  Extracellular->Cellwall={}".format(
            int(cm[CELLWALL_IDX, EXTRA_IDX]),
            int(cm[EXTRA_IDX, CELLWALL_IDX])))


if __name__ == "__main__":
    main()
