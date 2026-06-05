import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import copy
import warnings
import itertools
import random
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# =========================================================
# 配置参数
# =========================================================
DATA_PATH = "esm_features_3B_with_type.npz" # ⚠️ 确保路径指向你最新的 2D 矩阵特征文件
BATCH_SIZE = 64
EPOCHS = 100 
SEARCH_EPOCHS = 30
LABEL_SMOOTHING = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAGE2_SAVE_DIR = "stage2_models_remap"

print(f"Using device: {DEVICE}")

# =========================================================
# 0. 固定随机种子
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================
# 1. 矩阵数据集定义 (MatrixESMDataset)
# =========================================================
class MatrixESMDataset(Dataset):
    def __init__(self, g_feat, n_feat, c_feat, labels, types):
        self.g_feat = torch.tensor(g_feat, dtype=torch.float32)
        # Conv1d 要求维度为 [Batch, Channel, Length]，原特征是 [Length(50), Channel(2560)]，需转置
        self.n_feat = torch.tensor(n_feat, dtype=torch.float32).transpose(1, 2)
        self.c_feat = torch.tensor(c_feat, dtype=torch.float32).transpose(1, 2)
        
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.types = torch.tensor(types, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.g_feat[idx], self.n_feat[idx], self.c_feat[idx], self.labels[idx], self.types[idx]

# =========================================================
# 2. 核心架构：Motif CNN + Type Prior + MoE
# =========================================================
class SwiGLUExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MoeLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=8, top_k=2):
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
        if self.training:
            logits += torch.randn_like(logits) * 0.05
            
        gates = torch.softmax(logits, dim=-1)
        topk_val, topk_idx = torch.topk(gates, self.top_k, dim=-1)
        mask = torch.zeros_like(gates).scatter_(1, topk_idx, 1.0)
        gates = (gates * mask)
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-9)

        expert_outputs = torch.stack([exp(x_norm) for exp in self.experts], dim=1)
        moe_out = torch.einsum("bed,be->bd", expert_outputs, gates)
        
        importance = gates.mean(dim=0)
        load = (gates > 0).float().mean(dim=0)
        aux_loss = (importance * load).sum() * (self.num_experts ** 2)
        
        return residual + self.dropout(shared_out + moe_out), aux_loss

class MotifCNN_MoE(nn.Module):
    def __init__(self, num_classes, num_types=3, moe_hidden_dim=2048):
        super().__init__()
        # 全局特征降维
        self.g_proj = nn.Sequential(
            nn.Linear(2560, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2)
        )
        
        # 信号肽/N端 Motif 扫描器
        self.n_cnn = nn.Sequential(
            nn.Conv1d(2560, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 锚定信号/C端 Motif 扫描器
        self.c_cnn = nn.Sequential(
            nn.Conv1d(2560, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 物种先验嵌入
        self.t_emb = nn.Embedding(num_types, 64)
        
        # 融合与专家路由
        fusion_dim = 512 + 256 + 256 + 64 # = 1088
        self.fusion = nn.Linear(fusion_dim, 1024)
        self.moe = MoeLayer(1024, moe_hidden_dim, num_experts=8, top_k=2)
        
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
        h, aux_loss = self.moe(h)
        return self.classifier(h), aux_loss

# =========================================================
# 3. 核心评估函数
# =========================================================
def evaluate_and_print(y_true, y_pred, types, label_names, idx_to_type):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print("\n" + "="*50)
    print(f"GLOBAL METRICS: Acc={acc:.4f} | Macro-F1={f1:.4f} | MCC={mcc:.4f}")
    print("="*50)
    
    print("\n[PER-LABEL MCC]")
    for i, name in enumerate(label_names):
        y_t_bin = (y_true == i).astype(int)
        y_p_bin = (y_pred == i).astype(int)
        l_mcc = matthews_corrcoef(y_t_bin, y_p_bin)
        print(f"  {name:20s} : {l_mcc:.4f}")

    unique_types = np.unique(types)
    for t_idx in unique_types:
        idx = [i for i, val in enumerate(types) if val == t_idx]
        if len(idx) == 0: continue
        
        yt_sub, yp_sub = y_true[idx], y_pred[idx]
        t_acc = accuracy_score(yt_sub, yp_sub)
        t_f1 = f1_score(yt_sub, yp_sub, average='macro')
        t_mcc = matthews_corrcoef(yt_sub, yp_sub)
        
        type_name = idx_to_type.get(t_idx, f"Type_{t_idx}")
        print(f"\n>>> TYPE: {type_name} (n={len(idx)})")
        print(f"    Metrics: Acc={t_acc:.4f}, Macro-F1={t_f1:.4f}, MCC={t_mcc:.4f}")

# =========================================================
# 4. 单模型训练流程 (多输入 + Data Augmentation)
# =========================================================
def train_one_model(train_loader, val_loader, num_classes, lr, aux_weight, epochs):
    model = MotifCNN_MoE(num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3) # 增大了CNN的权重衰减防过拟合
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    best_state = None
    best_val_mcc = -1
    
    for epoch in range(epochs):
        model.train()
        for g, n, c, y, t in train_loader:
            g, n, c = g.to(DEVICE), n.to(DEVICE), c.to(DEVICE)
            y, t = y.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            
            # 特征级数据增强
            if torch.rand(1).item() < 0.5:
                g += torch.randn_like(g) * 0.02
                n += torch.randn_like(n) * 0.02
                c += torch.randn_like(c) * 0.02
                
            # MixUp 增强 (针对连续特征进行混合，类别标签插值)
            if torch.rand(1).item() < 0.3:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(g.size(0)).to(DEVICE)
                
                g_mix = lam * g + (1 - lam) * g[idx]
                n_mix = lam * n + (1 - lam) * n[idx]
                c_mix = lam * c + (1 - lam) * c[idx]
                y_a, y_b = y, y[idx]
                # MixUp 时采用 dominant 类型
                t_mix = t if lam > 0.5 else t[idx] 
                
                logits, aux_loss = model(g_mix, n_mix, c_mix, t_mix)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b) + aux_weight * aux_loss
            else:
                logits, aux_loss = model(g, n, c, t)
                loss = criterion(logits, y) + aux_weight * aux_loss
                
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for g, n, c, y, t in val_loader:
                logits, _ = model(g.to(DEVICE), n.to(DEVICE), c.to(DEVICE), t.to(DEVICE))
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                all_trues.append(y.numpy())
        
        current_mcc = matthews_corrcoef(np.concatenate(all_trues), np.concatenate(all_preds))
        if current_mcc > best_val_mcc:
            best_val_mcc = current_mcc
            best_state = copy.deepcopy(model.state_dict())
            
    return best_state, best_val_mcc

# =========================================================
# 5. 主程序
# =========================================================
def main():
    print(f"📥 加载矩阵特征文件: {DATA_PATH} ...")
    data = np.load(DATA_PATH, allow_pickle=True)
    g_feat = data['global_feat']
    n_feat = data['n_feat']
    c_feat = data['c_feat']
    all_labels = data['labels']
    all_folds = data['folds']
    all_types = data['types']
    
    num_classes = len(np.unique(all_labels))
    
    # 强制硬编码映射关系（根据之前 Fasta 处理的 sorted() 规则）
    idx_to_loc = {
        0: 'CYtoplasmicMembrane', 1: 'Cellwall', 2: 'Cytoplasmic',
        3: 'Extracellular', 4: 'OuterMembrane', 5: 'Periplasmic'
    }
    label_names = [idx_to_loc[i] for i in range(num_classes)]
    
    idx_to_type = {0: 'archaea', 1: 'negative', 2: 'positive'}
    
    # 获取掩码所需要的索引
    outer_membrane_idx = 4
    periplasmic_idx = 5
    cellwall_idx = 1

    # ---------------------------------------------------------
    # 阶段 1：超参数搜索
    # ---------------------------------------------------------
    print("\n" + "*"*80)
    print("🔍 阶段 1: 开始最优超参数与随机种子搜索 (基于 Motif CNN) ...")
    print("*"*80)
    
    param_grid = {
        'seed': [42, 3407], 
        'lr': [1e-4, 3e-4], 
        'moe_aux_weight': [1e-3, 5e-3]
    }
    
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    
    best_params = None
    global_best_mcc = -1
    
    search_train_mask = (all_folds >= 2)
    search_val_mask = (all_folds == 1)
    
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        seed_everything(params['seed'])
        
        train_dataset = MatrixESMDataset(g_feat[search_train_mask], n_feat[search_train_mask], c_feat[search_train_mask], all_labels[search_train_mask], all_types[search_train_mask])
        val_dataset = MatrixESMDataset(g_feat[search_val_mask], n_feat[search_val_mask], c_feat[search_val_mask], all_labels[search_val_mask], all_types[search_val_mask])
        
        y_train_idx = all_labels[search_train_mask]
        class_counts = np.bincount(y_train_idx, minlength=num_classes)
        class_weights = 1.0 / np.maximum(class_counts, 1.0)
        
        class_weights[cellwall_idx] *= 3.0
        class_weights[outer_membrane_idx] *= 1.5
            
        sample_weights = class_weights[y_train_idx]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        s_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        s_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        _, val_mcc = train_one_model(s_train_loader, s_val_loader, num_classes, lr=params['lr'], aux_weight=params['moe_aux_weight'], epochs=SEARCH_EPOCHS)
        
        print(f"[{i+1}/{len(combinations)}] 参数: {params} | 验证集 MCC: {val_mcc:.4f}")
        
        if val_mcc > global_best_mcc:
            global_best_mcc = val_mcc
            best_params = params

    print("\n" + "🌟"*30)
    print(f"🏆 搜索完成！选定最优参数组合: {best_params} (MCC: {global_best_mcc:.4f})")
    print("🌟"*30)

    # ---------------------------------------------------------
    # 阶段 2：完整的 5-Fold Nested CV
    # ---------------------------------------------------------
    print(f"\n🚀 阶段 2: 开始 5-Fold Nested CV (Motif CNN + MoE)")
    print("="*80)
    
    seed_everything(best_params['seed'])
    FINAL_LR = best_params['lr']
    FINAL_AUX = best_params['moe_aux_weight']
    save_dir = Path(STAGE2_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Stage2 模型将保存到: {save_dir.resolve()}")

    final_outer_metrics = {"acc": [], "macro_f1": [], "mcc": []}

    for outer_fold in range(5):
        print(f"\n\n[Outer Fold {outer_fold+1}/5] 测试中...")
        
        test_mask = (all_folds == outer_fold)
        test_dataset = MatrixESMDataset(g_feat[test_mask], n_feat[test_mask], c_feat[test_mask], all_labels[test_mask], all_types[test_mask])
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        y_test_idx = all_labels[test_mask]
        t_test_idx = all_types[test_mask]
        
        inner_model_states = []
        dev_indices = [f for f in range(5) if f != outer_fold]
        
        for val_f in dev_indices:
            print(f"  -> Inner Fold 验证集设为 Fold {val_f}")
            val_mask = (all_folds == val_f)
            train_mask = (all_folds != outer_fold) & (all_folds != val_f)
            
            train_dataset = MatrixESMDataset(g_feat[train_mask], n_feat[train_mask], c_feat[train_mask], all_labels[train_mask], all_types[train_mask])
            val_dataset = MatrixESMDataset(g_feat[val_mask], n_feat[val_mask], c_feat[val_mask], all_labels[val_mask], all_types[val_mask])
            
            y_train_idx = all_labels[train_mask]
            class_counts = np.bincount(y_train_idx, minlength=num_classes)
            class_weights = 1.0 / np.maximum(class_counts, 1.0)
            
            class_weights[cellwall_idx] *= 3.0       
            class_weights[outer_membrane_idx] *= 1.5 
                
            sample_weights = class_weights[y_train_idx]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            best_s, _ = train_one_model(train_loader, val_loader, num_classes, lr=FINAL_LR, aux_weight=FINAL_AUX, epochs=EPOCHS)
            inner_model_states.append(best_s)
            model_path = save_dir / f"outer{outer_fold}_val{val_f}.pt"
            torch.save(
                {
                    "state_dict": best_s,
                    "outer_fold": outer_fold,
                    "val_fold": val_f,
                    "num_classes": num_classes,
                    "best_params": best_params,
                    "idx_to_loc": idx_to_loc,
                    "idx_to_type": idx_to_type,
                },
                model_path,
            )
            print(f"    ✅ 已保存: {model_path.name}")

        print(f"\n🔮 对 Outer Fold {outer_fold} 进行 4-Model Ensemble 预测...")
        avg_probs = np.zeros((len(y_test_idx), num_classes))
        for state in inner_model_states:
            model = MotifCNN_MoE(num_classes).to(DEVICE)
            model.load_state_dict(state)
            model.eval()
            probs = []
            with torch.no_grad():
                for g, n, c, _, t in test_loader:
                    logits, _ = model(g.to(DEVICE), n.to(DEVICE), c.to(DEVICE), t.to(DEVICE))
                    probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            avg_probs += np.concatenate(probs, axis=0)
        
        avg_probs /= len(inner_model_states)

        # 💥 TRICK 1 (论文风格): 对不合理类别做 remap 到 Extracellular
        remap_count = 0
        extracellular_idx = 3
        y_pred_idx = np.argmax(avg_probs, axis=1)
        for i, t_val in enumerate(t_test_idx):
            type_str = idx_to_type.get(t_val, "")
            if type_str in ["archaea", "positive"] and y_pred_idx[i] in [outer_membrane_idx, periplasmic_idx]:
                y_pred_idx[i] = extracellular_idx
                remap_count += 1

        if remap_count > 0:
            print(f"🛠️ [Trick 1 生效] 已将 {remap_count} 个不合理预测重映射到 Extracellular。")

        evaluate_and_print(y_test_idx, y_pred_idx, t_test_idx, label_names, idx_to_type)

        final_outer_metrics["acc"].append(accuracy_score(y_test_idx, y_pred_idx))
        final_outer_metrics["macro_f1"].append(f1_score(y_test_idx, y_pred_idx, average='macro'))
        final_outer_metrics["mcc"].append(matthews_corrcoef(y_test_idx, y_pred_idx))

    print("\n" + "!"*60)
    print("🏆 5-FOLD NESTED CV FINAL RESULTS (Motif CNN) 🏆")
    print("!"*60)
    for k, v in final_outer_metrics.items():
        print(f"Global {k.upper()}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    print("="*60)

if __name__ == "__main__":
    main()