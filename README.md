# 蛋白质亚细胞定位预测

---

## 快速开始

### 1. 创建 Conda 环境

```bash
conda create -n protein python=3.10 -y
conda activate protein
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载特征和模型权重

从 Google Drive 下载 `features/` 和 `models_weights/` 文件夹，放到项目根目录：

```
https://drive.google.com/drive/folders/1lb2HhCWwEM3bZcjIQw3MAT81DioKJTje?usp=sharing
```

目录结构：

```
protein/
├── features/
└── models_weights/
    ├── outer{0-4}_val{*}.pt                 
    └── cwe_binary/cwe_outer{0-4}.pt        
```

### 4. 运行预测

```bash
python predict.py \
    --fasta your_sequences.fasta \
    --models_dir . \
    --output results.txt
```

### 5. 输出格式

两列制表符分隔：蛋白质ID + 定位标签

```
P12345  Cytoplasmic
Q9Y2Y9  Cellwall
A0A0B4  Extracellular
```
---

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|:---:|
| `--fasta` | 输入 FASTA 文件路径 | 必填 |
| `--models_dir` | 包含 `models_weights/` 的根目录 | 必填 |
| `--output` | 输出文件（不指定则打印到终端） | — |
---

---

## 目录结构

```
protein/
├── predict.py                              # 推理脚本（直接运行此文件）
├── train_0408_motifcnn_remap_save20.py      # 6分类 MotifCNN-MoE 训练
├── train_cellwall_extra_binary_oof.py       # 细胞壁/胞外二分类器训练
├── eval_oof_cascade_cwe2.py                 # 留一交叉验证级联评估
├── eval_benchmark_stage2_20.py              # Benchmark 评估
│
├── features/
│   ├── feature.py                           # ESM 特征提取脚本（参考）
│   └── graphpart_set.fasta                  # 训练参考 FASTA
│
├── models_weights/
│   ├── outer{0-4}_val{*}.pt                 # 20个 MoE 模型 (5折 × 4内层)
│   └── cwe_binary/cwe_outer{0-4}.pt         # 5个二分类器
│
└── README.md
```