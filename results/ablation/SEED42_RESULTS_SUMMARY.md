# Final BITENET Ablation Results — Seed 42

论文主消融统一使用 `seed=42`。

其余 seed 的原始运行结果仅作为备用，统一保存在各实验的 `other_seeds_backup/` 中。

## 01_pretrained_representation: Pre-trained Representation Ablation

| Variant | AUROC | AUPRC | MCC | Sensitivity | Specificity | Precision | F1 | Threshold | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ESM-2 + ESM-2 | 0.857447 | 0.398060 | 0.381948 | 0.411422 | 0.966626 | 0.420488 | 0.415906 | 0.590000 | 27 |
| ESM-2 + Learnable | 0.851587 | 0.379305 | 0.367252 | 0.408508 | 0.963538 | 0.397392 | 0.402874 | 0.530000 | 18 |
| Learnable + ESM-2 | 0.719779 | 0.145226 | 0.160949 | 0.270979 | 0.925430 | 0.176203 | 0.213548 | 0.510000 | 34 |
| Learnable + Learnable | 0.707204 | 0.144591 | 0.154129 | 0.477855 | 0.799101 | 0.122810 | 0.195401 | 0.380000 | 35 |

## 02_branch_design: Protein Prior / Interaction-refined Branch Ablation

| Variant | AUROC | AUPRC | MCC | Sensitivity | Specificity | Precision | F1 | Threshold | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full model / Ours | 0.857191 | 0.397242 | 0.383659 | 0.411422 | 0.967037 | 0.423515 | 0.417381 | 0.580000 | 27 |
| w/o interaction-refined | 0.834951 | 0.366464 | 0.354151 | 0.477855 | 0.942821 | 0.329715 | 0.390197 | 0.350000 | 8 |
| w/o protein prior | 0.852149 | 0.381340 | 0.363331 | 0.492424 | 0.942306 | 0.334389 | 0.398303 | 0.510000 | 14 |

## 03_cross_mamba_order: Mamba-Cross Ordering Ablation

| Variant | AUROC | AUPRC | MCC | Sensitivity | Specificity | Precision | F1 | Threshold | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-first fusion before Mamba | 0.851132 | 0.373322 | 0.357727 | 0.298951 | 0.982095 | 0.495652 | 0.372955 | 0.660000 | 19 |
| Full model / Ours | 0.857696 | 0.398213 | 0.381530 | 0.375291 | 0.973417 | 0.453841 | 0.410845 | 0.650000 | 27 |

## 04_mamba_backbone_versions: Multi-version Mamba Backbone Comparison

| Variant | AUROC | AUPRC | MCC | Sensitivity | Specificity | Precision | F1 | Threshold | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PepMamba-BioMamba / Ours | 0.856193 | 0.395116 | 0.381262 | 0.377622 | 0.972937 | 0.450939 | 0.411037 | 0.630000 | 27 |
| PepMamba-Mamba1 | 0.856179 | 0.395359 | 0.381009 | 0.461538 | 0.955684 | 0.380038 | 0.416842 | 0.390000 | 24 |
| PepMamba-Mamba2 | 0.852272 | 0.385977 | 0.368291 | 0.513403 | 0.938568 | 0.329716 | 0.401550 | 0.410000 | 14 |
| PepMamba-Transformer | 0.849364 | 0.378936 | 0.370133 | 0.430653 | 0.959491 | 0.384896 | 0.406491 | 0.340000 | 15 |
| VanillaMamba3Latest | 0.850220 | 0.375397 | 0.369494 | 0.354895 | 0.974652 | 0.451780 | 0.397520 | 0.470000 | 15 |
