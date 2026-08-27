# PAPER_ABLATION_FINAL_4GROUPS

This directory is the organized archive for the four final BITENET ablation studies.

## Reporting Rule

- Main paper results: seed = 42 only.
- Other seeds: retained only as backup/provenance.
- Original experiment directories are not deleted or modified.
- Run scripts are collected separately under `run_scripts/`.

## Four Ablation Groups

1. `01_pretrained_representation`
   - Learnable + Learnable
   - ESM-2 + Learnable
   - Learnable + ESM-2
   - ESM-2 + ESM-2

2. `02_branch_design`
   - Full model
   - w/o protein prior
   - w/o interaction-refined

3. `03_cross_mamba_order`
   - Mamba -> Cross-attention
   - Cross-attention -> Mamba

4. `04_mamba_backbone_versions`
   - Transformer
   - Mamba1
   - Mamba2
   - Vanilla Mamba3
   - BioMamba3 / PepMamba
   - Other variants are retained according to the original backbone experiment records.

## Directory Rule

Each group contains:

- `seed42_results/`: final seed=42 runs for paper use.
- `other_seeds_backup/`: non-42 runs retained as backup.
- `summaries/`: original aggregate CSV/XLSX/Markdown reports.

## Important

This organization step copies data only.
The original `tuning_results/` directories remain untouched.
