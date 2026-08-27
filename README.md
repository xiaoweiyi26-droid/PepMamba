# PepMamba

Official implementation and reproducibility package for **PepMamba**, a protein–peptide interaction-site prediction model based on bidirectional Mamba blocks and cross-sequence fusion.

> The repository is private during manuscript preparation. It is structured so it can be made public without repackaging.

## What is included

- Model and training code for Bitenet, InterPep, PepBind, and PepNN benchmarks.
- Fixed train/validation/test CSV splits in `data/`.
- Validation-only model-selection and threshold-selection entrypoints in `scripts/final/`.
- UMPPI preparation, audit, training, and evaluation utilities in `scripts/`.
- Paper tables, result provenance, ablation summaries, and the 4GVC interpretability figure in `results/` and `interpretability/`.
- Apache-2.0 license.

Large derived artifacts (ESM-2 embeddings, checkpoints, logs, and intermediate tuning runs) are intentionally excluded from Git. They can be regenerated locally; release checkpoints may later be attached to a GitHub Release or archived on Zenodo.

## Repository layout

```text
.
├── data/                    # Fixed benchmark splits
├── params/                  # Class weights and parameter package
├── scripts/final/           # Paper benchmark runners
├── scripts/                 # Preprocessing and UMPPI workflows
├── interpretability/        # 4GVC analysis and publication figure
├── results/                 # Tables, audits, and compact summaries
├── models.py                # PepMamba architecture
├── modules.py               # Bidirectional Mamba and fusion blocks
├── mamba3_bio.py            # Project-local BioMamba3 module
├── train.py / test.py       # Standard train/evaluation entrypoints
└── requirements.txt
```

## Installation

Python 3.10 and a CUDA-capable PyTorch environment are recommended.

```bash
git clone <repository-url>
cd PepMamba
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` are CUDA/PyTorch-version sensitive. If a wheel cannot be built, install versions matching your CUDA and PyTorch toolchain before installing the remaining requirements.

## Prepare ESM-2 features

The benchmark CSV files are included, while their multi-gigabyte derived embeddings are not. Generate both embedding dictionaries with:

```bash
python scripts/prepare_pair_esm2_embeddings.py --data-dir data --output-dir .
```

This creates `protein_esm2.pkl` and `peptide_esm2.pkl`, which are ignored by Git.

## Reproduce benchmark runs

Use the fixed seed-42 protocol:

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PEPMAMBA_DETERMINISTIC_KERNELS=1
export MAMBA_DETERMINISTIC=1

python train.py --dataset bitenet
python test.py --dataset bitenet
```

Replace `bitenet` with `interpep`, `pepbind`, or `pepnn`. Training selects checkpoints and thresholds using training/validation data only; `test.py` evaluates once at the locked validation threshold.

Paper-specific runners are available under `scripts/final/`:

```bash
python scripts/final/run_bitenet.py --help
python scripts/final/run_interpep.py --help
python scripts/final/run_pepbind.py --help
python scripts/final/run_pepnn.py --help
```

## Results and integrity

- Main comparison: `results/Table2_PepMamba_comparison_with_results.md`
- Result provenance: `results/PAPER_RESULT_PROVENANCE_AUDIT.md`
- Split/threshold audit: `results/DATA_SPLIT_AND_THRESHOLD_AUDIT.md`
- Ablations: `results/ablation/SEED42_RESULTS_SUMMARY.md`
- Interpretability figure: `interpretability/4GVC_A_B_residue_plot/`

## Citation

Please cite the accompanying PepMamba paper. The final BibTeX entry and DOI will be added after acceptance/preprint publication.

## License

Released under the [Apache License 2.0](LICENSE). Dataset and pretrained-model users must also comply with the licenses of the respective upstream sources.
