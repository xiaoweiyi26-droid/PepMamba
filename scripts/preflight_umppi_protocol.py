#!/usr/bin/env python3
"""Read-only UMPPI protocol audit; does not train or deserialize arbitrary classes."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
ESM_NATIVE_EXTRA = set("XB UZ".replace(" ", ""))


class RestrictedUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"forbidden global {module}.{name}") from exc


def load_legacy_torch_indices(path: Path):
    with path.open("rb") as handle:
        pickle.load(handle)
        pickle.load(handle)
        pickle.load(handle)
        return RestrictedUnpickler(handle, encoding="latin1").load()


def arr(value) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def site_status(row) -> str:
    if row.label == "0":
        return "pair_negative_NoBinding"
    if row.prot_binding_idx == "-99999":
        return "positive_missing_minus99999"
    try:
        sites = [int(x) for x in row.prot_binding_idx.split(",")]
    except ValueError:
        return "positive_malformed"
    if any(x < 0 or x >= len(row.prot_seq) for x in sites):
        return "positive_out_of_range"
    if len(set(sites)) != len(sites):
        return "positive_duplicate_indices"
    return "positive_explicit_valid"


def count_group(df: pd.DataFrame, indices: np.ndarray) -> dict:
    part = df.iloc[indices]
    statuses = part.protein_site_status.value_counts()
    valid = part[part.protein_site_status == "positive_explicit_valid"]
    positive_residues = int(valid.protein_binding_residue_count.sum())
    labeled_residues = int(valid.protein_length.sum())
    return {
        "sample_count": len(part),
        "positive_pair_count": int((part.label == "1").sum()),
        "negative_pair_count": int((part.label == "0").sum()),
        "valid_protein_site_pair_count": len(valid),
        "valid_protein_residue_label_count": labeled_residues,
        "binding_residue_positive_label_count": positive_residues,
        "binding_residue_negative_label_count": labeled_residues - positive_residues,
        "minus99999_count": int(statuses.get("positive_missing_minus99999", 0)),
        "NoBinding_count": int(statuses.get("pair_negative_NoBinding", 0)),
        "malformed_or_out_of_range_pair_count": int(sum(statuses.get(x, 0) for x in (
            "positive_malformed", "positive_out_of_range", "positive_duplicate_indices"))),
    }


def window_starts(length: int, window: int = 1022, overlap: int = 128) -> list[int]:
    if not (0 <= overlap < window):
        raise ValueError("overlap must be >=0 and < window")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, window - overlap))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project.resolve()
    imported = project / "imported_data_20260808_062534"
    data_dir = imported / "data"
    table = data_dir / "Dataset_all_balanced_new_2"
    df = pd.read_csv(table, sep="\t", dtype=str, keep_default_na=False)
    df["dataset_index"] = np.arange(len(df))
    df["protein_length"] = df.prot_seq.str.len()
    df["peptide_length"] = df.pep_seq.str.len()
    df["protein_site_status"] = [site_status(row) for row in df.itertuples(index=False)]
    df["protein_binding_residue_count"] = [
        len(row.prot_binding_idx.split(",")) if row.protein_site_status == "positive_explicit_valid" else 0
        for row in df.itertuples(index=False)
    ]

    split_rows = []
    loaded = {}
    role_names = {0: "official_train_idx", 1: "official_valid_idx", 2: "official_test_idx"}
    for path in sorted(data_dir.glob("clu_thre_*/*_balanced")):
        obj = load_legacy_torch_indices(path)
        loaded[path] = obj
        threshold = path.parent.name.removeprefix("clu_thre_")
        scenario = path.name.removesuffix("_balanced")
        for fold in range(len(obj[0])):
            groups = [arr(obj[g][fold]) for g in range(3)]
            sets = [set(x.tolist()) for x in groups]
            for group, indices in enumerate(groups):
                row = {
                    "index_file": str(path.relative_to(project)), "cluster_setting": threshold,
                    "scenario": scenario, "fold": fold, "group": group,
                    "official_source_role": role_names[group],
                    "group1_subset_of_group0": sets[1].issubset(sets[0]),
                    "group2_disjoint_from_group0": sets[2].isdisjoint(sets[0]),
                    "group2_disjoint_from_group1": sets[2].isdisjoint(sets[1]),
                }
                row.update(count_group(df, indices))
                split_rows.append(row)
    counts = pd.DataFrame(split_rows)
    counts.to_csv(project / "umppi_split_counts.csv", index=False)

    # Exact Table-1 match: threshold directory 0.0001, new_protein, fold 0.
    source = data_dir / "clu_thre_0.0001" / "new_protein_balanced"
    obj = loaded[source]
    fold = 0
    group0, group1, group2 = [arr(obj[g][fold]) for g in range(3)]
    effective_train = np.asarray(sorted(set(group0) - set(group1)), dtype=np.int64)
    validation = np.asarray(sorted(set(group1)), dtype=np.int64)
    test = np.asarray(sorted(set(group2)), dtype=np.int64)
    assert not (set(effective_train) & set(validation))
    assert not (set(effective_train) & set(test))
    assert not (set(validation) & set(test))
    assert set(effective_train) | set(validation) | set(test) == set(range(len(df)))

    assignment = {}
    for split, indices in (("train", effective_train), ("validation", validation), ("test", test)):
        assignment.update({int(i): split for i in indices})
    validity = df.copy()
    validity["split"] = validity.dataset_index.map(assignment)
    validity["eligible_for_protein_residue_loss_and_evaluation"] = validity.protein_site_status.eq("positive_explicit_valid")
    validity["protein_sequence_sha256"] = validity.prot_seq.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    validity["peptide_sequence_sha256"] = validity.pep_seq.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    validity["protein_noncanonical_characters"] = validity.prot_seq.map(lambda s: "".join(sorted(set(s) - CANONICAL)))
    validity["esm_token_strategy"] = validity.protein_noncanonical_characters.map(
        lambda s: "native_tokens_preserve_length" if set(s).issubset(ESM_NATIVE_EXTRA) else "unsupported_requires_explicit_mapping"
    )
    validity["protein_metric_has_both_classes"] = (
        validity.eligible_for_protein_residue_loss_and_evaluation
        & (validity.protein_binding_residue_count > 0)
        & (validity.protein_binding_residue_count < validity.protein_length)
    )
    validity_out = validity[[
        "dataset_index", "split", "label", "protein_site_status",
        "eligible_for_protein_residue_loss_and_evaluation", "protein_metric_has_both_classes",
        "protein_length", "peptide_length", "protein_binding_residue_count",
        "protein_noncanonical_characters", "esm_token_strategy",
        "protein_sequence_sha256", "peptide_sequence_sha256",
    ]]
    validity_out.to_csv(project / "umppi_residue_validity.csv", index=False)

    # One row per unique long protein, without exposing giant sequences in the report.
    seq_groups = validity.groupby("prot_seq", sort=False)
    long_rows = []
    alignment_rows = []
    for seq, group in seq_groups:
        length = len(seq)
        starts = window_starts(length)
        coverage = np.zeros(length, dtype=np.int16)
        for start in starts:
            coverage[start:min(start + 1022, length)] += 1
        common = {
            "protein_sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
            "protein_length": length, "dataset_pair_occurrences": len(group),
            "split_memberships": ";".join(sorted(group.split.unique())),
            "eligible_pair_occurrences": int(group.eligible_for_protein_residue_loss_and_evaluation.sum()),
            "noncanonical_characters": "".join(sorted(set(seq) - CANONICAL)),
            "esm_native_noncanonical_support": set(seq).issubset(CANONICAL | ESM_NATIVE_EXTRA),
            "window_size": 1022, "overlap": 128, "window_count": len(starts),
            "planned_reconstructed_length": int(len(coverage)),
            "every_residue_covered": bool((coverage > 0).all()),
            "overlap_averaging_required": bool((coverage > 1).any()),
        }
        alignment_rows.append({**common, "embedding_generated": False,
                               "check_type": "preflight_window_coverage_not_actual_embedding"})
        if length > 1022:
            long_rows.append(common)
    pd.DataFrame(long_rows).sort_values("protein_length", ascending=False).to_csv(
        project / "long_sequence_report.csv", index=False)
    pd.DataFrame(alignment_rows).sort_values("protein_length", ascending=False).to_csv(
        project / "embedding_alignment_check.csv", index=False)

    selected_counts = {
        split: count_group(df, indices)
        for split, indices in (("official_group0_training_pool", group0),
                               ("validation", validation), ("independent_test", test),
                               ("effective_train_after_validation_subtraction", effective_train))
    }
    unique_test = validity[validity.split == "test"]
    valid_test = unique_test[unique_test.eligible_for_protein_residue_loss_and_evaluation]
    recommended = {
        "protocol_status": "blocked_pending_chunked_ESM2_and_training_pipeline_adapter",
        "task": "protein+peptide_to_protein_binding_residues",
        "source_table": str(table.relative_to(project)),
        "index_file": str(source.relative_to(project)), "cluster_setting": "0.0001",
        "paper_setting_mapping": {"0.0001_to_NA": "unresolved"},
        "scenario": "new_protein", "fold": fold,
        "official_group_semantics_from_source": {"0": "train_idx", "1": "valid_idx", "2": "test_idx"},
        "effective_split_construction": {
            "train": "group0 minus group1", "validation": "group1", "test": "group2",
            "reason": "official group1 is contained in group0; subtraction prevents train-validation overlap",
        },
        "counts": selected_counts,
        "residue_task_eligibility": "positive pair with explicit, valid prot_binding_idx only",
        "NoBinding_policy": "exclude/mask from protein-residue loss and evaluation, matching official source flags",
        "minus99999_policy": "missing annotation; exclude/mask; never convert to residue negatives",
        "macro_evaluation": {
            "primary_comparison": "mean per-record protein AUROC and MCC, matching official batch_size=1 implementation",
            "require_both_residue_classes": True,
            "also_report_unique_sequence_grouped_macro": True,
            "valid_test_pair_records": len(valid_test),
            "valid_test_pair_records_with_both_classes": int(valid_test.protein_metric_has_both_classes.sum()),
            "valid_test_unique_protein_sequences": int(valid_test.prot_seq.nunique()),
        },
        "micro_evaluation": ["AUROC", "AUPRC", "MCC", "F1", "Precision", "Recall", "Specificity"],
        "checkpoint_and_threshold_rule": "must be selected only on validation; test evaluated once",
        "allowed_formal_cluster_settings": ["NA", "0.3", "0.4", "0.5"],
        "excluded_from_formal_reproduction": ["0.6"],
        "long_sequence_embedding": {"model_max_residues_per_window": 1022, "overlap": 128,
                                    "merge": "mean over overlapping windows", "silent_truncation": False},
        "noncanonical_strategy": {"X": "native ESM token", "B": "native ESM token",
                                  "U": "native ESM token", "Z": "native ESM token",
                                  "sequence_length_changed": False},
    }
    (project / "recommended_split.json").write_text(json.dumps(recommended, indent=2, ensure_ascii=False) + "\n")

    long_unique = len(long_rows)
    long_pairs = int((validity.protein_length > 1022).sum())
    report = f"""# UMPPI split and protocol preflight check

## Decision

**Do not start PepMamba training yet.** The split can now be constructed without overlap, but the current ESM pipeline is not safe for UMPPI: it deletes noncanonical residues, has no >1022-residue chunking, and the dataloader silently clips feature/label lengths.

## Data and official-source evidence

- Included balanced table: `{table.relative_to(project)}`; 17,978 pairs (8,989 positive, 8,989 negative).
- Official UMPPI `data_loader.py` names outer groups 0/1/2 as `train_idx_list`, `valid_idx_list`, `test_idx_list` (official repository lines 286-330).
- Official source assigns residue-site flags only to positive pairs with explicit annotations; negative `NoBinding` and positive `-99999` receive flag 0 and are excluded by the masked residue loss/evaluation (data_loader lines 18-41 and 97-119; main.py lines 15-20; evaluate_each.py lines 302-327).
- Official repository: https://github.com/mrjohnmpu/UMPPI
- Official article: https://doi.org/10.1021/acs.jcim.4c02365
- Official Supporting Information: https://acs.figshare.com/articles/journal_contribution/28587821
- The Supporting Information reports peptide lengths 2-50, median 14, and 12.6% containing noncanonical residues. The downloaded table matches all three values (2-50, median 14, 2,266/17,978 = 12.60%), supporting that this is the complete UMPPI balanced table.

## Table 1 match and split semantics

`data/clu_thre_0.0001/new_protein_balanced`, fold 0, is the unique exact match found for the supplied Table 1 standard independent counts:

| object | samples | positive | negative | interpretation |
|---|---:|---:|---:|---|
| group 0 | 14,452 | 7,226 | 7,226 | official training pool |
| group 1 | 1,445 | 734 | 711 | official validation subset; not a positive-only subset |
| group 2 | 3,526 | 1,763 | 1,763 | official independent test |
| group 0 minus group 1 | 13,007 | 6,492 | 6,515 | legal effective PepMamba training subset |

Group 1 is contained in group 0 in every supplied balanced file/fold. The official loader uses both lists directly, which would overlap training and validation. For our validation-based PepMamba protocol, use `group0 - group1` for training, group1 for validation, and group2 for the untouched test set.

The repository uses the literal setting `0.0001`, while the paper setting supplied for this audit is `NA`. No README/code statement equating them was found. Therefore **0.0001 = NA remains unresolved**, even though the 0.0001/new_protein/fold0 counts exactly match Table 1. Do not use 0.6 for formal reproduction.

## Protein-residue task eligibility

Only positive pairs with an explicit, in-range `prot_binding_idx` are eligible. `-99999` is missing annotation and is masked. Negative `NoBinding` pairs are excluded from protein-residue loss and evaluation, consistent with official flags; their residues are not relabeled as non-binding.

Recommended-split counts are recorded in `recommended_split.json` and per-record decisions in `umppi_residue_validity.csv`.

## Evaluation protocol

Primary UMPPI comparison must report per-record/per-protein AUROC and MCC, calculated separately only when both residue classes exist, followed by their arithmetic means and valid-record count. The official `evaluate_each.py` uses batch size 1 and appends one AUC/MCC per test record (lines 316-327), so repeated identical protein sequences paired with different peptides are separate records. Also report a grouped-by-unique-protein-sequence macro result as a transparent secondary analysis.

Additionally report pooled/micro AUROC, AUPRC, MCC, F1, Precision, Recall, and Specificity over all eligible test residues. Checkpoint and threshold selection must use validation only. The official `main.py` evaluates test every epoch and selects/saves by test AUC (lines 287-306); that leakage-prone behavior must not be copied into PepMamba.

The supplied Figure 4 reference values are mean per-protein AUC 0.914 and MCC 0.489; pooled metrics must not be presented as directly equivalent.

## Long sequences and ESM-2

- Unique proteins longer than 1022 residues: **{long_unique}** of {df.prot_seq.nunique()}.
- Pair rows containing a >1022-residue protein: **{long_pairs}** of {len(df)}.
- Maximum protein length: **{df.protein_length.max()}**.
- Current `esm.py` deletes every character outside the 20 canonical amino acids (line 43), changing sequence length and residue alignment.
- It feeds whole sequences into ESM2 without windows (lines 58-75); ESM2 t33 permits 1022 residues plus BOS/EOS, so long inputs are unsupported.
- Current `dataloaders.py` silently clips protein features and labels to their minimum length (lines 63-65).

The installed ESM alphabet has native tokens for X, B, U, and Z; preserve them without deletion or length change. `embedding_alignment_check.csv` is currently a **window-coverage dry-run**, not generated embeddings. It verifies a 1022-residue window with 128-residue overlap covers every original position and plans mean aggregation in overlap regions. Actual embeddings must later be checked to have exactly the original sequence length.

An isolated implementation is provided at `scripts/prepare_umppi_esm2_embeddings.py`; its tokenizer/window dry-run passed for all 3,972 unique proteins and 6,603 unique peptides. It has not generated embeddings or trained a model.

## Output files

- `umppi_split_counts.csv`: every balanced setting/fold/group and requested counts.
- `umppi_residue_validity.csv`: recommended split membership and per-pair residue eligibility.
- `long_sequence_report.csv`: every unique protein over 1022 residues.
- `embedding_alignment_check.csv`: dry-run window coverage/alignment plan.
- `recommended_split.json`: machine-readable recommended protocol.

## Final answers

A. Closest Table 1 setting: `clu_thre_0.0001/new_protein_balanced`, fold 0; it exactly matches 14,452/3,526 and 7,226/1,763 per class. Mapping 0.0001 to paper `NA` remains unresolved.

B. Groups: official source calls them train, validation, test. In this exact fold they are training pool, a 1,445-pair validation subset nested inside that pool, and the 3,526-pair independent test. Legal effective train is group0 minus group1.

C. Protein-site samples: only positive pairs with explicit valid `prot_binding_idx`; exclude/mask negative `NoBinding` and positive `-99999` for residue loss/evaluation.

D. Safe to train now: **No**. Split semantics are resolved enough to construct data, but actual chunked ESM embeddings and strict feature/label alignment checks are still required before training.
"""
    (project / "UMPPI_SPLIT_PROTOCOL_CHECK.md").write_text(report)
    print(json.dumps({"status": recommended["protocol_status"], "selected_counts": selected_counts,
                      "long_unique_proteins": long_unique, "long_pair_rows": long_pairs}, indent=2))


if __name__ == "__main__":
    main()
