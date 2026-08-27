#!/usr/bin/env python3
"""Build the audited, residue-only UMPPI split without touching source data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "imported_data_20260808_062534/data/Dataset_all_balanced_new_2"
VALIDITY = PROJECT / "umppi_residue_validity.csv"
OUTPUT = PROJECT / "data/umppi_protein_site"


def official_binding_vec_pos(raw: str, length: int) -> np.ndarray:
    """Exact indexing semantics of official UMPPI binding_vec_pos, for valid rows."""
    indices = [int(x) for x in raw.split(",")]
    indices = [x for x in indices if x < length]
    result = np.zeros(length, dtype=np.uint8)
    result[indices] = 1
    return result


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    source = pd.read_csv(SOURCE, sep="\t", dtype=str, keep_default_na=False)
    validity = pd.read_csv(VALIDITY)
    eligible = validity[validity.eligible_for_protein_residue_loss_and_evaluation].copy()
    eligible_indices = set(eligible.dataset_index.astype(int))
    built = source.iloc[sorted(eligible_indices)].copy()
    built.insert(0, "record_id", [f"umppi_{i:05d}" for i in built.index])
    built.insert(1, "dataset_index", built.index.astype(int))
    split_map = validity.set_index("dataset_index").split.to_dict()
    built.insert(2, "split", built.dataset_index.map(split_map))
    built["protein_sequence_hash"] = built.prot_seq.map(sha)
    built["peptide_sequence_hash"] = built.pep_seq.map(sha)
    built["protein_residue_label"] = [
        "".join(map(str, official_binding_vec_pos(raw, len(seq)).tolist()))
        for raw, seq in zip(built.prot_binding_idx, built.prot_seq)
    ]
    if not all(built.protein_residue_label.str.len() == built.prot_seq.str.len()):
        raise RuntimeError("label length mismatch")
    OUTPUT.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        part = built[built.split == split].copy()
        part.to_csv(OUTPUT / f"{split}.csv", index=False)

    # Deterministic 60-record audit, including long and noncanonical cases when available.
    rng = np.random.default_rng(42)
    audit_ids = set(rng.choice(built.index.to_numpy(), size=50, replace=False).tolist())
    long_ids = built[built.prot_seq.str.len() > 1022].index[:5]
    noncanonical_ids = built[built.prot_seq.str.contains("[^ACDEFGHIKLMNPQRSTVWY]", regex=True)].index[:5]
    audit_ids.update(long_ids.tolist()); audit_ids.update(noncanonical_ids.tolist())
    audit_rows = []
    for idx in sorted(audit_ids):
        row = built.loc[idx]
        official = official_binding_vec_pos(row.prot_binding_idx, len(row.prot_seq))
        ours = np.fromiter((int(x) for x in row.protein_residue_label), dtype=np.uint8)
        indices = [int(x) for x in row.prot_binding_idx.split(",")]
        concat_tokens = row.prot_concat_seq.split(",")
        selected_tokens = ";".join(f"{i}:{concat_tokens[i]}" for i in indices)
        selected_residues = "".join(row.prot_seq[i] for i in indices)
        token_residues = "".join(concat_tokens[i][0] for i in indices)
        audit_rows.append({
            "record_id": row.record_id, "dataset_index": row.dataset_index,
            "protein_length": len(row.prot_seq), "raw_binding_idx": row.prot_binding_idx,
            "official_positive_count": int(official.sum()), "our_positive_count": int(ours.sum()),
            "binary_label_length": len(ours), "labels_identical": bool(np.array_equal(official, ours)),
            "length_matches_original": len(ours) == len(row.prot_seq),
            "concat_token_count": len(concat_tokens),
            "concat_length_matches_original": len(concat_tokens) == len(row.prot_seq),
            "binding_token_residues_match_sequence": selected_residues == token_residues,
            "binding_tokens": selected_tokens, "indexing_scheme": "zero_based_python_sequence_position",
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(PROJECT / "umppi_label_alignment_check.csv", index=False)
    bool_cols = ["labels_identical", "length_matches_original", "concat_length_matches_original",
                 "binding_token_residues_match_sequence"]
    if len(audit) < 50 or not audit[bool_cols].all().all():
        raise RuntimeError("label alignment audit failed")

    sets = {s: set(built.loc[built.split == s, "dataset_index"]) for s in ("train", "validation", "test")}
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("record leakage")
    stats = {}
    for split in ("train", "validation", "test"):
        part = built[built.split == split]
        total_res = int(part.prot_seq.str.len().sum())
        positive_res = int(part.protein_residue_label.str.count("1").sum())
        stats[split] = {
            "records": len(part), "unique_proteins": int(part.prot_seq.nunique()),
            "unique_peptides": int(part.pep_seq.nunique()),
            "duplicated_protein_record_occurrences": int(len(part) - part.prot_seq.nunique()),
            "duplicated_peptide_record_occurrences": int(len(part) - part.pep_seq.nunique()),
            "unique_protein_peptide_pairs": int(part[["prot_seq", "pep_seq"]].drop_duplicates().shape[0]),
            "duplicate_identical_pair_occurrences": int(len(part) - part[["prot_seq", "pep_seq"]].drop_duplicates().shape[0]),
            "residue_positive_count": positive_res, "residue_total_count": total_res,
            "residue_positive_ratio": positive_res / total_res,
            "average_protein_length": float(part.prot_seq.str.len().mean()),
            "average_peptide_length": float(part.pep_seq.str.len().mean()),
        }
    stats["cross_split"] = {
        "record_overlap_train_validation": len(sets["train"] & sets["validation"]),
        "record_overlap_train_test": len(sets["train"] & sets["test"]),
        "record_overlap_validation_test": len(sets["validation"] & sets["test"]),
        "protein_sequence_overlap_train_validation": len(set(built[built.split == "train"].prot_seq) & set(built[built.split == "validation"].prot_seq)),
        "protein_sequence_overlap_train_test": len(set(built[built.split == "train"].prot_seq) & set(built[built.split == "test"].prot_seq)),
        "protein_sequence_overlap_validation_test": len(set(built[built.split == "validation"].prot_seq) & set(built[built.split == "test"].prot_seq)),
        "pair_overlap_train_validation": len(set(zip(built[built.split == "train"].prot_seq, built[built.split == "train"].pep_seq)) & set(zip(built[built.split == "validation"].prot_seq, built[built.split == "validation"].pep_seq))),
        "pair_overlap_train_test": len(set(zip(built[built.split == "train"].prot_seq, built[built.split == "train"].pep_seq)) & set(zip(built[built.split == "test"].prot_seq, built[built.split == "test"].pep_seq))),
        "pair_overlap_validation_test": len(set(zip(built[built.split == "validation"].prot_seq, built[built.split == "validation"].pep_seq)) & set(zip(built[built.split == "test"].prot_seq, built[built.split == "test"].pep_seq))),
    }
    (OUTPUT / "dataset_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "alignment_audit_records": len(audit), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
