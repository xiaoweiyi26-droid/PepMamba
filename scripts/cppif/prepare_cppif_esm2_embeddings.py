#!/usr/bin/env python3
"""Generate length-preserving CPPIF external benchmark ESM-2 embeddings with overlapping windows.

This is preprocessing only. It never trains PepMamba. Outputs are written to a
new directory and existing paths are refused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def starts_for(length: int, window: int, overlap: int) -> list[int]:
    if window <= 0 or not 0 <= overlap < window:
        raise ValueError("require window > 0 and 0 <= overlap < window")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, window - overlap))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def load_fair_esm(project: Path):
    # The project has an esm.py file that otherwise can shadow the fair-esm package.
    project_text = str(project.resolve())
    sys.path[:] = [p for p in sys.path if str(Path(p or os.getcwd()).resolve()) != project_text]
    import esm  # noqa: PLC0415
    module_path = Path(esm.__file__).resolve()
    if module_path == (project / "esm.py").resolve():
        raise RuntimeError("project esm.py shadowed the installed fair-esm package")
    return esm


def embed_sequence(seq, model, alphabet, batch_converter, layer, device, window, overlap):
    starts = starts_for(len(seq), window, overlap)
    total = None
    counts = np.zeros((len(seq), 1), dtype=np.float32)
    for start in starts:
        chunk = seq[start:start + window]
        _, strings, tokens = batch_converter([("sequence", chunk)])
        if strings[0] != chunk or tokens.shape[1] != len(chunk) + 2:
            raise RuntimeError(f"token alignment failure at window start {start}")
        with torch.inference_mode():
            result = model(tokens.to(device), repr_layers=[layer], return_contacts=False)
        embedding = result["representations"][layer][0, 1:len(chunk) + 1].float().cpu().numpy()
        if embedding.shape[0] != len(chunk):
            raise RuntimeError(f"embedding length {embedding.shape[0]} != chunk length {len(chunk)}")
        if total is None:
            total = np.zeros((len(seq), embedding.shape[1]), dtype=np.float32)
        total[start:start + len(chunk)] += embedding
        counts[start:start + len(chunk)] += 1
    if total is None or (counts == 0).any():
        raise RuntimeError("window coverage failure")
    merged = total / counts
    if merged.shape[0] != len(seq):
        raise RuntimeError("final embedding is not residue-aligned")
    return merged, starts, int(counts.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--window", type=int, default=1022)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate tokens/windows without loading the model")
    args = parser.parse_args()

    project = args.project.resolve()
    dataset_dir = (args.dataset_dir or project / "data/cppif").resolve()
    df = pd.concat([pd.read_csv(dataset_dir / f"{split}.csv", dtype=str, keep_default_na=False) for split in ("train", "validation", "test")], ignore_index=True)
    table = dataset_dir
    output = (args.output_dir or project / "embeddings/cppif_esm2_650m").resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)

    esm = load_fair_esm(project)
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    supported = set(alphabet.tok_to_idx)
    sides = [("protein", "prot_seq"), ("peptide", "pep_seq")]
    sequences = {side: df[column].drop_duplicates().tolist() for side, column in sides}
    unsupported = sorted({char for values in sequences.values() for seq in values for char in seq if char not in supported})
    if unsupported:
        raise ValueError(f"unsupported residues {unsupported}; no silent deletion or mapping is permitted")

    metadata = {
        "source_table": str(table), "model": args.model, "layer": args.layer,
        "window": args.window, "overlap": args.overlap, "merge": "arithmetic_mean",
        "device": args.device, "storage_dtype": args.storage_dtype, "dry_run": args.dry_run,
        "noncanonical_tokens": {x: alphabet.tok_to_idx[x] for x in "XBUZ"},
        "sequence_length_changed": False, "training_performed": False,
    }
    (output / "embedding_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model = batch_converter = None
    if not args.dry_run:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        loader = getattr(esm.pretrained, args.model)
        model, loaded_alphabet = loader()
        if loaded_alphabet.all_toks != alphabet.all_toks:
            raise RuntimeError("model alphabet differs from preflight alphabet")
        alphabet = loaded_alphabet
        batch_converter = alphabet.get_batch_converter()
        model.eval().requires_grad_(False).to(args.device)

    checks = []
    target_dtype = np.float16 if args.storage_dtype == "float16" else np.float32
    for side, values in sequences.items():
        side_dir = output / ("proteins" if side == "protein" else "peptides")
        side_dir.mkdir(exist_ok=True)
        for sequence_index, seq in enumerate(values):
            starts = starts_for(len(seq), args.window, args.overlap)
            embedding = None
            max_coverage = None
            seq_hash = hashlib.sha256(seq.encode()).hexdigest()
            final = side_dir / f"{seq_hash}.npy"
            if not args.dry_run and args.resume and final.exists():
                embedding = np.load(final, mmap_mode="r")
                if embedding.shape != (len(seq), 1280):
                    raise ValueError(f"invalid resumed embedding {embedding.shape}: {final}")
                max_coverage = None
            elif not args.dry_run:
                embedding, starts, max_coverage = embed_sequence(
                    seq, model, alphabet, batch_converter, args.layer, args.device,
                    args.window, args.overlap)
                temporary = final.with_suffix(".npy.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, embedding.astype(target_dtype))
                temporary.replace(final)
            checks.append({
                "side": side, "sequence_index": sequence_index,
                "sequence_sha256": seq_hash,
                "sequence_type": side,
                "file_path": "" if args.dry_run else str(final.relative_to(project)),
                "file_sha256": "DRY_RUN" if args.dry_run else hashlib.sha256(final.read_bytes()).hexdigest(),
                "original_length": len(seq), "window_count": len(starts),
                "embedding_length": None if embedding is None else embedding.shape[0],
                "embedding_dimension": None if embedding is None else embedding.shape[1],
                "alignment_ok": None if embedding is None else embedding.shape[0] == len(seq),
                "max_overlap_coverage": max_coverage, "dry_run": args.dry_run,
                "contains_X": "X" in seq, "contains_B": "B" in seq,
                "contains_U": "U" in seq, "contains_Z": "Z" in seq,
            })
    pd.DataFrame(checks).to_csv(output / "manifest.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "embedding_alignment_check.csv", index=False)
    if not args.dry_run and not all(row["alignment_ok"] for row in checks):
        raise RuntimeError("one or more final embeddings are misaligned")
    print(json.dumps({"output_dir": str(output), "dry_run": args.dry_run,
                      "sequence_counts": {k: len(v) for k, v in sequences.items()}}, indent=2))


if __name__ == "__main__":
    main()
