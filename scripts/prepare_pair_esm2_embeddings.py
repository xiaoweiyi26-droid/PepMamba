#!/usr/bin/env python3
"""Generate the ESM-2 dictionaries used by the four benchmark datasets."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def collect_sequences(data_dir: Path) -> tuple[list[str], list[str]]:
    proteins: set[str] = set()
    peptides: set[str] = set()
    for path in sorted(data_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if {"Prot_seq", "Pep_seq"}.issubset(frame.columns):
            proteins.update(frame["Prot_seq"].dropna().astype(str).str.strip())
            peptides.update(frame["Pep_seq"].dropna().astype(str).str.strip())
    if not proteins or not peptides:
        raise RuntimeError(f"No benchmark Prot_seq/Pep_seq columns found under {data_dir}")
    return sorted(proteins), sorted(peptides)


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


def embed_sequence(sequence, model, converter, device, window, overlap):
    total = np.zeros((len(sequence), 1280), dtype=np.float32)
    counts = np.zeros((len(sequence), 1), dtype=np.float32)
    for start in starts_for(len(sequence), window, overlap):
        chunk = sequence[start:start + window]
        _, strings, tokens = converter([("sequence", chunk)])
        if strings[0] != chunk or tokens.shape[1] != len(chunk) + 2:
            raise RuntimeError(f"ESM-2 token alignment failure at window start {start}")
        with torch.inference_mode():
            result = model(tokens.to(device), repr_layers=[33], return_contacts=False)
        embedding = result["representations"][33][0, 1:len(chunk) + 1].float().cpu().numpy()
        if embedding.shape != (len(chunk), 1280):
            raise RuntimeError(
                f"ESM-2 window embedding shape {embedding.shape} != ({len(chunk)}, 1280)"
            )
        total[start:start + len(chunk)] += embedding
        counts[start:start + len(chunk)] += 1
    if (counts == 0).any():
        raise RuntimeError("ESM-2 overlapping windows did not cover every residue")
    merged = total / counts
    if merged.shape != (len(sequence), 1280):
        raise RuntimeError(
            f"ESM-2 embedding shape {merged.shape} != ({len(sequence)}, 1280)"
        )
    return merged


def embed(
    sequences: list[str],
    model,
    alphabet,
    device: torch.device,
    window: int = 1022,
    overlap: int = 128,
) -> dict[str, object]:
    converter = alphabet.get_batch_converter()
    output = {}
    for sequence in tqdm(sequences, unit="sequence"):
        embedding = embed_sequence(sequence, model, converter, device, window, overlap)
        if len(sequence) != embedding.shape[0]:
            raise RuntimeError(
                f"sequence/embedding length mismatch: {len(sequence)} != {embedding.shape[0]}"
            )
        output[sequence] = embedding
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--window", type=int, default=1022)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()
    starts_for(1, args.window, args.overlap)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    proteins, peptides = collect_sequences(args.data_dir)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, sequences in (("protein_esm2.pkl", proteins), ("peptide_esm2.pkl", peptides)):
        values = embed(sequences, model, alphabet, device, args.window, args.overlap)
        with (args.output_dir / name).open("wb") as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"wrote {name}: {len(values)} sequences")


if __name__ == "__main__":
    main()
