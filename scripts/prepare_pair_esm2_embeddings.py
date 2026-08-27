#!/usr/bin/env python3
"""Generate the ESM-2 dictionaries used by the four benchmark datasets."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import esm
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


def embed(sequences: list[str], model, alphabet, device: torch.device) -> dict[str, object]:
    converter = alphabet.get_batch_converter()
    output = {}
    for sequence in tqdm(sequences, unit="sequence"):
        if len(sequence) > 1022:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds the ESM-2 single-window limit; "
                "use the windowed UMPPI preparation script instead."
            )
        _, strings, tokens = converter([("sequence", sequence)])
        with torch.inference_mode():
            result = model(tokens.to(device), repr_layers=[33], return_contacts=False)
        output[strings[0]] = result["representations"][33][0, 1 : len(sequence) + 1].cpu().numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

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
        values = embed(sequences, model, alphabet, device)
        with (args.output_dir / name).open("wb") as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"wrote {name}: {len(values)} sequences")


if __name__ == "__main__":
    main()
