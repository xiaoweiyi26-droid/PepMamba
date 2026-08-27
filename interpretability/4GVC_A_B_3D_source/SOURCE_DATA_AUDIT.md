# 4GVC_A_B 3D source-data recovery audit

Audit date: 2026-08-15 (UTC)

## Status

Recovery is incomplete and stopped without substituting data from another run.

## Dataset source found

- Source: `./data/bitenet_test.csv`
- SHA256: `85b429ce4f189f2e31b1b37d2316f5d65a10e16b075899311b0d99bdadfeed18`
- DataFrame case index (0-based): `102`
- Case ID as stored: `4gvc_A_B`
- Protein sequence (94 aa): `GAMGKVTHSIHIEKSDTAADTYGFSLSSVEEDGIRRLYVNSVKETGLASKKGLKAGDEILEINNRAADALNSSMLKDFLSQPSLGLLVRTYPEL`
- Peptide sequence (8 aa): `TKQEEFYA`
- True residue labels: `0000000000000000000011111111100000000001001000000000000000000000000000010001001000000000000000`
- True-positive local indices: `21,22,23,24,25,26,27,28,29,40,43,72,76,79` (14 residues)
- Flattened test-residue slice: Python half-open interval `[26775:26869]` in a 30,870-residue concatenation.

## Canonical trial-2 provenance found, but source artifacts missing

- `./PAPER_RESULT_PROVENANCE_AUDIT.md` identifies `tuning_results/BITENET_FINAL_KEEP/` as the canonical full-search trial 2 result and records final threshold `0.605`.
- `./PAPER_RESULT_PROVENANCE_TABLE.csv` records the intended checkpoint as `tuning_results/BITENET_FINAL_KEEP/checkpoints/trial2_best_state.pth` and threshold `0.605`.
- The referenced `tuning_results/BITENET_FINAL_KEEP/` directory is absent from the current filesystem.
- No canonical trial-2 saved probability file or checkpoint was found under `/root`, including the trash directory and local archive member listings.
- Other BITENET prediction files were inspected but were not substituted. In particular, `PAPER_MAIN_RESULTS_FINAL/TS125_BITENET/latest_repeat2/.../test_predictions.npz` is a later BioMamba repeat, not canonical full-search trial 2.

## Cross-check possible from a noncanonical run (not accepted as recovered source)

Using the exact saved-residue slice from the later `latest_repeat2` file and threshold 0.605 reproduces true=14, TP=12, FP=L75, FN=K43/K76. This coincidence is evidence for the case indexing and expected confusion pattern only; its probabilities are not canonical trial-2 probabilities and were therefore not exported.

## Structure search

No local `4GVC.pdb`, `4GVC.cif`, PyMOL (`.pse`/`.pml`), ChimeraX (`.cxc`/`.cxs`), or filename-matched 4GVC figure/source file was found in the project, under `/root`, in trash, or in local archive member listings. No external or alternative structure was downloaded or substituted.

Consequently, strict chain-A sequence alignment and numbering mapping, chain-B/PTR verification, and the independent `<3.5 Å` atom-distance interface check cannot be performed from current local evidence. The requested prediction and structure-derived CSV files are intentionally absent rather than populated with guessed or noncanonical data.
