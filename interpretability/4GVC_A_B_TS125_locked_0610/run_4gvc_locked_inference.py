#!/usr/bin/env python3
import csv
import hashlib
import importlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'runs/bitenet/seed_42/seed_42/pep_mamba_biomamba_latest_mamba3_env/best_checkpoint.pth'
EXPECTED_SHA = '5021fde1c48f27ceb31feaef2777f345a7ba300b4ecdc39e9722d80c4ffcacbf'
THRESHOLD = 0.610
CASE_ID = '4gvc_A_B'
OUTPUT = ROOT / 'interpretability/4GVC_A_B_TS125_locked_0610'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    if sha256(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError('Checkpoint SHA256 mismatch')

    sys.path.insert(0, str(ROOT / 'scripts/final'))
    runner = importlib.import_module('run_bitenet')
    env = runner.activate_latest_mamba3_with_biomamba(
        runner.LATEST_MAMBA3_ROOT, ROOT / 'mamba3_bio.py'
    )
    sys.path.insert(0, str(ROOT))
    training_core = importlib.import_module('bitenet_training_core')

    frame = pd.read_csv(ROOT / 'data/bitenet_test.csv')
    matched = frame.index[frame['pdb_id'].str.lower() == CASE_ID.lower()].tolist()
    if matched != [102]:
        raise RuntimeError(f'Unexpected case match/index: {matched}')
    case_index = matched[0]
    row = frame.iloc[case_index]
    protein_sequence = str(row['Prot_seq']).strip()
    peptide_sequence = str(row['Pep_seq']).strip()
    labels = np.asarray([int(x) for x in str(row['label']).strip()], dtype=np.int64)
    if len(protein_sequence) != 94 or len(labels) != 94:
        raise RuntimeError('Expected exactly 94 protein residues and labels')

    with (ROOT / 'protein_esm2.pkl').open('rb') as handle:
        protein_dict = pickle.load(handle)
    with (ROOT / 'peptide_esm2.pkl').open('rb') as handle:
        peptide_dict = pickle.load(handle)
    if protein_sequence not in protein_dict or peptide_sequence not in peptide_dict:
        raise RuntimeError('Required ESM-2 embedding is absent')
    protein_np = np.asarray(protein_dict[protein_sequence], dtype=np.float32)
    peptide_np = np.asarray(peptide_dict[peptide_sequence], dtype=np.float32)
    if protein_np.shape != (94, 1280) or peptide_np.shape != (8, 1280):
        raise RuntimeError(f'Unexpected embedding shapes: {protein_np.shape}, {peptide_np.shape}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA is required for the archived Mamba3 runtime')
    params = json.loads((CHECKPOINT.parent / 'params.json').read_text())
    p = params['params']; m = params['model_config']
    model = training_core.AblationModel(
        protein_esm=True, peptide_esm=True, model_config=m, dropout=p['dropout']
    ).to(device)
    state = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()

    test_dataset = training_core.RepresentationAblationDataset('test', True, True)
    test_loader = training_core.build_loader(test_dataset, 24, False)
    formal_batch = next(batch for number, batch in enumerate(test_loader) if number == 4)
    peptide, protein, batch_labels, protein_mask = [value.to(device) for value in formal_batch]
    case_slot = 6
    if int(batch_labels[case_slot, :94].ne(-100).sum()) != 94:
        raise RuntimeError('Formal batch case slot mismatch')

    @torch.no_grad()
    def predict(protein_input, peptide_input, return_attention=False):
        logits, attention = model(peptide_input, protein_input, class_mask=protein_mask)
        probability = F.softmax(logits, dim=-1)[case_slot, :94, 1].float().cpu().numpy()
        return (probability, attention) if return_attention else probability

    probabilities, attention = predict(protein, peptide, True)
    if probabilities.shape != (94,) or not np.isfinite(probabilities).all():
        raise RuntimeError('Incomplete or non-finite probabilities')
    predictions = (probabilities >= THRESHOLD).astype(np.int64)
    classes = np.where(
        (labels == 1) & (predictions == 1), 'TP',
        np.where((labels == 0) & (predictions == 1), 'FP',
                 np.where((labels == 1) & (predictions == 0), 'FN', 'TN'))
    )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    prediction_rows = []
    for i, (aa, true, prob, pred, cls) in enumerate(
        zip(protein_sequence, labels, probabilities, predictions, classes), start=1
    ):
        prediction_rows.append({
            'case_id': CASE_ID, 'local_index': i, 'amino_acid': aa,
            'true_label': int(true), 'probability': float(prob),
            'pred_label': int(pred), 'TP_FP_FN_TN': cls,
        })
    pd.DataFrame(prediction_rows).to_csv(OUTPUT / '4GVC_A_B_residue_predictions.csv', index=False)

    metrics = {
        'case_id': CASE_ID, 'case_index_0based': case_index,
        'protein_length': len(protein_sequence), 'peptide_length': len(peptide_sequence),
        'threshold': THRESHOLD, 'threshold_scan_performed': False,
        'auroc': float(roc_auc_score(labels, probabilities)),
        'auprc': float(average_precision_score(labels, probabilities)),
        'mcc': float(matthews_corrcoef(labels, predictions)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'tp': int(np.sum(classes == 'TP')), 'fp': int(np.sum(classes == 'FP')),
        'fn': int(np.sum(classes == 'FN')), 'tn': int(np.sum(classes == 'TN')),
    }
    write_json(OUTPUT / '4GVC_A_B_case_metrics.json', metrics)

    residue_lists = {
        'threshold': THRESHOLD,
        'predicted_positive_local_indices': (np.flatnonzero(predictions == 1) + 1).tolist(),
        'true_positive_local_indices': (np.flatnonzero(classes == 'TP') + 1).tolist(),
        'false_positive_local_indices': (np.flatnonzero(classes == 'FP') + 1).tolist(),
        'false_negative_local_indices': (np.flatnonzero(classes == 'FN') + 1).tolist(),
        'true_binding_local_indices': (np.flatnonzero(labels == 1) + 1).tolist(),
    }
    write_json(OUTPUT / '4GVC_A_B_3D_residue_lists.json', residue_lists)

    cross_rows = []
    for layer_index, tensor in enumerate(attention[:-1], start=1):
        values = tensor.detach().float().cpu().numpy()[case_slot]
        for head in range(values.shape[0]):
            for pi in range(94):
                for qi in range(8):
                    cross_rows.append({
                        'layer': layer_index, 'head': head + 1,
                        'protein_local_index': pi + 1, 'protein_aa': protein_sequence[pi],
                        'peptide_local_index': qi + 1, 'peptide_aa': peptide_sequence[qi],
                        'attention': float(values[head, pi, qi]),
                    })
    pd.DataFrame(cross_rows).to_csv(OUTPUT / '4GVC_A_B_cross_attention_long.csv', index=False)

    cls_tensor = attention[-1]['cls_transformer_attn'].detach().float().cpu().numpy()[case_slot, :, :94, :94]
    cls_rows = []
    mean_matrix = cls_tensor.mean(axis=0)
    received = mean_matrix.mean(axis=0)
    emitted_entropy = -(mean_matrix * np.log(np.clip(mean_matrix, 1e-12, 1))).sum(axis=1)
    for i in range(94):
        cls_rows.append({
            'local_index': i + 1, 'amino_acid': protein_sequence[i],
            'mean_attention_received': float(received[i]),
            'mean_query_entropy': float(emitted_entropy[i]),
        })
    pd.DataFrame(cls_rows).to_csv(OUTPUT / '4GVC_A_B_classifier_attention_summary.csv', index=False)
    np.savez_compressed(
        OUTPUT / '4GVC_A_B_attention_raw.npz',
        **{f'cross_layer_{i+1}': attention[i].detach().float().cpu().numpy()[case_slot]
           for i in range(len(attention) - 1)},
        classifier_self_attention=cls_tensor,
    )

    protein_occ_rows = []
    for i in range(94):
        altered = protein.clone(); altered[case_slot, i, :] = 0
        occ = predict(altered, peptide)
        delta = probabilities - occ
        protein_occ_rows.append({
            'local_index': i + 1, 'amino_acid': protein_sequence[i],
            'true_label': int(labels[i]), 'baseline_probability': float(probabilities[i]),
            'occluded_probability_at_same_residue': float(occ[i]),
            'delta_same_residue': float(delta[i]),
            'mean_probability_delta_all_residues': float(delta.mean()),
            'mean_absolute_probability_delta_all_residues': float(np.abs(delta).mean()),
            'max_absolute_probability_delta_all_residues': float(np.abs(delta).max()),
        })
    pd.DataFrame(protein_occ_rows).to_csv(OUTPUT / '4GVC_A_B_protein_occlusion.csv', index=False)

    peptide_occ_rows = []
    for i in range(8):
        altered = peptide.clone(); altered[case_slot, i, :] = 0
        occ = predict(protein, altered)
        delta = probabilities - occ
        peptide_occ_rows.append({
            'peptide_local_index': i + 1, 'peptide_amino_acid': peptide_sequence[i],
            'mean_probability_delta_all_protein_residues': float(delta.mean()),
            'mean_absolute_probability_delta_all_protein_residues': float(np.abs(delta).mean()),
            'max_absolute_probability_delta_all_protein_residues': float(np.abs(delta).max()),
            'mean_delta_true_binding_residues': float(delta[labels == 1].mean()),
        })
    pd.DataFrame(peptide_occ_rows).to_csv(OUTPUT / '4GVC_A_B_peptide_occlusion.csv', index=False)

    old_predicted = {21,22,23,24,25,26,27,28,29,40,72,75,79}
    new_predicted = set(residue_lists['predicted_positive_local_indices'])
    comparison = {
        'old_result_identity': 'user-supplied old threshold=0.605/trial2 predicted-positive list',
        'old_threshold': 0.605, 'new_threshold': THRESHOLD,
        'old_predicted_positive_local_indices': sorted(old_predicted),
        'new_predicted_positive_local_indices': sorted(new_predicted),
        'became_positive': sorted(new_predicted - old_predicted),
        'became_negative': sorted(old_predicted - new_predicted),
        'unchanged_positive': sorted(old_predicted & new_predicted),
        'probability_comparison_available': False,
        'note': 'Canonical old trial-2 probabilities were not present; only label-set changes can be compared.',
    }
    write_json(OUTPUT / '4GVC_A_B_old_vs_locked0610_comparison.json', comparison)

    audit = {
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha256(CHECKPOINT),
        'threshold': THRESHOLD, 'training_performed': False,
        'test_threshold_scan_performed': False, 'device': str(device),
        'runtime': {key: value for key, value in env.items() if key != 'biomamba_class'}, 'protein_sequence': protein_sequence,
        'peptide_sequence': peptide_sequence, 'embedding_shapes': {
            'protein': list(protein_np.shape), 'peptide': list(peptide_np.shape)
        }, 'outputs': sorted(p.name for p in OUTPUT.iterdir()),
    }
    write_json(OUTPUT / 'INFERENCE_AUDIT.json', audit)
    print(json.dumps({'output': str(OUTPUT), 'metrics': metrics, 'lists': residue_lists,
                      'comparison': comparison}, indent=2))


if __name__ == '__main__':
    main()
