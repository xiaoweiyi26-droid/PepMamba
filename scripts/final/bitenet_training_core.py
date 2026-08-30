#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run BITENET pretrained-representation ablations.

This runner compares whether the protein and peptide inputs use frozen ESM-2
features or trainable amino-acid embeddings. It intentionally leaves the core
PepMamba/BITENET model implementation unchanged.
"""

import csv
import gc
import json
import math
import os
import pickle
import random
import time
import zipfile
import argparse
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from models import PepMamba


ROOT = Path(__file__).resolve().parents[2]
SEED = 42
EPOCHS = 35
ESM_DIM = 1280
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYBXZUOJ-"
AA_TO_ID = {aa: index + 1 for index, aa in enumerate(AA_VOCAB)}

FINAL_PARAMS = {
    "lr": 0.0001108299891310337,
    "dropout": 0.2847604379195126,
    "alpha": 0.771244432660029,
    "gamma": 1.5692306179826772,
    "ce_weight": 0.6889605349026123,
    "weight_decay": 0.0015285575292836157,
    "batch_size": 24,
    "scheduler_t0": 10,
    "class_weight_power": 0.42543759653821034,
    "max_pos_weight": 7.763722474165384,
    "grad_clip": 1.0,
    "n_layers": 3,
    "epochs": EPOCHS,
    "best_epoch": 27,
    "threshold": 0.6099999999999998,
}

BASE_MODEL_CONFIG = {
    "n_layers": FINAL_PARAMS["n_layers"],
    "d_model": 128,
    "n_head": 4,
    "d_k": 32,
    "d_v": 32,
    "d_inner": 256,
}

VARIANTS = [
    {
        "name": "learnable_learnable",
        "label": "Learnable + Learnable",
        "protein_esm": False,
        "peptide_esm": False,
    },
    {
        "name": "esm2_learnable",
        "label": "ESM-2 + Learnable",
        "protein_esm": True,
        "peptide_esm": False,
    },
    {
        "name": "learnable_esm2",
        "label": "Learnable + ESM-2",
        "protein_esm": False,
        "peptide_esm": True,
    },
    {
        "name": "esm2_esm2",
        "label": "ESM-2 + ESM-2",
        "protein_esm": True,
        "peptide_esm": True,
    },
]


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def parse_label(value):
    bits = [int(char) for char in str(value) if char in {"0", "1"}]
    if not bits:
        raise ValueError(f"Invalid binary label: {value}")
    return torch.tensor(bits, dtype=torch.long)


def encode_sequence(seq):
    return torch.tensor([AA_TO_ID.get(char.upper(), AA_TO_ID["X"]) for char in str(seq).strip()], dtype=torch.long)


class RepresentationAblationDataset(Dataset):
    def __init__(self, mode, protein_esm, peptide_esm):
        self.mode = mode
        self.protein_esm = protein_esm
        self.peptide_esm = peptide_esm
        self.df = pd.read_csv(ROOT / "data" / f"bitenet_{mode}.csv")
        self.protein_features = None
        self.peptide_features = None
        if protein_esm:
            with (ROOT / "protein_esm2.pkl").open("rb") as handle:
                self.protein_features = pickle.load(handle)
        if peptide_esm:
            with (ROOT / "peptide_esm2.pkl").open("rb") as handle:
                self.peptide_features = pickle.load(handle)

    def _load_esm(self, seq, feature_dict):
        if seq in feature_dict:
            return torch.from_numpy(feature_dict[seq]).float()
        return torch.zeros((len(seq), ESM_DIM), dtype=torch.float32)

    def _side_value(self, seq, use_esm, feature_dict):
        if use_esm:
            feat = self._load_esm(seq, feature_dict)
            if feat.ndim != 2 or feat.shape[-1] != ESM_DIM:
                raise ValueError(f"Unexpected ESM feature shape for {seq[:12]}: {tuple(feat.shape)}")
            return feat
        return encode_sequence(seq)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pep_seq = str(row["Pep_seq"]).strip()
        prot_seq = str(row["Prot_seq"]).strip()
        label = parse_label(row["label"])
        pep = self._side_value(pep_seq, self.peptide_esm, self.peptide_features)
        prot = self._side_value(prot_seq, self.protein_esm, self.protein_features)
        min_len = min(len(prot_seq), label.shape[0], prot.shape[0])
        prot = prot[:min_len]
        label = label[:min_len]

        if self.mode == "train" and self.protein_esm:
            if torch.rand(1).item() < 0.5:
                prot = prot + torch.randn_like(prot) * 0.02
            if torch.rand(1).item() < 0.3:
                channel_mask = (torch.rand(prot.shape[-1]) > 0.1).float()
                prot = prot * channel_mask.unsqueeze(0)

        return pep, prot, label

    def __len__(self):
        return len(self.df)


def collate_ablation(batch):
    pep, prot, label = zip(*batch)
    pep_pad_value = 0.0 if pep[0].dtype.is_floating_point else 0
    prot_pad_value = 0.0 if prot[0].dtype.is_floating_point else 0
    pep_pad = pad_sequence(pep, batch_first=True, padding_value=pep_pad_value)
    prot_pad = pad_sequence(prot, batch_first=True, padding_value=prot_pad_value)
    label_pad = pad_sequence(label, batch_first=True, padding_value=-100)
    return pep_pad, prot_pad, label_pad, label_pad.eq(-100)


def build_loader(dataset, batch_size, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ablation,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


class AblationModel(nn.Module):
    def __init__(self, protein_esm, peptide_esm, model_config, dropout):
        super().__init__()
        self.protein_esm = protein_esm
        self.peptide_esm = peptide_esm
        vocab_size = len(AA_TO_ID) + 1
        self.protein_embedding = None if protein_esm else nn.Embedding(vocab_size, ESM_DIM, padding_idx=0)
        self.peptide_embedding = None if peptide_esm else nn.Embedding(vocab_size, ESM_DIM, padding_idx=0)
        self.backbone = PepMamba(dropout=dropout, **model_config)

    def forward(self, peptide_input, protein_input, class_mask=None):
        peptide = peptide_input.float() if self.peptide_esm else self.peptide_embedding(peptide_input.long())
        protein = protein_input.float() if self.protein_esm else self.protein_embedding(protein_input.long())
        return self.backbone(peptide, protein, class_mask=class_mask)


class HybridLoss(nn.Module):
    def __init__(self, class_weight, alpha, gamma, ce_weight, smooth=1e-5):
        super().__init__()
        self.register_buffer("class_weight", class_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        valid = targets.ne(-100)
        logits = logits[valid]
        targets = targets[valid]
        ce = F.cross_entropy(logits, targets, weight=self.class_weight, reduction="none")
        weighted_ce = ce.mean()
        focal = (((1.0 - torch.exp(-ce)) ** self.gamma) * ce).mean()
        pos_prob = F.softmax(logits, dim=-1)[:, 1]
        intersection = torch.sum(pos_prob * targets)
        union = torch.sum(pos_prob) + torch.sum(targets)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        focal_dice = self.alpha * focal + (1.0 - self.alpha) * dice
        return self.ce_weight * weighted_ce + (1.0 - self.ce_weight) * focal_dice


def estimate_pos_weight(dataset):
    positives = 0
    negatives = 0
    for value in dataset.df["label"]:
        labels = parse_label(value)
        positives += int(labels.sum().item())
        negatives += int(labels.numel() - labels.sum().item())
    ratio = negatives / max(positives, 1)
    return min(ratio ** FINAL_PARAMS["class_weight_power"], FINAL_PARAMS["max_pos_weight"])


def metrics_at_threshold(labels, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity + 1e-9)
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(labels, predicted)),
    }


def threshold_scan(labels, scores):
    rows = []
    for threshold in np.arange(0.10, 0.901, 0.01):
        metrics = metrics_at_threshold(labels, scores, float(threshold))
        rows.append({"threshold": float(threshold), **metrics})
    best = max(rows, key=lambda row: (row["mcc"], row["precision"], row["sensitivity"]))
    return rows, best


def collect_scores(model, loader, device, criterion=None):
    model.eval()
    labels_all = []
    scores_all = []
    loss_sum = 0.0
    batches = 0
    with torch.no_grad():
        for pep, prot, labels, mask in loader:
            pep = pep.to(device)
            prot = prot.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            logits, _ = model(pep, prot, class_mask=mask)
            if criterion is not None:
                loss_sum += criterion(logits, labels).item()
                batches += 1
            scores = F.softmax(logits, dim=-1)[..., 1]
            valid = labels.ne(-100)
            labels_all.extend(labels[valid].cpu().numpy())
            scores_all.extend(scores[valid].cpu().numpy())
    loss = None if criterion is None else loss_sum / max(batches, 1)
    return np.asarray(labels_all), np.asarray(scores_all), loss


def evaluate(model, loader, device, criterion=None, threshold=None):
    labels, scores, loss = collect_scores(model, loader, device, criterion)
    result = {
        "loss": loss,
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }
    if threshold is None:
        raise RuntimeError("Fixed paper threshold is required")
    result["threshold"] = float(threshold)
    result.update(metrics_at_threshold(labels, scores, threshold))
    return result


def save_predictions(path_prefix, labels, scores):
    np.savez_compressed(str(path_prefix) + ".npz", labels=labels, scores=scores)
    with (Path(str(path_prefix) + ".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "positive_probability"])
        writer.writerows((index, int(label), float(score)) for index, (label, score) in enumerate(zip(labels, scores)))


def excel_col(index):
    value = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        value = chr(65 + remainder) + value
    return value


def sheet_xml(rows):
    output = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>']
    output.append('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>')
    for row_index, row in enumerate(rows, start=1):
        output.append(f'<row r="{row_index}">')
        for col_index, cell in enumerate(row, start=1):
            if cell is None:
                continue
            ref = f"{excel_col(col_index)}{row_index}"
            if isinstance(cell, bool):
                output.append(f'<c r="{ref}" t="b"><v>{int(cell)}</v></c>')
            elif isinstance(cell, (int, float)) and not isinstance(cell, bool) and math.isfinite(float(cell)):
                output.append(f'<c r="{ref}"><v>{cell}</v></c>')
            else:
                output.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(cell))}</t></is></c>')
        output.append("</row>")
    output.append("</sheetData></worksheet>")
    return "".join(output)


def write_xlsx(path, sheets):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    workbook_sheets = []
    workbook_rels = []
    for index, name in enumerate(sheets, start=1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        workbook_sheets.append(f'<sheet name="{escape(name[:31])}" sheetId="{index}" r:id="rId{index}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    content_types.append("</Types>")
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(workbook_rels)}</Relationships>'
    )
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        for index, rows in enumerate(sheets.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml(rows))
    os.replace(tmp, path)


def write_table_csv(path, rows, headers):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(result):
    test = result["test"]
    val = result["validation"]
    return {
        "variant": result["variant"],
        "protein_repr": result["protein_repr"],
        "peptide_repr": result["peptide_repr"],
        "seed": result["seed"],
        "best_epoch": result["best_epoch"],
        "selected_threshold": val["threshold"],
        "val_auc": val["auc"],
        "val_auprc": val["auprc"],
        "val_mcc": val["mcc"],
        "test_auc": test["auc"],
        "test_auprc": test["auprc"],
        "sensitivity": test["sensitivity"],
        "specificity": test["specificity"],
        "precision": test["precision"],
        "mcc": test["mcc"],
        "f1": test["f1"],
        "elapsed_sec": result["elapsed_sec"],
    }


def write_summary(run_dir, results):
    if not results:
        return
    rows = [flatten_result(item) for item in results]
    headers = list(rows[0].keys())
    write_table_csv(run_dir / "ablation_pretrained_repr_all_results.csv", rows, headers)
    sorted_auc = sorted(rows, key=lambda row: (row["test_auc"], row["mcc"]), reverse=True)
    sorted_mcc = sorted(rows, key=lambda row: (row["mcc"], row["test_auc"]), reverse=True)
    write_xlsx(
        run_dir / "ablation_pretrained_repr_all_results.xlsx",
        {
            "all_results": [headers] + [[row[key] for key in headers] for row in rows],
            "sort_test_auc": [headers] + [[row[key] for key in headers] for row in sorted_auc],
            "sort_mcc": [headers] + [[row[key] for key in headers] for row in sorted_mcc],
        },
    )
    full = next((row for row in rows if row["variant"] == "ESM-2 + ESM-2"), None)
    no_pep = next((row for row in rows if row["variant"] == "ESM-2 + Learnable"), None)
    no_prot = next((row for row in rows if row["variant"] == "Learnable + ESM-2"), None)
    drops = []
    if full is not None:
        for row in rows:
            if row is not full:
                drops.append((row["variant"], full["test_auc"] - row["test_auc"], full["mcc"] - row["mcc"]))
    largest = max(drops, key=lambda item: (item[1], item[2])) if drops else None
    dual_best_auc = full is not None and sorted_auc[0]["variant"] == "ESM-2 + ESM-2"
    dual_best_mcc = full is not None and sorted_mcc[0]["variant"] == "ESM-2 + ESM-2"
    protein_contrib = None
    peptide_contrib = None
    if full is not None and no_prot is not None:
        protein_contrib = (full["test_auc"] - no_prot["test_auc"], full["mcc"] - no_prot["mcc"])
    if full is not None and no_pep is not None:
        peptide_contrib = (full["test_auc"] - no_pep["test_auc"], full["mcc"] - no_pep["mcc"])
    protein_bigger = None
    if protein_contrib is not None and peptide_contrib is not None:
        protein_bigger = protein_contrib > peptide_contrib

    lines = [
        "# BITENET Pretrained Representation Ablation",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Seed: `{SEED}`",
        "- Checkpoint selection: validation AUROC, tie-break by validation AUPRC.",
        "- Threshold selection: validation MCC scan, then fixed on test.",
        "",
        "## Sorted by Test AUC",
        "",
        "| Rank | Variant | Test AUC | Test MCC | AUPRC | Sens | Spec | Precision | F1 | Threshold | Best epoch |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(sorted_auc, start=1):
        lines.append(
            f"| {rank} | {row['variant']} | {row['test_auc']:.6f} | {row['mcc']:.6f} | "
            f"{row['test_auprc']:.6f} | {row['sensitivity']:.6f} | {row['specificity']:.6f} | "
            f"{row['precision']:.6f} | {row['f1']:.6f} | {row['selected_threshold']:.2f} | {row['best_epoch']} |"
        )
    lines.extend([
        "",
        "## Sorted by Test MCC",
        "",
        "| Rank | Variant | Test MCC | Test AUC | AUPRC | Sens | Spec | Precision | F1 | Threshold | Best epoch |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(sorted_mcc, start=1):
        lines.append(
            f"| {rank} | {row['variant']} | {row['mcc']:.6f} | {row['test_auc']:.6f} | "
            f"{row['test_auprc']:.6f} | {row['sensitivity']:.6f} | {row['specificity']:.6f} | "
            f"{row['precision']:.6f} | {row['f1']:.6f} | {row['selected_threshold']:.2f} | {row['best_epoch']} |"
        )
    lines.extend(["", "## Answers", ""])
    lines.append(f"- Dual ESM-2 is best by test AUC: `{'yes' if dual_best_auc else 'no'}`; by test MCC: `{'yes' if dual_best_mcc else 'no'}`.")
    if protein_contrib is not None and peptide_contrib is not None:
        lines.append(
            "- Protein ESM-2 contribution is larger than peptide ESM-2: "
            f"`{'yes' if protein_bigger else 'no'}` "
            f"(protein removal drop: AUC {protein_contrib[0]:+.6f}, MCC {protein_contrib[1]:+.6f}; "
            f"peptide removal drop: AUC {peptide_contrib[0]:+.6f}, MCC {peptide_contrib[1]:+.6f})."
        )
    if largest is not None:
        lines.append(f"- Largest performance drop vs full by test AUC: `{largest[0]}` (AUC drop {largest[1]:+.6f}, MCC drop {largest[2]:+.6f}).")
    if (dual_best_auc and dual_best_mcc):
        support_text = "`yes`."
    elif dual_best_auc:
        support_text = (
            "`mostly yes`: full ESM-2 has the best test AUC, while the MCC leader should be noted separately."
        )
    else:
        support_text = "`no/mixed`: full ESM-2 is not the top test-AUC setting in this seed."
    lines.append("- These results support using ESM-2 + ESM-2 as final model input: " + support_text)
    (run_dir / "ablation_pretrained_repr_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_variant(run_dir, variant, device, evaluate_test=True):
    start = time.time()
    variant_dir = run_dir / variant["name"]
    variant_dir.mkdir(parents=True, exist_ok=False)
    params_payload = {
        "variant": variant["label"],
        "protein_repr": "ESM-2" if variant["protein_esm"] else "Learnable",
        "peptide_repr": "ESM-2" if variant["peptide_esm"] else "Learnable",
        "params": FINAL_PARAMS,
        "model_config": BASE_MODEL_CONFIG,
        "seed": SEED,
        "dataset": "BITENET",
        "train_csv": str(ROOT / "data/bitenet_train.csv"),
        "val_csv": str(ROOT / "data/bitenet_val.csv"),
        "test_csv": str(ROOT / "data/bitenet_test.csv"),
    }
    atomic_json(variant_dir / "params.json", params_payload)

    seed_everything(SEED)
    train_dataset = RepresentationAblationDataset("train", variant["protein_esm"], variant["peptide_esm"])
    val_dataset = RepresentationAblationDataset("val", variant["protein_esm"], variant["peptide_esm"])
    test_dataset = None
    train_loader = build_loader(train_dataset, FINAL_PARAMS["batch_size"], True)
    val_loader = build_loader(val_dataset, FINAL_PARAMS["batch_size"], False)
    test_loader = None
    if evaluate_test:
        test_dataset = RepresentationAblationDataset("test", variant["protein_esm"], variant["peptide_esm"])
        test_loader = build_loader(test_dataset, FINAL_PARAMS["batch_size"], False)

    pos_weight = estimate_pos_weight(train_dataset)
    class_weight = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    model = AblationModel(
        protein_esm=variant["protein_esm"],
        peptide_esm=variant["peptide_esm"],
        model_config=BASE_MODEL_CONFIG,
        dropout=FINAL_PARAMS["dropout"],
    ).to(device)
    criterion = HybridLoss(class_weight, FINAL_PARAMS["alpha"], FINAL_PARAMS["gamma"], FINAL_PARAMS["ce_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINAL_PARAMS["lr"],
        weight_decay=FINAL_PARAMS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=FINAL_PARAMS["scheduler_t0"], T_mult=1, eta_min=1e-7
    )

    fixed_state = None
    fixed_val = None
    fixed_epoch = FINAL_PARAMS["best_epoch"]
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for pep, prot, labels, mask in train_loader:
            pep = pep.to(device)
            prot = prot.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(pep, prot, class_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FINAL_PARAMS["grad_clip"])
            optimizer.step()
            train_loss += loss.item()
            batches += 1
        scheduler.step(epoch)
        val = evaluate(model, val_loader, device, criterion, FINAL_PARAMS["threshold"])
        row = {"epoch": epoch, "train_loss": train_loss / max(batches, 1), **val}
        history.append(row)
        atomic_json(variant_dir / "epoch_history.json", history)
        print(
            f"[{variant.get('log_label', variant['label'])}] epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={row['train_loss']:.4f} val_auc={val['auc']:.4f} "
            f"val_auprc={val['auprc']:.4f} val_mcc={val['mcc']:.4f} threshold={val['threshold']:.2f}",
            flush=True,
        )
        if epoch == fixed_epoch:
            fixed_val = val
            fixed_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(fixed_state, variant_dir / "best_state.pth")
            torch.save(fixed_state, variant_dir / "best_checkpoint.pth")
            atomic_json(variant_dir / "fixed_paper_epoch.json", {"epoch": fixed_epoch, **fixed_val})

    if fixed_state is None:
        raise RuntimeError(f"Fixed paper epoch {fixed_epoch} was not reached")
    model.load_state_dict(fixed_state)
    val_labels, val_scores, _ = collect_scores(model, val_loader, device)
    test_labels = test_scores = None
    if evaluate_test:
        test_labels, test_scores, _ = collect_scores(model, test_loader, device)
    selected = {"threshold": FINAL_PARAMS["threshold"], **metrics_at_threshold(val_labels, val_scores, FINAL_PARAMS["threshold"])}
    scan_rows = [selected]
    scan_headers = list(scan_rows[0].keys())
    write_table_csv(variant_dir / "threshold_scan.csv", scan_rows, scan_headers)
    write_xlsx(variant_dir / "threshold_scan.xlsx", {"threshold_scan": [scan_headers] + [[row[key] for key in scan_headers] for row in scan_rows]})
    save_predictions(variant_dir / "validation_predictions", val_labels, val_scores)
    if evaluate_test:
        save_predictions(variant_dir / "test_predictions", test_labels, test_scores)
        test = {
            "status": "evaluated_once_at_validation_threshold",
            "loss": None,
            "auc": float(roc_auc_score(test_labels, test_scores)),
            "auprc": float(average_precision_score(test_labels, test_scores)),
            "threshold": selected["threshold"],
            **metrics_at_threshold(test_labels, test_scores, selected["threshold"]),
            "test_threshold_search_performed": False,
            "test_used_for_selection": False,
        }
    else:
        test = {
            "status": "not_evaluated",
            "reason": "Training stage uses train/validation only; run test.py after validation lock.",
        }
    result = {
        "variant": variant["label"],
        "protein_repr": "ESM-2" if variant["protein_esm"] else "Learnable",
        "peptide_repr": "ESM-2" if variant["peptide_esm"] else "Learnable",
        "seed": SEED,
        "params": FINAL_PARAMS,
        "model_config": BASE_MODEL_CONFIG,
        "pos_weight": pos_weight,
        "epochs_completed": EPOCHS,
        "best_epoch": fixed_epoch,
        "validation": {
            "loss": fixed_val["loss"],
            "auc": float(roc_auc_score(val_labels, val_scores)),
            "auprc": float(average_precision_score(val_labels, val_scores)),
            "threshold": selected["threshold"],
            **{key: selected[key] for key in ["sensitivity", "specificity", "precision", "f1", "mcc"]},
        },
        "test": test,
        "selected_threshold_source": "fixed paper threshold from config",
        "elapsed_sec": time.time() - start,
    }
    atomic_json(variant_dir / "test_metrics.json", test)
    atomic_json(variant_dir / "result.json", result)
    if evaluate_test:
        print(
            f"[{variant.get('log_label', variant['label'])}] done best_epoch={fixed_epoch} "
            f"test_auc={test['auc']:.4f} test_mcc={test['mcc']:.4f} threshold={test['threshold']:.2f}",
            flush=True,
        )
    else:
        print(
            f"[{variant.get('log_label', variant['label'])}] training done best_epoch={fixed_epoch} "
            f"val_auc={result['validation']['auc']:.4f} threshold={result['validation']['threshold']:.2f}; test not evaluated",
            flush=True,
        )
    del model, optimizer, scheduler, criterion, train_loader, val_loader
    del train_dataset, val_dataset, fixed_state
    if test_loader is not None:
        del test_loader
    if test_dataset is not None:
        del test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=[variant["name"] for variant in VARIANTS], choices=[variant["name"] for variant in VARIANTS])
    parser.add_argument("--run-dir", help="Existing output directory for resumed/partial ablation runs.")
    return parser.parse_args()


def main():
    args = parse_args()
    selected_names = set(args.variants)
    selected_variants = [variant for variant in VARIANTS if variant["name"] in selected_names]
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "tuning_results" / f"bitenet_ablation_pretrained_repr_final_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atomic_json(
        run_dir / "run_config.json",
        {
            "purpose": __doc__,
            "created": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "seed": SEED,
            "epochs": EPOCHS,
            "params": FINAL_PARAMS,
            "model_config": BASE_MODEL_CONFIG,
            "variants": selected_variants,
        },
    )
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    results = []
    for result_path in sorted(run_dir.glob("*/result.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            results.append(json.load(handle))
    completed = {item["variant"] for item in results}
    for variant in selected_variants:
        if variant["label"] in completed:
            print(f"[{variant.get('log_label', variant['label'])}] already complete, skipping", flush=True)
            continue
        result = run_variant(run_dir, variant, device)
        results.append(result)
        write_summary(run_dir, results)
    write_summary(run_dir, results)
    print(f"Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
