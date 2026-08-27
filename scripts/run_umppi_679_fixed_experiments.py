#!/usr/bin/env python3
"""Run exactly two protocol-locked UMPPI 679-residue experiments.

The held-out test set is used for metrics only after each experiment's validation
checkpoint and threshold have been durably locked.  No hyperparameter search is
performed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("PEPMAMBA_DETERMINISTIC_KERNELS", "1")
os.environ.setdefault("MAMBA_DETERMINISTIC", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from models import ConvPositionEncoding, PepMamba, PepMambaLayer, ResiduePredictionHead
from umppi_reference_run import HybridLoss

DATA_DIR = ROOT / "data" / "umppi_protein_site"
EMBEDDING_DIR = ROOT / "embeddings" / "umppi_esm2_650m"
FEATURE_ZIP = ROOT / "data_feature_dict-20260812T170123Z-1-001.zip"
TRIAL003_DIR = ROOT / "tuning_results" / "umppi_validation_tuning_20260808_125913" / "trial_003"
PROTEIN_LIMIT = 679
PEPTIDE_LIMIT = 50


def json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class Logger:
    def __init__(self, path): self.path = path
    def __call__(self, message):
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with self.path.open("a") as handle: handle.write(line + "\n")


def load_pickle_from_zip(archive, member):
    with archive.open("data_feature_dict/" + member) as handle:
        return pickle.load(handle, encoding="latin1")


def raw_from_ss_key(key):
    tokens = key.split(",")
    if not tokens or any(len(token) < 2 or token[-1] not in "HCE" for token in tokens):
        raise ValueError("invalid AA+SS dictionary key")
    return "".join(token[0] for token in tokens)


class BiologicalFeatureStore:
    """Original author feature dictionaries, read directly from the supplied ZIP."""
    def __init__(self, zip_path):
        with zipfile.ZipFile(zip_path) as archive:
            self.protein_cat = load_pickle_from_zip(archive, "protein_feature_dict")
            protein_ss = load_pickle_from_zip(archive, "protein_ss_feature_dict")
            self.protein_phys = load_pickle_from_zip(archive, "protein_2_feature_dict")
            self.protein_dense = load_pickle_from_zip(archive, "protein_dense_feature_dict")
            self.peptide_cat = load_pickle_from_zip(archive, "peptide_feature_dict")
            peptide_ss = load_pickle_from_zip(archive, "peptide_ss_feature_dict")
            self.peptide_phys = load_pickle_from_zip(archive, "peptide_2_feature_dict")
            self.peptide_dense = load_pickle_from_zip(archive, "peptide_dense_feature_dict")
            self.pad_protein = int(np.load(archive.open("data_feature_dict/pad_prot_len.npy")))
            self.pad_peptide = int(np.load(archive.open("data_feature_dict/pad_pep_len.npy")))
        self.protein_ss = self._rekey_ss(protein_ss, "protein")
        self.peptide_ss = self._rekey_ss(peptide_ss, "peptide")
        if self.pad_protein != PROTEIN_LIMIT or self.pad_peptide != PEPTIDE_LIMIT:
            raise ValueError(f"feature pad protocol mismatch: {self.pad_protein}/{self.pad_peptide}")

    @staticmethod
    def _rekey_ss(source, kind):
        result = {}
        for key, value in source.items():
            sequence = raw_from_ss_key(key)
            if sequence in result: raise ValueError(f"duplicate reconstructed {kind} SS key")
            result[sequence] = value
        return result

    def arrays(self, kind, sequence, length):
        if kind == "protein":
            dictionaries = (self.protein_cat, self.protein_ss, self.protein_phys, self.protein_dense)
            expected = ((PROTEIN_LIMIT,), (PROTEIN_LIMIT,), (PROTEIN_LIMIT,), (PROTEIN_LIMIT, 23))
        else:
            dictionaries = (self.peptide_cat, self.peptide_ss, self.peptide_phys, self.peptide_dense)
            expected = ((PEPTIDE_LIMIT,), (PEPTIDE_LIMIT,), (PEPTIDE_LIMIT,), (PEPTIDE_LIMIT, 3))
        names = ("categorical", "secondary_structure", "physicochemical", "dense")
        values = []
        for name, dictionary, shape in zip(names, dictionaries, expected):
            if sequence not in dictionary: raise KeyError(f"missing {kind} {name} feature for sequence")
            value = np.asarray(dictionary[sequence])
            if value.shape != shape: raise ValueError(f"{kind} {name} shape {value.shape} != {shape}")
            values.append(value)
        cat, ss, phys, dense = values
        if not np.all((cat[:length] >= 1) & (cat[:length] <= 21)):
            raise ValueError(f"{kind} categorical non-padding positions invalid")
        if not np.all((ss[:length] >= 1) & (ss[:length] <= 63)):
            raise ValueError(f"{kind} SS non-padding positions invalid")
        if not np.all((phys[:length] >= 1) & (phys[:length] <= 7)):
            raise ValueError(f"{kind} physicochemical non-padding positions invalid")
        return (
            torch.tensor(cat[:length], dtype=torch.long),
            torch.tensor(ss[:length], dtype=torch.long),
            torch.tensor(phys[:length], dtype=torch.long),
            torch.tensor(dense[:length], dtype=torch.float32),
        )


class UMppi679Dataset(Dataset):
    def __init__(self, csv_path, biological_store=None):
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        manifest = pd.read_csv(EMBEDDING_DIR / "manifest.csv", dtype=str, keep_default_na=False)
        if manifest.sequence_hash.duplicated().any(): raise ValueError("duplicate embedding manifest hash")
        self.manifest = manifest.set_index("sequence_hash").to_dict("index")
        self.bio = biological_store

    def __len__(self): return len(self.frame)

    def embedding(self, sequence_hash, original_length, sequence_type, retained_length):
        if sequence_hash not in self.manifest: raise KeyError(f"missing ESM embedding {sequence_hash}")
        item = self.manifest[sequence_hash]
        if item["sequence_type"] != sequence_type: raise ValueError(f"wrong embedding type: {sequence_hash}")
        value = np.load(ROOT / item["file_path"], mmap_mode="r")
        if value.shape != (original_length, 1280):
            raise ValueError(f"ESM shape mismatch {value.shape} != ({original_length},1280)")
        return torch.from_numpy(np.array(value[:retained_length], dtype=np.float32, copy=True))

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        protein, peptide, label_text = str(row.prot_seq), str(row.pep_seq), str(row.protein_residue_label)
        if set(label_text) - {"0", "1"}: raise ValueError(f"invalid labels: {row.record_id}")
        if len(label_text) != len(protein): raise ValueError(f"protein/label mismatch: {row.record_id}")
        if len(peptide) > PEPTIDE_LIMIT: raise ValueError(f"peptide exceeds protocol: {row.record_id}")
        protein_length = min(len(protein), PROTEIN_LIMIT)
        peptide_length = min(len(peptide), PEPTIDE_LIMIT)
        result = {
            "record_id": row.record_id,
            "dataset_index": int(row.dataset_index),
            "protein": self.embedding(row.protein_sequence_hash, len(protein), "protein", protein_length),
            "peptide": self.embedding(row.peptide_sequence_hash, len(peptide), "peptide", peptide_length),
            "labels": torch.tensor([int(x) for x in label_text[:protein_length]], dtype=torch.long),
            "protein_length": protein_length,
            "peptide_length": peptide_length,
            "original_protein_length": len(protein),
        }
        if result["protein"].shape[0] != result["labels"].shape[0]:
            raise ValueError(f"retained ESM/label mismatch: {row.record_id}")
        if self.bio is not None:
            result["protein_bio"] = self.bio.arrays("protein", protein, protein_length)
            result["peptide_bio"] = self.bio.arrays("peptide", peptide, peptide_length)
        return result


def collate(batch):
    def pad(name, value=0.0): return pad_sequence([x[name] for x in batch], batch_first=True, padding_value=value)
    protein = pad("protein")
    peptide = pad("peptide")
    labels = pad("labels", -100)
    pl = torch.tensor([x["protein_length"] for x in batch], dtype=torch.long)
    ql = torch.tensor([x["peptide_length"] for x in batch], dtype=torch.long)
    protein_mask = torch.arange(protein.shape[1]).unsqueeze(0) >= pl.unsqueeze(1)
    peptide_mask = torch.arange(peptide.shape[1]).unsqueeze(0) >= ql.unsqueeze(1)
    if not torch.equal(protein_mask, labels.eq(-100)): raise ValueError("protein mask/labels misaligned")
    out = {"record_ids": [x["record_id"] for x in batch], "dataset_indices": [x["dataset_index"] for x in batch],
           "protein": protein, "peptide": peptide, "labels": labels, "protein_pad_mask": protein_mask,
           "peptide_pad_mask": peptide_mask, "protein_lengths": pl, "peptide_lengths": ql}
    if "protein_bio" in batch[0]:
        for side in ("protein", "peptide"):
            features = list(zip(*(x[f"{side}_bio"] for x in batch)))
            out[f"{side}_cat"] = pad_sequence(features[0], batch_first=True, padding_value=0)
            out[f"{side}_ss"] = pad_sequence(features[1], batch_first=True, padding_value=0)
            out[f"{side}_phys"] = pad_sequence(features[2], batch_first=True, padding_value=0)
            out[f"{side}_dense"] = pad_sequence(features[3], batch_first=True, padding_value=0.0)
    return out


class BioFeatureEncoder(nn.Module):
    def __init__(self, dense_width, d_model, include_pssm):
        super().__init__()
        self.include_pssm = include_pssm
        self.aa = nn.Embedding(22, 8, padding_idx=0)
        self.ss = nn.Embedding(64, 8, padding_idx=0)
        self.phys = nn.Embedding(8, 4, padding_idx=0)
        if include_pssm:
            if dense_width != 23: raise ValueError("protein dense feature must be 20 PSSM + 3 disorder")
            self.pssm = nn.Sequential(nn.Linear(20, 16), nn.GELU())
            self.disorder = nn.Sequential(nn.Linear(3, 4), nn.GELU())
            total = 40
        else:
            if dense_width != 3: raise ValueError("peptide dense feature must contain only 3 disorder channels")
            self.disorder = nn.Sequential(nn.Linear(3, 4), nn.GELU())
            total = 24
        self.project = nn.Sequential(nn.Linear(total, d_model), nn.LayerNorm(d_model), nn.GELU())

    def forward(self, cat, ss, phys, dense):
        pieces = [self.aa(cat), self.ss(ss), self.phys(phys)]
        if self.include_pssm: pieces.extend([self.pssm(dense[..., :20]), self.disorder(dense[..., 20:])])
        else: pieces.append(self.disorder(dense))
        return self.project(torch.cat(pieces, dim=-1))


class BioPepMamba(nn.Module):
    """PepMamba body unchanged; auxiliary encodings are fused at its input projections."""
    def __init__(self, n_layers, d_model, n_head, d_k, d_v, d_inner, dropout):
        super().__init__()
        self.protein_proj = nn.Linear(1280, d_model)
        self.protein_skip_proj = nn.Linear(1280, d_model)
        self.peptide_proj = nn.Linear(1280, d_model)
        self.protein_bio = BioFeatureEncoder(23, d_model, True)
        self.peptide_bio = BioFeatureEncoder(3, d_model, False)
        self.protein_gate = nn.Parameter(torch.tensor(0.0))
        self.peptide_gate = nn.Parameter(torch.tensor(0.0))
        self.protein_pos = ConvPositionEncoding(d_model)
        self.peptide_pos = ConvPositionEncoding(d_model)
        self.layers = nn.ModuleList([PepMambaLayer(d_model, d_inner, n_head, d_k, d_v, dropout) for _ in range(n_layers)])
        self.head = ResiduePredictionHead(d_model, dropout)

    def forward(self, peptide, protein, protein_mask, peptide_mask, protein_bio, peptide_bio):
        paux = self.protein_bio(*protein_bio)
        qaux = self.peptide_bio(*peptide_bio)
        pg, qg = torch.sigmoid(self.protein_gate), torch.sigmoid(self.peptide_gate)
        ph = self.protein_proj(protein) + pg * paux
        ps = self.protein_skip_proj(protein) + pg * paux
        qh = self.peptide_proj(peptide) + qg * qaux
        ph, qh = self.protein_pos(ph), self.peptide_pos(qh)
        attention = []
        for layer in self.layers:
            ph, qh, attn, _ = layer(ph, qh, protein_mask=protein_mask, peptide_mask=peptide_mask)
            attention.append(attn)
        logits, cls_attn = self.head(ps, ph, qh, peptide_mask, protein_mask)
        attention.append({"cls_transformer_attn": cls_attn})
        return logits, attention


def forward_model(model, batch, device, biological):
    protein = batch["protein"].to(device, non_blocking=True)
    peptide = batch["peptide"].to(device, non_blocking=True)
    pmask = batch["protein_pad_mask"].to(device, non_blocking=True)
    if not biological: return model(peptide, protein, class_mask=pmask)
    qmask = batch["peptide_pad_mask"].to(device, non_blocking=True)
    pbio = tuple(batch[f"protein_{x}"].to(device, non_blocking=True) for x in ("cat", "ss", "phys", "dense"))
    qbio = tuple(batch[f"peptide_{x}"].to(device, non_blocking=True) for x in ("cat", "ss", "phys", "dense"))
    return model(peptide, protein, pmask, qmask, pbio, qbio)


def mcc_counts(tp, tn, fp, fn):
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return float((tp * tn - fp * fn) / math.sqrt(denominator)) if denominator > 0 else 0.0


def binary(y, p, threshold):
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1); sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1); f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-15)
    return {"MCC": mcc_counts(tp, tn, fp, fn), "F1": f1, "Precision": precision,
            "Sensitivity": sensitivity, "Specificity": specificity, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def collect(model, loader, device, biological):
    model.eval(); ys, ps, records = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits, _ = forward_model(model, batch, device, biological)
            if logits.shape[:2] != labels.shape: raise RuntimeError("prediction/label padded shape mismatch")
            probabilities = F.softmax(logits, dim=-1)[..., 1]
            for i, record_id in enumerate(batch["record_ids"]):
                length = int(batch["protein_lengths"][i])
                y = labels[i, :length].cpu().numpy(); p = probabilities[i, :length].cpu().numpy()
                if len(y) != length or len(p) != length: raise RuntimeError(f"alignment failure {record_id}")
                ys.append(y); ps.append(p); records.append({"record_id": record_id, "y": y, "p": p})
    return np.concatenate(ys), np.concatenate(ps), records


def threshold_scan(records, pooled_y, pooled_p):
    rows = []
    for threshold in np.round(np.arange(0.05, 0.9501, 0.005), 3):
        record_mcc = [binary(r["y"], r["p"], threshold)["MCC"] for r in records if len(np.unique(r["y"])) == 2]
        pooled = binary(pooled_y, pooled_p, threshold)
        rows.append({"threshold": float(threshold), "mean_record_MCC": float(np.mean(record_mcc)),
                     "n_valid_records": len(record_mcc), **{f"pooled_{k}": v for k, v in pooled.items()}})
    chosen = max(rows, key=lambda r: (r["mean_record_MCC"], r["pooled_MCC"], r["pooled_F1"], -r["threshold"]))
    return rows, chosen


def summarize(y, p, records, threshold):
    valid = [r for r in records if len(np.unique(r["y"])) == 2]
    record_auc = [roc_auc_score(r["y"], r["p"]) for r in valid]
    record_mcc = [binary(r["y"], r["p"], threshold)["MCC"] for r in valid]
    return {"mean_record_AUROC": float(np.mean(record_auc)), "mean_record_MCC": float(np.mean(record_mcc)),
            "n_valid_records": len(valid), "pooled_AUROC": float(roc_auc_score(y, p)),
            "pooled_AUPRC": float(average_precision_score(y, p)), **binary(y, p, threshold),
            "threshold": float(threshold)}


def save_predictions(path, y, p, records):
    np.savez_compressed(path, labels=y.astype(np.int8), probabilities=p.astype(np.float32),
                        record_ids=np.asarray([r["record_id"] for r in records], dtype=str),
                        record_lengths=np.asarray([len(r["y"]) for r in records], dtype=np.int64))


def make_batches(lengths, nominal, n_layers, d_model, seed, effective_target):
    budget = 48_000_000 * (3 * 128) / (n_layers * d_model)
    order = np.argsort(lengths); batches = []; i = 0
    while i < len(order):
        length = max(int(lengths[order[i]]), 1)
        capacity = max(1, min(nominal, int(budget / (length * length))))
        batch = order[i:i + capacity].tolist(); batches.append(batch); i += len(batch)
    random.Random(seed).shuffle(batches)
    denominators = [0] * len(batches); step_after = [False] * len(batches); group = []; count = 0
    for index, batch in enumerate(batches):
        if group and count + len(batch) > effective_target:
            for j in group: denominators[j] = count
            step_after[group[-1]] = True; group = []; count = 0
        group.append(index); count += len(batch)
    if group:
        for j in group: denominators[j] = count
        step_after[group[-1]] = True
    return batches, denominators, step_after


def preflight(store, output_dir):
    rows = []
    for split in ("train", "validation", "test"):
        dataset = UMppi679Dataset(DATA_DIR / f"{split}.csv", store)
        truncated = 0; retained = 0; original = 0
        for index in range(len(dataset)):
            item = dataset[index]
            p_len, q_len = item["protein_length"], item["peptide_length"]
            original += item["original_protein_length"]; retained += p_len
            truncated += int(item["original_protein_length"] > PROTEIN_LIMIT)
            if item["protein"].shape != (p_len, 1280) or item["labels"].shape != (p_len,):
                raise RuntimeError(f"preflight protein alignment failure: {item['record_id']}")
            if item["peptide"].shape != (q_len, 1280): raise RuntimeError(f"preflight peptide alignment failure: {item['record_id']}")
            if any(x.shape[0] != p_len for x in item["protein_bio"]): raise RuntimeError("protein bio position mismatch")
            if any(x.shape[0] != q_len for x in item["peptide_bio"]): raise RuntimeError("peptide bio position mismatch")
        rows.append({"split": split, "records": len(dataset), "truncated_protein_records": truncated,
                     "original_residues": original, "retained_residues": retained,
                     "retained_pct": 100 * retained / original, "protein_limit": PROTEIN_LIMIT,
                     "peptide_limit": PEPTIDE_LIMIT, "alignment_pass": True})
    pd.DataFrame(rows).to_csv(output_dir / "feature_sequence_label_alignment_check.csv", index=False)
    write_json(output_dir / "preflight_status.json", {"passed": True, "rows": rows,
               "protein_dense_semantics": "columns 0-19 PSSM; columns 20-22 intrinsic disorder",
               "peptide_dense_semantics": "3 intrinsic-disorder channels; no peptide PSSM provided",
               "test_metrics_computed": False, "test_used_for_selection": False})


def run_experiment(name, biological, cfg, store, root_out, device):
    out = root_out / name
    out.mkdir(parents=True, exist_ok=False)
    log = Logger(out / "train.log")
    train_ds = UMppi679Dataset(DATA_DIR / "train.csv", store if biological else None)
    val_ds = UMppi679Dataset(DATA_DIR / "validation.csv", store if biological else None)
    if (len(train_ds), len(val_ds)) != (4726, 539): raise RuntimeError("split count changed")
    seed_all(42)
    if biological:
        model = BioPepMamba(cfg["n_layers"], cfg["d_model"], cfg["cross_attention_heads"], cfg["d_k"], cfg["d_v"],
                            cfg["d_model"] * cfg["d_inner_multiplier"], cfg["dropout"]).to(device)
    else:
        model = PepMamba(n_layers=cfg["n_layers"], d_model=cfg["d_model"], n_head=cfg["cross_attention_heads"],
                         d_k=cfg["d_k"], d_v=cfg["d_v"], d_inner=cfg["d_model"] * cfg["d_inner_multiplier"],
                         dropout=cfg["dropout"]).to(device)
    config = {**cfg, "experiment": name, "biological_features": biological, "protein_max_length": PROTEIN_LIMIT,
              "peptide_max_length": PEPTIDE_LIMIT, "protein_truncation": "retain first 679 residues and labels",
              "peptide_truncation": "retain first 50 residues", "model_parameters": sum(p.numel() for p in model.parameters()),
              "checkpoint_selection": ["validation mean-record AUROC", "validation mean-record MCC", "validation pooled AUPRC"],
              "threshold_selection": "validation mean-record MCC; ties pooled MCC, F1, lower threshold",
              "test_used_for_selection": False, "test_threshold_search_performed": False,
              "feature_source": str(FEATURE_ZIP) if biological else None,
              "peptide_pssm_used": False}
    write_json(out / "config.json", config)
    text = "".join(train_ds.frame.protein_residue_label.str.slice(0, PROTEIN_LIMIT))
    positives, negatives = text.count("1"), text.count("0")
    pos_weight = min((negatives / positives) ** cfg["class_weight_power"], cfg["max_pos_weight"])
    config["resolved_positive_class_weight"] = pos_weight; write_json(out / "config.json", config)
    criterion = HybridLoss(cfg["ce_weight"], torch.tensor([1.0, pos_weight], device=device), cfg["focal_alpha"], cfg["focal_gamma"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-7)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)
    lengths = np.minimum(train_ds.frame.prot_seq.str.len().to_numpy(), PROTEIN_LIMIT)
    history = []; best_key = None; checkpoint = out / "best_validation_checkpoint.pth"
    for epoch in range(1, cfg["epochs"] + 1):
        batches, denominators, step_after = make_batches(lengths, cfg["batch_size"], cfg["n_layers"], cfg["d_model"], 42 + epoch, cfg["effective_batch_size"])
        loader = DataLoader(train_ds, batch_sampler=batches, num_workers=0, collate_fn=collate, pin_memory=True)
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(loader, 1):
            targets = batch["labels"].to(device, non_blocking=True)
            logits, _ = forward_model(model, batch, device, biological)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss epoch={epoch} step={step}")
            count = len(batch["record_ids"]); (loss * count / denominators[step - 1]).backward(); losses.append(float(loss.detach()))
            if step_after[step - 1]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        vy, vp, records = collect(model, val_loader, device, biological)
        scan_rows, chosen = threshold_scan(records, vy, vp)
        summary = summarize(vy, vp, records, chosen["threshold"])
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": float(np.mean(losses)), **summary}
        history.append(row); pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        key = (summary["mean_record_AUROC"], summary["mean_record_MCC"], summary["pooled_AUPRC"])
        if best_key is None or key > best_key:
            best_key = key
            torch.save({"model_state_dict": model.state_dict(), "best_epoch": epoch, "selection_key": key, "config": config}, checkpoint)
        log(f"{name} epoch {epoch:02d}/35 loss={row['train_loss']:.6f} mean_record_auc={summary['mean_record_AUROC']:.6f} mean_record_mcc={summary['mean_record_MCC']:.6f} pooled_auprc={summary['pooled_AUPRC']:.6f}")
        scheduler.step(epoch)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"]); model.eval()
    vy, vp, records = collect(model, val_loader, device, biological)
    scan_rows, chosen = threshold_scan(records, vy, vp)
    threshold = chosen["threshold"]; validation = summarize(vy, vp, records, threshold)
    pd.DataFrame(scan_rows).to_csv(out / "validation_threshold_scan.csv", index=False)
    save_predictions(out / "validation_predictions.npz", vy, vp, records)
    digest = sha256(checkpoint)
    lock = {"experiment": name, "best_epoch": state["best_epoch"], **validation, "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest, "locked_at": datetime.now(timezone.utc).isoformat(),
            "test_loaded_before_validation_lock": False, "test_used_for_selection": False,
            "test_threshold_search_performed": False}
    write_json(out / "validation_metrics.json", validation)
    write_json(out / "selected_validation_threshold.json", lock)
    (out / "checkpoint_sha256.txt").write_text(digest + "\n")
    log(f"VALIDATION_LOCKED best_epoch={state['best_epoch']} threshold={threshold:.3f} sha256={digest}")
    # The held-out dataset is instantiated only after the durable validation lock.
    test_ds = UMppi679Dataset(DATA_DIR / "test.csv", store if biological else None)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)
    ty, tp, trecords = collect(model, test_loader, device, biological)
    test = summarize(ty, tp, trecords, threshold)
    save_predictions(out / "test_predictions.npz", ty, tp, trecords)
    test.update({"best_epoch": state["best_epoch"], "checkpoint_sha256": digest,
                 "checkpoint_selected_on_validation_only": True, "threshold_selected_on_validation_only": True,
                 "test_threshold_search_performed": False, "test_used_for_selection": False})
    write_json(out / "test_metrics.json", test)
    log(f"TEST_COMPLETE mean_record_auc={test['mean_record_AUROC']:.6f} mean_record_mcc={test['mean_record_MCC']:.6f} pooled_auc={test['pooled_AUROC']:.6f} pooled_auprc={test['pooled_AUPRC']:.6f}")
    del model, optimizer, criterion, train_ds, val_ds, test_ds
    gc.collect(); torch.cuda.empty_cache()
    return {"method": name, **test}


def comparison(root_out, rows):
    previous = pd.read_csv(ROOT / "tuning_results" / "umppi_validation_tuning_20260808_125913" /
                           "exploratory_test_trial_001_004" / "exploratory_test_comparison.csv")
    trial = previous.loc[previous.trial_id.eq("trial_003")].iloc[0]
    references = [
        {"method": "CPPIF (reported)", "mean_record_AUROC": np.nan, "mean_record_MCC": np.nan},
        {"method": "UMPPI (reported)", "mean_record_AUROC": 0.914, "mean_record_MCC": 0.489},
        {"method": "PepMamba full-length trial_003", "mean_record_AUROC": trial.mean_record_AUROC,
         "mean_record_MCC": trial.mean_record_MCC, "pooled_AUROC": trial.pooled_AUROC,
         "pooled_AUPRC": trial.pooled_AUPRC, "MCC": trial.pooled_MCC, "F1": trial.F1,
         "Precision": trial.precision, "Sensitivity": trial.sensitivity, "Specificity": trial.specificity,
         "threshold": trial.fixed_validation_threshold},
    ]
    all_rows = references + rows
    columns = ["method", "mean_record_AUROC", "mean_record_MCC", "pooled_AUROC", "pooled_AUPRC", "MCC", "F1",
               "Precision", "Sensitivity", "Specificity", "threshold", "best_epoch", "checkpoint_sha256"]
    frame = pd.DataFrame(all_rows).reindex(columns=columns)
    frame.to_csv(root_out / "comparison_with_CPPIF_UMPPI_and_trial003.csv", index=False)
    def show(value):
        return "–" if pd.isna(value) else (f"{float(value):.6f}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value))
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows(): lines.append("| " + " | ".join(show(row[column]) for column in columns) + " |")
    table = "\n".join(lines)
    note = ("# UMPPI 679 fixed-experiment comparison\n\n" + table +
            "\n\n- CPPIF numeric reference fields are left unresolved rather than guessed because the supplied local CPPIF source/report was not present during the feature audit.\n"
            "- UMPPI reported mean-record AUROC/MCC: 0.914/0.489.\n"
            "- Both new checkpoints and thresholds were selected using validation only; test was evaluated once per locked model.\n")
    (root_out / "FINAL_COMPARISON_REPORT.md").write_text(note)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = json.loads((TRIAL003_DIR / "config.json").read_text())
    required = {"trial_id", "seed", "epochs", "learning_rate", "dropout", "focal_alpha", "focal_gamma", "ce_weight",
                "weight_decay", "class_weight_power", "max_pos_weight", "batch_size", "n_layers", "d_model",
                "cross_attention_heads", "d_k", "d_v", "d_inner_multiplier", "effective_batch_size", "grad_clip"}
    if not required.issubset(cfg): raise ValueError("trial_003 config incomplete")
    if cfg["seed"] != 42 or cfg["epochs"] != 35: raise ValueError("trial_003 seed/epoch protocol changed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root_out = args.output_dir or ROOT / "tuning_results" / f"umppi_679_fixed_experiments_{stamp}"
    root_out.mkdir(parents=True, exist_ok=False)
    write_json(root_out / "run_manifest.json", {"experiments": ["PepMamba_679_baseline", "PepMamba_679_biological_features"],
               "trial_003_config": cfg, "feature_zip": str(FEATURE_ZIP), "started_at": datetime.now(timezone.utc).isoformat(),
               "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                               "platform": platform.platform()}, "additional_hyperparameter_tuning": False})
    print("Loading original feature dictionaries for strict preflight...", flush=True)
    store = BiologicalFeatureStore(FEATURE_ZIP)
    preflight(store, root_out)
    print("PREFLIGHT PASSED: sequence/label/ESM/biological feature positions aligned", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [run_experiment("PepMamba_679_baseline", False, cfg, None, root_out, device),
            run_experiment("PepMamba_679_biological_features", True, cfg, store, root_out, device)]
    comparison(root_out, rows)
    write_json(root_out / "run_complete.json", {"completed": True, "completed_at": datetime.now(timezone.utc).isoformat(),
               "experiments": rows, "test_used_for_selection": False, "test_threshold_search_performed": False})
    print(json.dumps({"output_dir": str(root_out), "results": rows}, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
