#!/usr/bin/env python3
"""Small, validation-selected PepMamba tuning study for TS092/PepNN only.

Strict protocol
---------------
* Train, checkpoint, threshold, and hyperparameter selection use validation only.
* Trials are ranked by validation MCC, then AUPRC, AUROC, and the prespecified composite.
* The test Dataset is constructed only after the validation winner is locked.
* Exactly that one winner is evaluated once at its fixed validation threshold.
* Test metrics are presentation-only and never alter the selected trial.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PEPMAMBA_DETERMINISTIC_KERNELS", "1")
os.environ.setdefault("MAMBA_DETERMINISTIC", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROTEIN_PKL = ROOT / "protein_esm2.pkl"
PEPTIDE_PKL = ROOT / "peptide_esm2.pkl"
RESULTS_ROOT = ROOT / "runs" / "final"
SCRIPT_PATH = ROOT / "scripts" / "final" / "run_pepnn.py"
MODEL_FIXED = {"d_model": 128, "n_head": 4, "d_k": 32, "d_v": 32, "d_inner": 256}
SEARCH_SPACE = {
    "lr": {"type": "log_uniform", "low": 8e-5, "high": 2.5e-4},
    "dropout": {"type": "uniform", "low": 0.18, "high": 0.38},
    "focal_alpha": {"type": "uniform", "low": 0.60, "high": 0.82},
    "focal_gamma": {"type": "uniform", "low": 1.4, "high": 3.0},
    "ce_weight": {"type": "uniform", "low": 0.45, "high": 0.75},
    "weight_decay": {"type": "log_uniform", "low": 5e-5, "high": 3e-3},
    "batch_size": {"type": "categorical", "values": [8, 16, 24]},
    "class_weight_power": {"type": "uniform", "low": 0.30, "high": 0.70},
    "max_pos_weight": {"type": "uniform", "low": 5.0, "high": 10.0},
    "n_layers": {"type": "categorical", "values": [2, 3]},
}
COMPARATORS = {
    "PepNN-Seq": {"sensitivity": None, "specificity": None, "precision": None, "mcc": .272, "auc": .781},
    "PepCA original": {"sensitivity": .405, "specificity": .935, "precision": .434, "mcc": .351, "auc": .817},
}



def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def git_info():
    result = {}
    for key, command in {
        "commit": ["git", "rev-parse", "HEAD"],
        "status_porcelain": ["git", "status", "--porcelain"],
    }.items():
        try:
            result[key] = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=True
            ).stdout.strip()
        except Exception as exc:
            result[key] = f"unavailable: {exc}"
    return result


def audit_data(dataset):
    required = ["pdb_id", "Prot_seq", "Pep_seq", "label"]
    valid_columns = [required, required + ["weight"]]
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
    audit = {"dataset": dataset, "created": utc_now(), "files": {}}
    for split in ("train", "val", "test"):
        path = DATA_DIR / f"{dataset}_{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype=str)
        if list(frame.columns) not in valid_columns:
            raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
        if "weight" in frame.columns:
            weights = pd.to_numeric(frame["weight"], errors="coerce")
            if weights.isna().any() or not np.isfinite(weights).all():
                raise ValueError(f"Invalid weight values in {path}")
        problems = {"null_rows": 0, "invalid_labels": 0, "length_mismatches": 0,
                    "invalid_protein_sequences": 0, "invalid_peptide_sequences": 0}
        positive = negative = 0
        for _, row in frame.iterrows():
            if row.isna().any():
                problems["null_rows"] += 1
                continue
            protein = row["Prot_seq"].strip(); peptide = row["Pep_seq"].strip(); label = row["label"].strip()
            problems["invalid_labels"] += int(bool(set(label) - {"0", "1"}) or not label)
            problems["length_mismatches"] += int(len(protein) != len(label))
            problems["invalid_protein_sequences"] += int(bool(set(protein) - valid_aa))
            problems["invalid_peptide_sequences"] += int(bool(set(peptide) - valid_aa))
            positive += label.count("1"); negative += label.count("0")
        if any(problems.values()):
            raise ValueError(f"Data validation failed for {path}: {problems}")
        audit["files"][split] = {
            "path": str(path), "sha256": sha256(path), "rows": int(len(frame)),
            "columns": list(frame.columns), "has_weight_column": "weight" in frame.columns,
            "unique_proteins": int(frame["Prot_seq"].nunique()),
            "unique_peptides": int(frame["Pep_seq"].nunique()),
            "positive_residues": positive, "negative_residues": negative,
            "protein_length_min": int(frame["Prot_seq"].str.len().min()),
            "protein_length_max": int(frame["Prot_seq"].str.len().max()),
            "peptide_length_min": int(frame["Pep_seq"].str.len().min()),
            "peptide_length_max": int(frame["Pep_seq"].str.len().max()),
            **problems,
        }
    return audit


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_configs(n_trials, seed):
    if n_trials < 1:
        raise ValueError("--trials must be >= 1")
    rng = np.random.default_rng(seed)
    configs = [{
        "trial_id": "trial_000_anchor", "lr": 1.5e-4, "dropout": .30,
        "focal_alpha": .70, "focal_gamma": 2.0, "ce_weight": .50,
        "weight_decay": 1e-3, "batch_size": 16,
        "class_weight_power": .50, "max_pos_weight": 6.0, "n_layers": 3,
    }]
    for index in range(1, n_trials):
        configs.append({
            "trial_id": f"trial_{index:03d}",
            "lr": log_uniform(rng, 8e-5, 2.5e-4),
            "dropout": float(rng.uniform(.18, .38)),
            "focal_alpha": float(rng.uniform(.60, .82)),
            "focal_gamma": float(rng.uniform(1.4, 3.0)),
            "ce_weight": float(rng.uniform(.45, .75)),
            "weight_decay": log_uniform(rng, 5e-5, 3e-3),
            "batch_size": int(rng.choice([8, 16, 24])),
            "class_weight_power": float(rng.uniform(.30, .70)),
            "max_pos_weight": float(rng.uniform(5.0, 10.0)),
            "n_layers": int(rng.choice([2, 3])),
        })
    return configs


def verify_config(config):
    bounds = {
        "lr": (8e-5, 2.5e-4), "dropout": (.18, .38),
        "focal_alpha": (.60, .82), "focal_gamma": (1.4, 3.0),
        "ce_weight": (.45, .75), "weight_decay": (5e-5, 3e-3),
        "class_weight_power": (.30, .70), "max_pos_weight": (5.0, 10.0),
    }
    for key, (low, high) in bounds.items():
        if not low <= float(config[key]) <= high:
            raise ValueError(f"{key} outside search range: {config[key]}")
    if int(config["batch_size"]) not in (8, 16, 24) or int(config["n_layers"]) not in (2, 3):
        raise ValueError(f"Invalid categorical config: {config}")


def confusion_metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1); specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    return {
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "precision": float(precision), "f1": float(f1),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def scan_thresholds(labels, scores, threshold_min, threshold_max, threshold_step):
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    auprc = float(average_precision_score(labels, scores))
    rows = [{"threshold": float(value), **confusion_metrics(labels, scores, float(value))}
            for value in thresholds]
    # Threshold selection is validation MCC, then validation F1, then validation AUPRC.
    # AUPRC is threshold-independent and therefore constant within a trial.
    selected = max(rows, key=lambda row: (row["mcc"], row["f1"], auprc,
                                           row["sensitivity"], row["precision"], -row["threshold"]))
    return rows, selected


def collect_scores(model, loader, criterion, device):
    model.eval(); labels_all = []; scores_all = []; loss_sum = 0.0; batches = 0
    with torch.no_grad():
        for peptide, protein, labels, mask in loader:
            peptide = peptide.to(device); protein = protein.to(device)
            labels = labels.to(device); mask = mask.to(device)
            logits, _ = model(peptide, protein, class_mask=mask)
            if criterion is not None:
                loss_sum += criterion(logits, labels).item(); batches += 1
            probabilities = F.softmax(logits, dim=-1)[..., 1]
            valid = labels.ne(-100)
            labels_all.extend(labels[valid].cpu().numpy())
            scores_all.extend(probabilities[valid].cpu().numpy())
    return np.asarray(labels_all), np.asarray(scores_all), (
        None if criterion is None else loss_sum / max(batches, 1)
    )


def evaluate_validation(model, loader, criterion, device, args):
    labels, scores, loss = collect_scores(model, loader, criterion, device)
    scan, selected = scan_thresholds(
        labels, scores, args.threshold_min, args.threshold_max, args.threshold_step
    )
    if getattr(args, "unified_selection", False):
        selected = max(scan, key=lambda row: row["mcc"])
    metrics = {
        "loss": loss, "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "threshold": selected["threshold"],
        **{key: selected[key] for key in ("sensitivity", "specificity", "precision", "f1", "mcc", "tp", "fp", "tn", "fn")},
    }
    metrics["composite"] = 0.45 * metrics["mcc"] + 0.35 * metrics["auprc"] + 0.20 * metrics["auc"]
    return labels, scores, scan, metrics


def save_predictions(prefix, labels, scores):
    np.savez_compressed(str(prefix) + ".npz", labels=labels, scores=scores)
    with Path(str(prefix) + ".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "positive_probability"])
        writer.writerows((index, int(label), float(score))
                         for index, (label, score) in enumerate(zip(labels, scores)))


def estimate_weight(dataset, power, maximum, device):
    positive = negative = 0
    for value in dataset.df["label"]:
        label = str(value); positive += label.count("1"); negative += label.count("0")
    pos_weight = min((negative / max(positive, 1)) ** power, maximum)
    return torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device), {
        "positive_residues": positive, "negative_residues": negative,
        "raw_ratio": negative / max(positive, 1), "effective_pos_weight": pos_weight,
    }


def validation_rank(result):
    metrics = result["validation"]
    return metrics["mcc"], metrics["auprc"], metrics["auc"], metrics["composite"]


def validation_rank_composite(result):
    metrics = result["validation"]
    return metrics["composite"], metrics["mcc"], metrics["auprc"], metrics["auc"]


def result_row(result):
    config = result["config"]; validation = result["validation"]
    test = result.get("test", {})
    row = {"trial_id": result["trial_id"], **config,
           **{f"validation_{key}": value for key, value in validation.items()},
           "best_epoch": result["best_epoch"], "checkpoint": result["checkpoint"],
           "checkpoint_sha256": result["checkpoint_sha256"],
           "test_status": test.get("status", "not_evaluated")}
    for key in ("auc", "auprc", "threshold", "sensitivity", "specificity", "precision", "f1", "mcc"):
        row[f"test_{key}"] = test.get(key)
    return row


def write_aggregate_tables(output_dir, results):
    rows = [result_row(item) for item in results]
    pd.DataFrame(rows).to_csv(output_dir / "all_trials.csv", index=False)
    ranked = sorted(results, key=validation_rank, reverse=True)
    ranked_composite = sorted(results, key=validation_rank_composite, reverse=True)
    pd.DataFrame([result_row(item) for item in ranked]).to_csv(
        output_dir / "selected_by_validation.csv", index=False
    )
    pd.DataFrame([result_row(item) for item in ranked_composite]).to_csv(
        output_dir / "selected_by_validation_composite.csv", index=False
    )
    return ranked, ranked_composite


def train_trial(config, output_dir, args, modules, environment):
    train_module, dataloaders, models = modules
    trial_dir = output_dir / config["trial_id"]
    trial_dir.mkdir(parents=True, exist_ok=False)
    write_json(trial_dir / "trial_config.json", config)
    write_json(trial_dir / "environment.json", environment)
    (trial_dir / "run_command.txt").write_text(" ".join(shlex.quote(x) for x in sys.argv) + "\n", encoding="utf-8")
    write_json(trial_dir / "test_metrics.json", {
        "status": "not_evaluated", "reason": "Strict protocol: only validation-locked winner is tested."
    })
    seed_everything(args.seed)
    device = torch.device("cuda")
    data_args = (str(DATA_DIR) + "/", str(PROTEIN_PKL), str(PEPTIDE_PKL))
    train_set = dataloaders.PepMambaDataset("train", args.dataset, *data_args)
    val_set = dataloaders.PepMambaDataset("val", args.dataset, *data_args)
    train_loader = train_module.build_dataloader(train_set, config["batch_size"], True, 0, args.seed)
    val_loader = train_module.build_dataloader(val_set, config["batch_size"], False, 0, args.seed)
    model = models.PepMamba(
        n_layers=config["n_layers"], dropout=config["dropout"], **MODEL_FIXED
    ).to(device)
    class_weight, class_weight_info = estimate_weight(
        train_set, config["class_weight_power"], config["max_pos_weight"], device
    )
    criterion = train_module.HybridLoss(
        ce_weight=config["ce_weight"], class_weight=class_weight,
        focal_alpha=config["focal_alpha"], focal_gamma=config["focal_gamma"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=1, eta_min=1e-7
    )
    best = None; start = time.time()
    checkpoint = trial_dir / "best_validation_checkpoint.pth"
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0; batches = 0
        for peptide, protein, labels, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                peptide.to(device), protein.to(device), class_mask=mask.to(device)
            )
            loss = criterion(logits, labels.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); loss_sum += loss.item(); batches += 1
        scheduler.step(epoch)
        _, _, _, validation = evaluate_validation(model, val_loader, criterion, device, args)
        epoch_row = {"epoch": epoch, "train_loss": loss_sum / max(batches, 1),
                     "learning_rate": optimizer.param_groups[0]["lr"], **validation}
        append_jsonl(trial_dir / "train_log.jsonl", epoch_row)
        print(f"[{config['trial_id']}] " + json.dumps(epoch_row), flush=True)
        if getattr(args, "unified_selection", False):
            proposal = (validation["auc"], validation["auprc"])
            incumbent = None if best is None else (
                best["validation"]["auc"], best["validation"]["auprc"]
            )
        else:
            proposal = (validation["mcc"], validation["auprc"], validation["auc"], validation["composite"])
            incumbent = None if best is None else (
                best["validation"]["mcc"], best["validation"]["auprc"], best["validation"]["auc"], best["validation"]["composite"]
            )
        if incumbent is None or proposal > incumbent:
            best = {"epoch": epoch, "validation": validation}
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(state, checkpoint)
            write_json(trial_dir / "best_validation_epoch.json", best)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    val_labels, val_scores, threshold_scan, validation = evaluate_validation(
        model, val_loader, criterion, device, args
    )
    pd.DataFrame(threshold_scan).to_csv(trial_dir / "validation_threshold_scan.csv", index=False)
    save_predictions(trial_dir / "validation_predictions", val_labels, val_scores)
    write_json(trial_dir / "validation_metrics.json", validation)
    unified = getattr(args, "unified_selection", False)
    checkpoint_rule = "validation AUROC, then validation AUPRC" if unified else "validation MCC, then AUPRC, then AUROC, then composite"
    threshold_rule = "maximum validation MCC; lowest threshold on an exact tie" if unified else "validation MCC, then F1, then AUPRC"
    write_json(trial_dir / "selected_validation_threshold.json", {
        "threshold": validation["threshold"], "selection_metric": threshold_rule,
        "scan_file": str(trial_dir / "validation_threshold_scan.csv"),
    })
    result = {
        "trial_id": config["trial_id"], "seed": args.seed, "dataset": args.dataset,
        "single_model": True, "architecture": "unchanged project PepMamba/BioMamba",
        "config": config, "fixed_model_config": MODEL_FIXED,
        "class_weight": class_weight_info, "best_epoch": best["epoch"],
        "checkpoint_selection": checkpoint_rule,
        "threshold_selection": threshold_rule,
        "validation": validation, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "test": {"status": "not_evaluated"}, "elapsed_seconds": time.time() - start,
    }
    write_json(trial_dir / "trial_result.json", result)
    del model, optimizer, scheduler, criterion, train_loader, val_loader, train_set, val_set
    torch.cuda.empty_cache()
    return result


def evaluate_locked_test(winner, output_dir, args, modules):
    train_module, dataloaders, models = modules
    lock = {
        "locked_at": utc_now(), "selection_data": "pepnn_val.csv only",
        "selection_rule": winner["checkpoint_selection"] + "; " + winner["threshold_selection"],
        "trial_id": winner["trial_id"], "checkpoint": winner["checkpoint"],
        "checkpoint_sha256": winner["checkpoint_sha256"], "best_epoch": winner["best_epoch"],
        "validation": winner["validation"],
        "test_policy": "Construct test after lock; evaluate once at fixed validation threshold; no threshold scan.",
    }
    write_json(output_dir / "locked_validation_winner.json", lock)
    append_jsonl(output_dir / "protocol_events.jsonl", {"event": "validation_winner_locked", **lock})
    config = winner["config"]; seed_everything(args.seed); device = torch.device("cuda")
    data_args = (str(DATA_DIR) + "/", str(PROTEIN_PKL), str(PEPTIDE_PKL))
    append_jsonl(output_dir / "protocol_events.jsonl", {
        "event": "test_dataset_construction_started", "at": utc_now(), "trial_id": winner["trial_id"]
    })
    test_set = dataloaders.PepMambaDataset("test", args.dataset, *data_args)
    test_loader = train_module.build_dataloader(test_set, config["batch_size"], False, 0, args.seed)
    model = models.PepMamba(
        n_layers=config["n_layers"], dropout=config["dropout"], **MODEL_FIXED
    ).to(device)
    checkpoint = Path(winner["checkpoint"])
    if sha256(checkpoint) != winner["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after validation lock")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    # Reuse the training-derived class weight; never derive a parameter from test labels.
    class_weight = torch.tensor(
        [1.0, winner["class_weight"]["effective_pos_weight"]],
        dtype=torch.float32,
        device=device,
    )
    criterion = train_module.HybridLoss(
        ce_weight=config["ce_weight"], class_weight=class_weight,
        focal_alpha=config["focal_alpha"], focal_gamma=config["focal_gamma"],
    )
    labels, scores, loss = collect_scores(model, test_loader, criterion, device)
    threshold = float(winner["validation"]["threshold"])
    test = {
        "status": "evaluated_once_after_validation_lock", "loss": loss,
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "threshold": threshold, **confusion_metrics(labels, scores, threshold),
        "test_threshold_search_performed": False,
        "test_used_for_trial_selection": False,
    }
    trial_dir = output_dir / winner["trial_id"]
    save_predictions(trial_dir / "test_predictions", labels, scores)
    write_json(trial_dir / "test_metrics.json", test)
    winner["test"] = test
    write_json(trial_dir / "trial_result.json", winner)
    write_json(output_dir / "final_test_once.json", {
        **lock, "evaluated_at": utc_now(), "test": test,
        "n_test_residues": int(labels.size),
        "n_test_positive_residues": int(np.sum(labels == 1)),
        "n_test_negative_residues": int(np.sum(labels == 0)),
    })
    append_jsonl(output_dir / "protocol_events.jsonl", {
        "event": "test_evaluation_completed", "at": utc_now(), "trial_id": winner["trial_id"]
    })
    return winner


def presentation_summary(winner, output_dir):
    test = winner["test"]
    pepnn = COMPARATORS["PepNN-Seq"]; original = COMPARATORS["PepCA original"]
    tradeoff = []
    for key, label in (("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("precision", "Precision")):
        if test[key] < original[key]:
            tradeoff.append(f"{label} decreased from {original[key]:.3f} to {test[key]:.3f}")
    row = {
        "trial_id": winner["trial_id"], "selection_basis": "validation only",
        **{f"test_{key}": test[key] for key in ("sensitivity", "specificity", "precision", "f1", "mcc", "auc", "auprc", "threshold")},
        "test_auc_exceeds_pepnn_seq": test["auc"] > pepnn["auc"],
        "test_auc_exceeds_pepca_original": test["auc"] > original["auc"],
        "test_mcc_exceeds_pepnn_seq": test["mcc"] > pepnn["mcc"],
        "test_mcc_exceeds_pepca_original": test["mcc"] > original["mcc"],
        "sensitivity_ge_0_38": test["sensitivity"] >= .38,
        "specificity_ge_0_92": test["specificity"] >= .92,
        "precision_ge_0_40": test["precision"] >= .40,
        "suitable_for_main_table": test["auc"] > .817 and test["mcc"] > .351 and test["sensitivity"] >= .38 and test["specificity"] >= .92 and test["precision"] >= .40,
        "tradeoff": "; ".join(tradeoff) if tradeoff else "No requested operating-point metric decreased versus PepCA original.",
        "presentation_only_not_selection": True,
    }
    pd.DataFrame([row]).to_csv(output_dir / "paper_friendly_summary.csv", index=False)
    return row


def comparison_text(value, comparator, metric):
    reference = COMPARATORS[comparator].get(metric)
    if reference is None:
        return f"{comparator} did not report {metric}."
    relation = "exceeds" if value > reference else ("equals" if value == reference else "does not exceed")
    return f"{relation} {comparator} ({reference:.3f})"


def write_manuscript_outputs(winner, ranked, ranked_composite, presentation, output_dir):
    val = winner["validation"]; test = winner["test"]
    original = COMPARATORS["PepCA original"]; pepnn = COMPARATORS["PepNN-Seq"]
    relation = lambda value, baseline: "exceeds" if value > baseline else ("equals" if value == baseline else "does not exceed")
    markdown = f"""# Best PepNN candidate for manuscript

## Validation-based recommendation

Recommended trial: `{winner['trial_id']}`. It ranked first among {len(ranked)} trials by validation MCC, with validation AUPRC, AUROC, and the prespecified composite used as successive tie-breakers. Hyperparameters, checkpoint (epoch {winner['best_epoch']}), and validation-derived threshold ({val['threshold']:.3f}) were locked before the single test evaluation. Test metrics did not participate in selection.

Validation composite: `{val['composite']:.6f}` = 0.45 × MCC + 0.35 × AUPRC + 0.20 × AUROC.

## Fixed-threshold TS092 result

| Sensitivity | Specificity | Precision | F1 | MCC | AUROC | AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| {test['sensitivity']:.4f} | {test['specificity']:.4f} | {test['precision']:.4f} | {test['f1']:.4f} | {test['mcc']:.4f} | {test['auc']:.4f} | {test['auprc']:.4f} |

- AUROC {relation(test['auc'], pepnn['auc'])} PepNN-Seq (0.781) and {relation(test['auc'], original['auc'])} PepCA original (0.817).
- MCC {relation(test['mcc'], pepnn['mcc'])} PepNN-Seq (0.272) and {relation(test['mcc'], original['mcc'])} PepCA original (0.351).
- Operating-point trade-off: {presentation['tradeoff']}.

"""
    (output_dir / "best_candidate_for_manuscript.md").write_text(markdown, encoding="utf-8")
    comparison_rows = [{"model": name, **values} for name, values in COMPARATORS.items()]
    comparison_rows.append({"model": "PepMamba validation-selected", **{key: test[key] for key in ("sensitivity", "specificity", "precision", "mcc", "auc")}})
    with pd.ExcelWriter(output_dir / "best_candidate_for_manuscript.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([result_row(winner)]).to_excel(writer, sheet_name="Selected candidate", index=False)
        pd.DataFrame([result_row(item) for item in ranked]).to_excel(writer, sheet_name="Validation rank", index=False)
        pd.DataFrame([result_row(item) for item in ranked_composite]).to_excel(writer, sheet_name="Validation composite rank", index=False)
        pd.DataFrame(comparison_rows).to_excel(writer, sheet_name="Method comparison", index=False)
        pd.DataFrame([presentation]).to_excel(writer, sheet_name="Presentation only", index=False)
    rerun = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(ROOT))}
python {shlex.quote(str(SCRIPT_PATH))} --dataset pepnn --seed 42 --trials 1 --config-json {shlex.quote(str(output_dir / winner['trial_id'] / 'trial_config.json'))}
"""
    rerun_path = output_dir / "rerun_best_candidate.sh"
    rerun_path.write_text(rerun, encoding="utf-8"); rerun_path.chmod(0o755)


def consistency_report(results, winner, output_dir, args):
    lines = ["# Consistency check report", "", f"Generated: {utc_now()}", ""]
    failures = []; checkpoint_hashes = {}
    for result in results:
        trial = result["trial_id"]; validation = result["validation"]
        expected_f1 = 2 * validation["precision"] * validation["sensitivity"] / max(validation["precision"] + validation["sensitivity"], 1e-12)
        f1_ok = abs(expected_f1 - validation["f1"]) < 1e-10
        scan = pd.read_csv(output_dir / trial / "validation_threshold_scan.csv")
        threshold_ok = bool(np.isclose(scan["threshold"].to_numpy(), validation["threshold"], atol=1e-12).any())
        checkpoint_ok = Path(result["checkpoint"]).is_file() and sha256(result["checkpoint"]) == result["checkpoint_sha256"]
        prediction = np.load(output_dir / trial / "validation_predictions.npz")
        auc_ok = abs(roc_auc_score(prediction["labels"], prediction["scores"]) - validation["auc"]) < 1e-10
        auprc_ok = abs(average_precision_score(prediction["labels"], prediction["scores"]) - validation["auprc"]) < 1e-10
        checkpoint_hashes.setdefault(result["checkpoint_sha256"], []).append(trial)
        checks = {"F1 identity": f1_ok, "threshold appears in validation scan": threshold_ok,
                  "checkpoint exists and SHA256 matches": checkpoint_ok,
                  "AUROC is reproduced from probabilities": auc_ok,
                  "AUPRC is reproduced from probabilities": auprc_ok}
        lines += [f"## {trial}", ""] + [f"- [{'x' if ok else ' '}] {name}" for name, ok in checks.items()] + [""]
        failures.extend(f"{trial}: {name}" for name, ok in checks.items() if not ok)
    duplicates = {value: trials for value, trials in checkpoint_hashes.items() if len(trials) > 1}
    actual_tests = [item["trial_id"] for item in results if item.get("test", {}).get("status") == "evaluated_once_after_validation_lock"]
    nonwinner_status_ok = all(
        item["trial_id"] == winner["trial_id"] or item.get("test", {}).get("status") == "not_evaluated"
        for item in results
    )
    leakage_ok = actual_tests == [winner["trial_id"]] and nonwinner_status_ok
    test = winner["test"]
    test_expected_f1 = 2 * test["precision"] * test["sensitivity"] / max(test["precision"] + test["sensitivity"], 1e-12)
    test_f1_ok = abs(test_expected_f1 - test["f1"]) < 1e-10
    test_threshold_ok = abs(test["threshold"] - winner["validation"]["threshold"]) < 1e-12
    test_flags_ok = test.get("test_threshold_search_performed") is False and test.get("test_used_for_trial_selection") is False
    test_prediction = np.load(output_dir / winner["trial_id"] / "test_predictions.npz")
    test_auc_ok = abs(roc_auc_score(test_prediction["labels"], test_prediction["scores"]) - test["auc"]) < 1e-10
    test_auprc_ok = abs(average_precision_score(test_prediction["labels"], test_prediction["scores"]) - test["auprc"]) < 1e-10
    test_checks = {
        "test F1 identity": test_f1_ok,
        "test threshold equals locked validation threshold": test_threshold_ok,
        "test threshold-search/selection flags are false": test_flags_ok,
        "test AUROC reproduced from probabilities": test_auc_ok,
        "test AUPRC reproduced from probabilities": test_auprc_ok,
        "exactly one validation-locked trial was test-evaluated": leakage_ok,
    }
    lines += ["## Cross-trial and final-test checks", "",
              f"- Duplicate checkpoint hashes: `{json.dumps(duplicates)}`",
              f"- Test-evaluated trials: `{actual_tests}`"]
    lines += [f"- [{'x' if ok else ' '}] {name}" for name, ok in test_checks.items()]
    lines += ["",
              "- Test CSV schema/label integrity was audited before training; no test metric, prediction, or threshold was computed before the validation winner lock.",
              "- Primary validation ranking is MCC descending, then AUPRC, AUROC, and composite descending.",
              "- A separate composite-first ranking is generated for audit only.",
              "- AUROC and AUPRC are computed directly from probabilities and are threshold-independent.", ""]
    if duplicates:
        failures.append("duplicate checkpoint hashes detected")
    failures.extend(name for name, ok in test_checks.items() if not ok)
    lines += ["## Overall", "", "PASS" if not failures else "FAIL: " + "; ".join(failures), ""]
    (output_dir / "consistency_check_report.md").write_text("\n".join(lines), encoding="utf-8")
    if failures:
        raise RuntimeError("Consistency checks failed: " + "; ".join(failures))


def environment_info(runtime=None):
    info = {
        "created": utc_now(), "python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__, "pandas": pd.__version__, "git": git_info(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
    }
    if runtime:
        info["mamba_runtime"] = {key: value for key, value in runtime.items() if isinstance(value, str)}
    return info


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pepnn", choices=["pepnn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--threshold-min", type=float, default=.20)
    parser.add_argument("--threshold-max", type=float, default=.75)
    parser.add_argument("--threshold-step", type=float, default=.005)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-json", type=Path, help="Run one exact saved trial config")
    parser.add_argument("--dry-run", action="store_true", help="Audit and print configs; do not train or write a result directory")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed != 42:
        raise ValueError("This PepNN tuning run is restricted to seed=42")
    if args.epochs != 35:
        raise ValueError("This protocol requires exactly 35 epochs")
    if not (0 < args.threshold_min < args.threshold_max < 1) or args.threshold_step <= 0:
        raise ValueError("Invalid threshold grid")
    data_audit = audit_data(args.dataset)
    if args.config_json:
        config = json.loads(args.config_json.read_text(encoding="utf-8"))
        config["trial_id"] = "trial_000_rerun"
        configs = [config]
    else:
        configs = sample_configs(args.trials, args.seed)
    for config in configs:
        verify_config(config)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_only", "data_audit": data_audit,
                          "search_space": SEARCH_SPACE, "configs": configs}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from biomamba_runtime import (
        BIOMAMBA_FILE, LATEST_MAMBA3_ROOT, activate_latest_mamba3_with_biomamba,
    )
    runtime = activate_latest_mamba3_with_biomamba(LATEST_MAMBA3_ROOT, BIOMAMBA_FILE)
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import dataloaders
    import models
    import train as train_module
    output_dir = args.output_dir or RESULTS_ROOT / f"pepnn_paper_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    environment = environment_info(runtime)
    write_json(output_dir / "environment_info.json", environment)
    write_json(output_dir / "data_audit.json", data_audit)
    write_json(output_dir / "search_space.json", SEARCH_SPACE)
    protocol = {
        "created": utc_now(), "dataset": args.dataset, "seed": args.seed,
        "trials": len(configs), "epochs": args.epochs,
        "architecture": "unchanged project PepMamba; n_layers limited to 2 or 3",
        "training_data": str(DATA_DIR / "pepnn_train.csv"),
        "selection_data": str(DATA_DIR / "pepnn_val.csv"),
        "test_data": str(DATA_DIR / "pepnn_test.csv"),
        "checkpoint_and_trial_selection": "validation MCC, AUPRC, AUROC, then composite",
        "validation_composite": "0.45*MCC + 0.35*AUPRC + 0.20*AUROC",
        "threshold_selection": f"validation MCC, then F1, then AUPRC over [{args.threshold_min}, {args.threshold_max}] step {args.threshold_step}",
        "test_policy": "only locked validation winner evaluated once; no test threshold scan",
        "non_winner_test_metrics": "explicit not_evaluated records are saved to prevent test leakage",
        "paper_friendly_summary": "presentation only; never a selection criterion",
    }
    write_json(output_dir / "protocol.json", protocol)
    (output_dir / "run_command.txt").write_text(environment["command"] + "\n", encoding="utf-8")
    append_jsonl(output_dir / "protocol_events.jsonl", {"event": "run_started", "at": utc_now()})
    results = []
    for config in configs:
        result = train_trial(config, output_dir, args, (train_module, dataloaders, models), environment)
        results.append(result)
        write_aggregate_tables(output_dir, results)
        append_jsonl(output_dir / "protocol_events.jsonl", {
            "event": "validation_trial_completed", "at": utc_now(), "trial_id": result["trial_id"]
        })
    ranked, ranked_composite = write_aggregate_tables(output_dir, results)
    winner = evaluate_locked_test(ranked[0], output_dir, args, (train_module, dataloaders, models))
    for index, item in enumerate(results):
        if item["trial_id"] == winner["trial_id"]:
            results[index] = winner
    ranked, ranked_composite = write_aggregate_tables(output_dir, results)
    presentation = presentation_summary(winner, output_dir)
    consistency_report(results, winner, output_dir, args)
    write_manuscript_outputs(winner, ranked, ranked_composite, presentation, output_dir)
    append_jsonl(output_dir / "protocol_events.jsonl", {"event": "run_completed", "at": utc_now()})
    print(f"Completed strict PepNN paper tuning: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
