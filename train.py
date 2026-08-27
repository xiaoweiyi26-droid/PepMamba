#!/usr/bin/env python3
"""Train PepMamba and select the checkpoint/threshold using validation only."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("PEPMAMBA_DETERMINISTIC_KERNELS", "1")
os.environ.setdefault("MAMBA_DETERMINISTIC", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import BEST_PARAMS

ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "scripts" / "final"
DATASETS = ("bitenet", "interpep", "pepbind", "pepnn")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, batch_size, shuffle, num_workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=__import__("dataloaders").collate_fn,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


class FocalDiceLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, alpha=0.7, smooth=1e-5, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        valid = targets.ne(self.ignore_index)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        logits = logits[valid]
        targets = targets[valid]
        probs = F.softmax(logits, dim=-1)[:, 1]
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        focal = (((1.0 - torch.exp(-ce)) ** self.gamma) * ce).mean()
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        return self.alpha * focal + (1.0 - self.alpha) * dice


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        valid = targets.ne(self.ignore_index)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return F.cross_entropy(logits[valid], targets[valid], weight=self.weight)


class HybridLoss(nn.Module):
    def __init__(self, ce_weight=0.5, class_weight=None, focal_alpha=0.7, focal_gamma=2.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.ce = WeightedCrossEntropyLoss(weight=class_weight)
        self.fd = FocalDiceLoss(weight=class_weight, alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + (1.0 - self.ce_weight) * self.fd(logits, targets)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def activate_runtime():
    if str(FINAL_DIR) not in sys.path:
        sys.path.insert(0, str(FINAL_DIR))
    from biomamba_runtime import BIOMAMBA_FILE, LATEST_MAMBA3_ROOT, activate_latest_mamba3_with_biomamba
    return activate_latest_mamba3_with_biomamba(LATEST_MAMBA3_ROOT, BIOMAMBA_FILE)


def fixed_config(dataset: str) -> dict:
    params = BEST_PARAMS[dataset]
    return {
        "trial_id": "seed_42",
        "lr": params["lr"],
        "dropout": params["dropout"],
        "focal_alpha": params["alpha"],
        "focal_gamma": params["gamma"],
        "ce_weight": params["ce_weight"],
        "weight_decay": params["weight_decay"],
        "batch_size": params["batch_size"],
        "class_weight_power": params["class_weight_power"],
        "max_pos_weight": params["max_pos_weight"],
        "n_layers": params["n_layers"],
    }


def default_output(dataset: str, seed: int) -> Path:
    return ROOT / "runs" / dataset / f"seed_{seed}"


def validate_inputs(dataset: str) -> None:
    for split in ("train", "val"):
        path = ROOT / "data" / f"{dataset}_{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (ROOT / "protein_esm2.pkl", ROOT / "peptide_esm2.pkl"):
        if not path.is_file():
            raise FileNotFoundError(path)


def train_regular(dataset: str, seed: int, epochs: int, output_dir: Path, unified_selection: bool = False) -> None:
    runtime = activate_runtime()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import dataloaders
    import models

    runner = importlib.import_module(f"run_{dataset}")
    config = fixed_config(dataset)
    runner.verify_config(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(
        dataset=dataset,
        seed=seed,
        epochs=epochs,
        threshold_min=0.20,
        threshold_max=0.75,
        threshold_step=0.005,
        fixed_validation_threshold=(
            float(BEST_PARAMS[dataset]["threshold"])
            if dataset == "interpep" and not unified_selection else None
        ),
        unified_selection=unified_selection,
    )
    environment = runner.environment_info(runtime)
    runner.write_json(output_dir / "environment_info.json", environment)
    runner.write_json(output_dir / "fixed_config.json", config)
    runner.write_json(output_dir / "protocol.json", {
        "dataset": dataset,
        "seed": seed,
        "training_data": f"{dataset}_train.csv",
        "selection_data": f"{dataset}_val.csv",
        "test_data_accessed_during_training": False,
        "checkpoint_and_threshold_selection": (
            "checkpoint: maximum validation AUROC, tie validation AUPRC; threshold: maximum validation MCC"
            if unified_selection else ("checkpoint selected on validation; threshold fixed to the validation-constrained operating point in config.py" if dataset == "interpep" else "validation only")
        ),
        "test_policy": "run test.py after training; fixed validation threshold; no test threshold search",
    })
    result = runner.train_trial(config, output_dir, args, (sys.modules[__name__], dataloaders, models), environment)
    lock = {
        "dataset": dataset,
        "seed": seed,
        "trial_id": result["trial_id"],
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "best_epoch": result["best_epoch"],
        "validation": result["validation"],
        "selection_data": f"{dataset}_val.csv only",
        "test_status": "not_evaluated",
    }
    runner.write_json(output_dir / "training_result.json", result)
    runner.write_json(output_dir / "locked_validation_winner.json", lock)


def train_bitenet(seed: int, output_dir: Path) -> None:
    runtime = activate_runtime()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    base = importlib.import_module("bitenet_training_core")
    runner = importlib.import_module("run_bitenet")
    output_dir.mkdir(parents=True, exist_ok=False)
    base.SEED = seed
    device = torch.device("cuda")
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    result = base.run_variant(seed_dir, runner.VARIANT, device, evaluate_test=False)
    variant_dir = runner.trial_dir(output_dir, seed)
    checkpoint = variant_dir / "best_checkpoint.pth"
    result.update({
        "dataset": "bitenet",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "runtime_environment": {key: value for key, value in runtime.items() if isinstance(value, (str, int, float, bool)) or value is None},
    })
    base.atomic_json(variant_dir / "result.json", result)
    base.atomic_json(output_dir / "training_result.json", result)
    base.atomic_json(output_dir / "locked_validation_winner.json", {
        "dataset": "bitenet",
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "best_epoch": result["best_epoch"],
        "validation": result["validation"],
        "selection_data": "bitenet_val.csv only",
        "test_status": "not_evaluated",
    })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--unified-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Final PepMamba reproduction is fixed to seed=42")
    if args.epochs != 35:
        raise ValueError("Final PepMamba reproduction requires 35 epochs")
    validate_inputs(args.dataset)
    output_dir = (args.output_dir or default_output(args.dataset, args.seed)).resolve()
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "output_dir": str(output_dir),
        "parameters": BEST_PARAMS[args.dataset],
        "training_reads": [f"{args.dataset}_train.csv", f"{args.dataset}_val.csv"],
        "test_data_accessed": False,
        "unified_selection": args.unified_selection,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if output_dir.exists():
        if args.output_dir is not None:
            raise FileExistsError(
                f"Explicit --output-dir already exists: {output_dir}. "
                "Choose a new directory."
            )
        expected_default = default_output(args.dataset, args.seed).resolve()
        if output_dir != expected_default or ROOT.resolve() not in output_dir.parents:
            raise RuntimeError(f"Refusing unsafe overwrite target: {output_dir}")
        print(f"Overwriting previous default run: {output_dir}", flush=True)
        shutil.rmtree(output_dir)
    seed_everything(args.seed)
    if args.dataset == "bitenet":
        train_bitenet(args.seed, output_dir)
    else:
        train_regular(args.dataset, args.seed, args.epochs, output_dir, args.unified_selection)
    print(f"Training complete: {output_dir}")
    print(f"Next: python test.py --dataset {args.dataset} --seed {args.seed}")


if __name__ == "__main__":
    main()
