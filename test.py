#!/usr/bin/env python3
"""Evaluate a trained PepMamba checkpoint at its locked validation threshold."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "scripts" / "final"
DATASETS = ("bitenet", "interpep", "pepbind", "pepnn")


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


def default_run(dataset: str, seed: int) -> Path:
    return ROOT / "runs" / dataset / f"seed_{seed}"


def test_regular(dataset: str, seed: int, run_dir: Path) -> dict:
    activate_runtime()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import dataloaders
    import models
    import train as train_module

    runner = importlib.import_module(f"run_{dataset}")
    result = json.loads((run_dir / "training_result.json").read_text(encoding="utf-8"))
    if result["dataset"] != dataset or int(result["seed"]) != seed:
        raise RuntimeError("Training result dataset/seed does not match this test command")
    args = SimpleNamespace(dataset=dataset, seed=seed)
    result = runner.evaluate_locked_test(result, run_dir, args, (train_module, dataloaders, models))
    runner.write_json(run_dir / "test_result.json", result)
    return result["test"]


def test_bitenet(seed: int, run_dir: Path) -> dict:
    activate_runtime()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    base = importlib.import_module("bitenet_training_core")
    runner = importlib.import_module("run_bitenet")
    result_path = run_dir / "training_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = Path(result["checkpoint"])
    if sha256(checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA256 differs from the validation-locked record")
    base.SEED = seed
    base.seed_everything(seed)
    device = torch.device("cuda")
    dataset = base.RepresentationAblationDataset("test", True, True)
    loader = base.build_loader(dataset, result["params"]["batch_size"], False)
    model = base.AblationModel(True, True, result["model_config"], result["params"]["dropout"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    labels, scores, _ = base.collect_scores(model, loader, device)
    threshold = float(result["validation"]["threshold"])
    metrics = base.metrics_at_threshold(labels, scores, threshold)
    test = {
        "status": "evaluated_once_at_validation_threshold",
        "loss": None,
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "threshold": threshold,
        **metrics,
        "test_threshold_search_performed": False,
        "test_used_for_selection": False,
    }
    variant_dir = runner.trial_dir(run_dir, seed)
    base.save_predictions(variant_dir / "test_predictions", labels, scores)
    base.atomic_json(variant_dir / "test_metrics.json", test)
    result["test"] = test
    base.atomic_json(variant_dir / "result.json", result)
    base.atomic_json(run_dir / "test_result.json", result)
    base.atomic_json(run_dir / "final_test_once.json", {
        "dataset": "bitenet",
        "seed": seed,
        "validation_threshold": threshold,
        "test_threshold_search_performed": False,
        "test_used_for_selection": False,
        "test": test,
    })
    return test


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("Final PepMamba reproduction is fixed to seed=42")
    run_dir = (args.run_dir or default_run(args.dataset, args.seed)).resolve()
    if args.dry_run:
        print(json.dumps({
            "dataset": args.dataset,
            "seed": args.seed,
            "run_dir": str(run_dir),
            "required_training_record": str(run_dir / "locked_validation_winner.json"),
            "threshold_source": "validation only",
            "test_threshold_search_performed": False,
        }, indent=2, ensure_ascii=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for required in (run_dir / "training_result.json", run_dir / "locked_validation_winner.json"):
        if not required.is_file():
            raise FileNotFoundError(f"Run train.py first; missing: {required}")
    if (run_dir / "final_test_once.json").exists():
        raise FileExistsError(f"Test was already evaluated for this run: {run_dir}")
    test = test_bitenet(args.seed, run_dir) if args.dataset == "bitenet" else test_regular(args.dataset, args.seed, run_dir)
    print(json.dumps({"dataset": args.dataset, "seed": args.seed, **test}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
