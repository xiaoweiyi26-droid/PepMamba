#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run PepMamba-BioMamba inside the Python/package environment that can load the
latest official Mamba3.

Purpose
-------
1. Load installed or development Mamba3 as the active mamba_ssm package.
2. Load the bundled BioMamba implementation from
   project-root mamba3_bio.py in that package.
3. Import the unchanged PepMamba project only after the injection.
4. Run BioMamba/Ours with the existing BITENET final training protocol.
5. Save results to a new directory and never overwrite completed trials.

This script does not modify models.py, modules.py, train.py, or either Mamba
source directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PEPMAMBA_DETERMINISTIC_KERNELS", "1")
os.environ.setdefault("MAMBA_DETERMINISTIC", "1")

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEVELOPMENT_MAMBA3_ROOT = PROJECT_ROOT.parent / "mamba-main1" / "mamba-main"
_ENV_MAMBA3_ROOT = os.environ.get("PEPMAMBA_MAMBA_ROOT")
LATEST_MAMBA3_ROOT = (
    Path(_ENV_MAMBA3_ROOT).expanduser()
    if _ENV_MAMBA3_ROOT
    else (_DEVELOPMENT_MAMBA3_ROOT if _DEVELOPMENT_MAMBA3_ROOT.is_dir() else None)
)
BIOMAMBA_FILE = PROJECT_ROOT / "mamba3_bio.py"
EXPECTED_BIOMAMBA_SHA256 = "82f431945d030fd40b2f2aa39aeec965afe2d61f7823591763b69ac6514a870b"

VARIANT = {
    "name": "pep_mamba_biomamba_latest_mamba3_env",
    "label": "PepMamba-BioMamba / Ours (latest Mamba3 env)",
    "log_label": "BITENET",
    "protein_esm": True,
    "peptide_esm": True,
}

MODEL_FLOW = (
    "protein/peptide ESM-2 -> projection -> ConvPositionEncoding -> "
    "LocalPatternBlock -> bidirectional BioMamba -> protein-peptide "
    "cross-attention -> SequenceFFN -> prediction head"
)

HEAD_INPUT = (
    "concat(protein_skip/protein_prior, "
    "protein_hidden/interaction_refined); no peptide_global"
)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def purge_mamba_modules() -> None:
    """Remove any previously imported mamba_ssm package from this process."""
    for name in list(sys.modules):
        if name == "mamba3_bio" or name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]


def activate_latest_mamba3_with_biomamba(
    latest_root: Path | None,
    biomamba_file: Path,
) -> dict[str, Any]:
    """Load installed/development Mamba3 and the project-local BioMamba module."""
    require_path(biomamba_file, "Project-local BioMamba implementation")
    digest = hashlib.sha256(biomamba_file.read_bytes()).hexdigest()
    if digest != EXPECTED_BIOMAMBA_SHA256:
        raise RuntimeError(
            f"BioMamba source SHA256 mismatch: expected {EXPECTED_BIOMAMBA_SHA256}, got {digest}"
        )
    purge_mamba_modules()
    importlib.invalidate_caches()

    if latest_root is not None:
        latest_root = latest_root.resolve()
        require_path(latest_root / "mamba_ssm", "Development Mamba3 package")
        latest_root_str = str(latest_root)
        sys.path = [
            entry for entry in sys.path
            if not entry or Path(entry).resolve() != latest_root
        ]
        sys.path.insert(0, latest_root_str)

    latest_pkg = importlib.import_module("mamba_ssm")
    latest_modules_pkg = importlib.import_module("mamba_ssm.modules")
    try:
        latest_mamba3_cls = getattr(latest_pkg, "Mamba3")
    except AttributeError as exc:
        raise ImportError(
            "The installed mamba_ssm package does not provide Mamba3. "
            "Install the Mamba3-compatible version documented in README.md."
        ) from exc

    module_name = "mamba_ssm.modules.mamba3_bio"
    spec = importlib.util.spec_from_file_location(module_name, biomamba_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load BioMamba implementation: {biomamba_file}")
    biomamba_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = biomamba_module
    sys.modules["mamba3_bio"] = biomamba_module
    setattr(latest_modules_pkg, "mamba3_bio", biomamba_module)
    try:
        spec.loader.exec_module(biomamba_module)
    except Exception:
        sys.modules.pop(module_name, None)
        sys.modules.pop("mamba3_bio", None)
        raise

    biomamba_cls = getattr(biomamba_module, "BioMamba3Block")
    package_file = Path(inspect.getfile(latest_pkg)).resolve()
    return {
        "mamba_ssm_file": str(package_file),
        "mamba_ssm_version": str(getattr(latest_pkg, "__version__", "unknown")),
        "mamba_root": str(package_file.parent.parent),
        "latest_mamba3_class_file": str(Path(inspect.getfile(latest_mamba3_cls)).resolve()),
        "biomamba_class_file": str(Path(inspect.getfile(biomamba_cls)).resolve()),
        "biomamba_class": biomamba_cls,
    }


def forward_check(
    biomamba_cls: type,
    d_model: int,
    d_state: int,
    dropout: float,
    device: torch.device,
) -> dict[str, Any]:
    """Minimal forward/backward check before launching a full experiment."""
    block = biomamba_cls(
        d_model=d_model,
        d_state=d_state,
        dropout=dropout,
    ).to(device)

    block.train()
    x = torch.randn(
        2,
        17,
        d_model,
        device=device,
        requires_grad=True,
    )
    y = block(x)

    if isinstance(y, tuple):
        y = y[0]
    if y.shape != x.shape:
        raise RuntimeError(
            f"BioMamba forward shape mismatch: input={tuple(x.shape)}, "
            f"output={tuple(y.shape)}"
        )
    if not torch.isfinite(y).all():
        raise RuntimeError("BioMamba forward produced NaN or Inf.")

    loss = y.float().square().mean()
    loss.backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("BioMamba backward check failed.")

    result = {
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "forward_finite": True,
        "backward_finite": True,
        "device": str(device),
    }

    del block, x, y, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def trial_dir(run_dir: Path, seed: int) -> Path:
    return run_dir / f"seed_{seed}" / VARIANT["name"]


def is_complete(path: Path) -> bool:
    return (
        (path / "result.json").is_file()
        and (path / "test_metrics.json").is_file()
    )


def archive_incomplete(path: Path) -> None:
    if not path.exists() or is_complete(path):
        return
    archived = path.with_name(
        f"{path.name}_incomplete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    path.rename(archived)
    print(f"Archived incomplete trial: {archived}", flush=True)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    validation = result["validation"]
    test = result["test"]
    return {
        "variant": result.get("variant", VARIANT["label"]),
        "seed": result["seed"],
        "best_epoch": result["best_epoch"],
        "selected_threshold": validation["threshold"],
        "val_auc": validation["auc"],
        "val_auprc": validation["auprc"],
        "val_mcc": validation["mcc"],
        "test_auc": test["auc"],
        "test_auprc": test["auprc"],
        "sensitivity": test["sensitivity"],
        "specificity": test["specificity"],
        "precision": test["precision"],
        "mcc": test["mcc"],
        "f1": test["f1"],
        "elapsed_sec": result.get("elapsed_sec"),
    }


def aggregate(run_dir: Path, seeds: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = trial_dir(run_dir, seed)
        if not is_complete(path):
            continue
        with (path / "result.json").open("r", encoding="utf-8") as handle:
            rows.append(flatten_result(json.load(handle)))

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    df.to_csv(run_dir / "biomamba_latest_mamba3_env_all_results.csv", index=False)

    with pd.ExcelWriter(
        run_dir / "biomamba_latest_mamba3_env_all_results.xlsx"
    ) as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)

    metric_cols = [
        "test_auc",
        "test_auprc",
        "mcc",
        "f1",
        "sensitivity",
        "specificity",
        "precision",
    ]
    summary_rows = []
    for metric in metric_cols:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        run_dir / "biomamba_latest_mamba3_env_mean_std.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Seeds to run. Default: 42",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing/new result directory. Default: create timestamped directory.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--latest-mamba3-root",
        type=Path,
        default=LATEST_MAMBA3_ROOT,
    )
    parser.add_argument(
        "--biomamba-file",
        type=Path,
        default=BIOMAMBA_FILE,
    )
    parser.add_argument(
        "--skip-forward-check",
        action="store_true",
        help="Skip the minimal BioMamba CUDA forward/backward check.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only verify imports and forward compatibility; do not train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    latest_root = args.latest_mamba3_root.resolve() if args.latest_mamba3_root else None
    biomamba_file = args.biomamba_file.resolve()

    require_path(project_root, "PepMamba project root")
    require_path(
        project_root / "scripts" / "final" / "bitenet_training_core.py",
        "Existing BITENET training runner",
    )

    # Activate latest mamba_ssm and inject BioMamba BEFORE importing PepMamba code.
    env_info = activate_latest_mamba3_with_biomamba(
        latest_root=latest_root,
        biomamba_file=biomamba_file,
    )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    # These imports must happen only after BioMamba injection.
    base = importlib.import_module("bitenet_training_core")
    project_modules = importlib.import_module("modules")

    imported_biomamba_file = Path(
        inspect.getfile(project_modules.BioMamba2Block)
    ).resolve()
    if imported_biomamba_file != biomamba_file:
        raise RuntimeError(
            "PepMamba modules.py did not bind to the requested BioMamba file.\n"
            f"Expected: {biomamba_file}\n"
            f"Actual:   {imported_biomamba_file}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is unavailable. This experiment was designed for the "
            "same GPU training environment used by the prior Mamba3 run."
        )

    forward_audit = None
    if not args.skip_forward_check:
        forward_audit = forward_check(
            biomamba_cls=env_info["biomamba_class"],
            d_model=int(base.BASE_MODEL_CONFIG["d_model"]),
            d_state=128,
            dropout=float(base.FINAL_PARAMS["dropout"]),
            device=device,
        )

    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            project_root
            / "PAPER_MAIN_RESULTS_FINAL"
            / "new_runs"
            / f"bitenet_biomamba_latest_mamba3_env_{stamp}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = args.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

    audit = {
        "created": datetime.now(timezone.utc).isoformat(),
        "purpose": __doc__,
        "project_root": str(project_root),
        "latest_mamba3_root": str(latest_root) if latest_root else env_info["mamba_root"],
        "biomamba_file": str(biomamba_file),
        "active_mamba_ssm_file": env_info["mamba_ssm_file"],
        "latest_mamba3_class_file": env_info["latest_mamba3_class_file"],
        "active_biomamba_class_file": env_info["biomamba_class_file"],
        "project_modules_biomamba_file": str(imported_biomamba_file),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "seeds": args.seeds,
        "final_params": base.FINAL_PARAMS,
        "model_config": base.BASE_MODEL_CONFIG,
        "model_flow": MODEL_FLOW,
        "head_input": HEAD_INPUT,
        "forward_check": forward_audit,
    }
    (run_dir / "latest_mamba3_env_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88, flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Active mamba_ssm: {env_info['mamba_ssm_file']}", flush=True)
    print(f"Official Mamba3: {env_info['latest_mamba3_class_file']}", flush=True)
    print(f"BioMamba source: {env_info['biomamba_class_file']}", flush=True)
    print(f"Device: {device} / {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 88, flush=True)

    if args.audit_only:
        print("Audit passed. Training was not started.", flush=True)
        return

    original_atomic_json = base.atomic_json

    def audited_atomic_json(path: Path, payload: Any) -> None:
        if isinstance(payload, dict) and Path(path).name in {
            "params.json",
            "result.json",
        }:
            payload = dict(payload)
            payload["runtime_environment"] = {
                "active_mamba_ssm_file": env_info["mamba_ssm_file"],
                "latest_mamba3_class_file": env_info[
                    "latest_mamba3_class_file"
                ],
                "biomamba_class_file": env_info["biomamba_class_file"],
            }
            payload["model_flow"] = MODEL_FLOW
            payload["head_input"] = HEAD_INPUT
            payload["comparison_note"] = (
                "BioMamba implementation is unchanged, but it is imported "
                "inside the latest official Mamba3 mamba_ssm package environment."
            )
        original_atomic_json(path, payload)

    base.atomic_json = audited_atomic_json

    try:
        for seed in args.seeds:
            base.SEED = int(seed)
            seed_dir = run_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            path = trial_dir(run_dir, seed)

            if is_complete(path):
                print(f"[BioMamba] seed={seed} complete, skipping", flush=True)
                continue

            archive_incomplete(path)
            print(f"[BioMamba] starting seed={seed}", flush=True)
            base.run_variant(seed_dir, VARIANT, device)
            aggregate(run_dir, args.seeds)
    finally:
        base.atomic_json = original_atomic_json

    aggregate(run_dir, args.seeds)
    print(f"Finished. Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
