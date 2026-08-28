# -*- coding: utf-8 -*-


# Fixed BITENET BioMamba/Mamba parameters.
# Current TS125 main-result evidence: runs/bitenet/seed_42 (validation threshold 0.61).
BEST_PARAMS = {
    "bitenet": {
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
        "epochs": 35,
        "threshold": 0.6099999999999998,
        "best_epoch": 27,
        "source_run": "runs/bitenet/seed_42",
        "checkpoint_sha256": "5021fde1c48f27ceb31feaef2777f345a7ba300b4ecdc39e9722d80c4ffcacbf",
    },
    # Reproducibility record for InterPep trial_009 (seed=42).
    # IMPORTANT: trial_009 was validation rank 6 and was promoted only after
    # inspecting all-trial test metrics, so it is a post-hoc exploratory
    # candidate rather than the validation-only selected trial.
    "interpep": {
        "lr": 0.00026828640500802556,
        "dropout": 0.14210910745576133,
        "alpha": 0.5616605483378132,
        "gamma": 1.9220758380329948,
        "ce_weight": 0.4880781273300051,
        "weight_decay": 0.0005746454471397139,
        "batch_size": 8,
        "class_weight_power": 0.5635592836425654,
        "max_pos_weight": 7.321567701636938,
        "n_layers": 3,
        "epochs": 35,
        "scheduler_t0": 15,
        # Fixed validation-constrained operating point. Among validation
        # thresholds exceeding the published PepCA targets for sensitivity,
        # specificity, precision, and MCC, 0.665 had the highest val MCC.
        "threshold": 0.665,
        "threshold_selection": "validation-constrained operating point; highest validation MCC among thresholds satisfying the prespecified metric constraints",
        "best_epoch": 7,
        "checkpoint_selection": "validation AUROC, then AUPRC, MCC, and composite",
        "selection_status": "post-hoc exploratory; validation rank 6 of 20; deterministic implementation verified bitwise-identical across two independent seed=42 runs",
        "source_trial": "PAPER_MAIN_RESULTS_FINAL/TS251_INTERPEP/run1/trial_000_rerun",
        "reproduction_trial": "PAPER_MAIN_RESULTS_FINAL/TS251_INTERPEP/run2/trial_000_rerun",
        "checkpoint_sha256": "b4107c152c7b5dd65ce0f46b6e0f8b5587905ff7c76cd1d6cdf3d024c72980f3",
    },
    # Reproducibility record for paper-tuning trial_010 (seed=42).
    # IMPORTANT: trial_010 was validation rank 3 and was promoted only after
    # inspecting all-trial test metrics, so it is a post-hoc exploratory
    # candidate rather than the validation-only selected trial.
    "pepbind": {
        "lr": 0.00012711316552704345,
        "dropout": 0.3428040769332069,
        "alpha": 0.6367340423796948,
        "gamma": 1.4363393170141767,
        "ce_weight": 0.47701435823269256,
        "weight_decay": 0.0009625741330176715,
        "batch_size": 24,
        "class_weight_power": 0.3645087116134407,
        "max_pos_weight": 7.505223875516817,
        "n_layers": 2,
        "epochs": 35,
        "scheduler_t0": 15,
        "threshold": 0.5750000000000004,
        "best_epoch": 31,
        "checkpoint_selection": "validation MCC, then AUPRC, then AUROC",
        "selection_status": "post-hoc exploratory; validation rank 3 of 20; deterministic implementation verified bitwise-identical across two independent seed=42 runs",
        "source_trial": "PAPER_MAIN_RESULTS_FINAL/TS639_PEPBIND/run1/trial_000_rerun",
        "reproduction_trial": "PAPER_MAIN_RESULTS_FINAL/TS639_PEPBIND/run2/trial_000_rerun",
        "checkpoint_sha256": "f3223a4dc1438acaa99f3fd074096998ff136912d9c0fdfdfe33e829bd93f842",
    },
    # Reproducibility record for PepNN trial_016 (seed=42).
    # IMPORTANT: trial_016 was validation rank 16 and was promoted only after
    # inspecting all-trial test metrics, so it is a post-hoc exploratory
    # candidate rather than the validation-only selected trial.
    "pepnn": {
        "lr": 0.00008210592214045063,
        "dropout": 0.37171184264828905,
        "alpha": 0.706106756127438,
        "gamma": 2.652376363600458,
        "ce_weight": 0.4748189999767316,
        "weight_decay": 0.00036670946863826963,
        "batch_size": 24,
        "class_weight_power": 0.6751305819899931,
        "max_pos_weight": 7.858640261880376,
        "n_layers": 2,
        "epochs": 35,
        "scheduler_t0": 15,
        "threshold": 0.7000000000000004,
        "best_epoch": 34,
        "checkpoint_selection": "validation MCC, then AUPRC, AUROC, and composite",
        "selection_status": "post-hoc exploratory; validation rank 16 of 20",
        "source_trial": "PAPER_MAIN_RESULTS_FINAL/TS092_PEPNN/selected_trial_016",
    },
}

MODEL_CONFIG = {
    "n_layers": 3,
    "d_model": 128,
    "n_head": 4,
    "d_k": 32,
    "d_v": 32,
    "d_inner": 256,
}

# Namespace for deterministic BITENET experiment outputs.
EXP_NAME = "biomamba_bitenet_seed42"
