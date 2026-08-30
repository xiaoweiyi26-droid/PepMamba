# -*- coding: utf-8 -*-
import os
import pickle

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class PepMambaDataset(Dataset):
    """
    Returns:
        pep_feat : [L_pep, 1280]
        prot_feat: [L_prot, 1280]
        target   : [L_prot]
    """

    def __init__(self, mode, data_set, df_dir, prot_pkl_path, pep_pkl_path, esm_feat_dim=1280):
        self.phase = mode
        self.esm_feat_dim = esm_feat_dim

        csv_name = f"{data_set}_{mode}.csv"
        self.df = pd.read_csv(
            os.path.join(df_dir, csv_name),
            dtype={"Pep_seq": str, "Prot_seq": str, "label": str},
            keep_default_na=False,
        )
        required_columns = {"Pep_seq", "Prot_seq", "label"}
        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing required CSV columns: {sorted(missing_columns)}")

        with open(prot_pkl_path, "rb") as f:
            self.prot_feat_dict = pickle.load(f)

        with open(pep_pkl_path, "rb") as f:
            self.pep_feat_dict = pickle.load(f)

    @staticmethod
    def _parse_binary_label_string(target_str):
        target = str(target_str)
        if not target or any(c not in {"0", "1"} for c in target):
            raise ValueError("label must be a non-empty contiguous binary string")
        return torch.tensor([int(c) for c in target], dtype=torch.long)

    def _load_feature(self, seq, feat_dict):
        if seq not in feat_dict:
            raise KeyError(f"Sequence is missing from the ESM feature dictionary: {seq}")
        return torch.from_numpy(feat_dict[seq]).float()

    def __getitem__(self, index):
        row = self.df.iloc[index]

        pep_seq = str(row["Pep_seq"]).strip()
        prot_seq = str(row["Prot_seq"]).strip()
        target = self._parse_binary_label_string(row["label"])

        pep_feat = self._load_feature(pep_seq, self.pep_feat_dict)
        prot_feat = self._load_feature(prot_seq, self.prot_feat_dict)

        if pep_feat.ndim != 2 or pep_feat.shape[1] != self.esm_feat_dim:
            raise ValueError(f"Unexpected peptide feature shape: {tuple(pep_feat.shape)}")
        if prot_feat.ndim != 2 or prot_feat.shape[1] != self.esm_feat_dim:
            raise ValueError(f"Unexpected protein feature shape: {tuple(prot_feat.shape)}")

        if not (len(target) == len(prot_seq) == prot_feat.shape[0]):
            raise ValueError(
                "Protein lengths disagree: "
                f"label={len(target)}, sequence={len(prot_seq)}, feature={prot_feat.shape[0]}"
            )
        if len(pep_seq) != pep_feat.shape[0]:
            raise ValueError(
                "Peptide lengths disagree: "
                f"sequence={len(pep_seq)}, feature={pep_feat.shape[0]}"
            )

        if self.phase == "train":
            if torch.rand(1).item() < 0.5:
                prot_feat = prot_feat + torch.randn_like(prot_feat) * 0.02

            if torch.rand(1).item() < 0.3:
                channel_mask = (torch.rand(prot_feat.shape[-1]) > 0.1).float()
                prot_feat = prot_feat * channel_mask.unsqueeze(0)

        return pep_feat, prot_feat, target

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    pep, prot, label = zip(*batch)

    pep_pad = pad_sequence(pep, batch_first=True, padding_value=0.0)
    prot_pad = pad_sequence(prot, batch_first=True, padding_value=0.0)
    label_pad = pad_sequence(label, batch_first=True, padding_value=-100)

    # True means padding / invalid protein positions.
    mask = label_pad.eq(-100)

    return pep_pad, prot_pad, label_pad, mask
