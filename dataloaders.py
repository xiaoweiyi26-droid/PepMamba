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
        self.df = pd.read_csv(os.path.join(df_dir, csv_name))

        with open(prot_pkl_path, "rb") as f:
            self.prot_feat_dict = pickle.load(f)

        with open(pep_pkl_path, "rb") as f:
            self.pep_feat_dict = pickle.load(f)

    @staticmethod
    def _parse_binary_label_string(target_str):
        """
        More robust than raw character-by-character parsing.
        Supports compact strings like '001010' and strings containing separators.
        """
        filtered = [c for c in str(target_str).strip() if c in {"0", "1"}]
        return torch.tensor([int(c) for c in filtered], dtype=torch.long)

    def _load_feature(self, seq, feat_dict):
        if seq in feat_dict:
            feat = torch.from_numpy(feat_dict[seq]).float()
        else:
            feat = torch.zeros((len(seq), self.esm_feat_dim), dtype=torch.float32)
        return feat

    def __getitem__(self, index):
        row = self.df.iloc[index]

        pep_seq = str(row["Pep_seq"]).strip()
        prot_seq = str(row["Prot_seq"]).strip()
        target = self._parse_binary_label_string(row["label"])

        pep_feat = self._load_feature(pep_seq, self.pep_feat_dict)
        prot_feat = self._load_feature(prot_seq, self.prot_feat_dict)

        if pep_feat.ndim != 2 or pep_feat.shape[-1] != self.esm_feat_dim:
            raise ValueError(f"Unexpected peptide feature shape: {tuple(pep_feat.shape)}")
        if prot_feat.ndim != 2 or prot_feat.shape[-1] != self.esm_feat_dim:
            raise ValueError(f"Unexpected protein feature shape: {tuple(prot_feat.shape)}")

        min_len = min(prot_feat.shape[0], target.shape[0])
        prot_feat = prot_feat[:min_len]
        target = target[:min_len]

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
