#!/usr/bin/env python3
"""Strict, no-clipping UMPPI protein-residue dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class UMppiProteinSiteDataset(Dataset):
    def __init__(self, csv_path, project_root, embedding_dir):
        self.csv_path=Path(csv_path); self.root=Path(project_root); self.embedding_dir=Path(embedding_dir)
        self.frame=pd.read_csv(self.csv_path,dtype=str,keep_default_na=False)
        manifest=pd.read_csv(self.embedding_dir/'manifest.csv',dtype=str,keep_default_na=False)
        if manifest.sequence_hash.duplicated().any(): raise ValueError('duplicate manifest sequence hash')
        self.manifest=manifest.set_index('sequence_hash').to_dict('index')

    def __len__(self): return len(self.frame)

    def _embedding(self, sequence_hash, expected_length, expected_type):
        if sequence_hash not in self.manifest: raise KeyError(f'missing embedding {sequence_hash}')
        item=self.manifest[sequence_hash]
        if item['sequence_type']!=expected_type: raise ValueError(f'wrong sequence type for {sequence_hash}')
        path=self.root/item['file_path']; value=np.load(path,mmap_mode='r')
        if value.ndim!=2 or value.shape!=(expected_length,1280):
            raise ValueError(f'embedding shape {value.shape} != ({expected_length}, 1280): {path}')
        return torch.from_numpy(np.array(value,dtype=np.float32,copy=True))

    def __getitem__(self,index):
        row=self.frame.iloc[index]; protein=str(row.prot_seq); peptide=str(row.pep_seq); label_text=str(row.protein_residue_label)
        if set(label_text)-{'0','1'}: raise ValueError(f'invalid label characters: {row.record_id}')
        if len(label_text)!=len(protein): raise ValueError(f'label/protein mismatch: {row.record_id}')
        protein_value=self._embedding(row.protein_sequence_hash,len(protein),'protein')
        peptide_value=self._embedding(row.peptide_sequence_hash,len(peptide),'peptide')
        labels=torch.tensor([int(x) for x in label_text],dtype=torch.long)
        if protein_value.shape[0]!=labels.shape[0]: raise ValueError(f'feature/label mismatch: {row.record_id}')
        return {'record_id':row.record_id,'dataset_index':int(row.dataset_index),'protein':protein_value,
                'peptide':peptide_value,'labels':labels,'protein_length':len(protein),'peptide_length':len(peptide)}


def umppi_collate(batch):
    proteins=[x['protein'] for x in batch]; peptides=[x['peptide'] for x in batch]; labels=[x['labels'] for x in batch]
    protein_pad=pad_sequence(proteins,batch_first=True,padding_value=0.0)
    peptide_pad=pad_sequence(peptides,batch_first=True,padding_value=0.0)
    label_pad=pad_sequence(labels,batch_first=True,padding_value=-100)
    protein_lengths=torch.tensor([x['protein_length'] for x in batch],dtype=torch.long)
    peptide_lengths=torch.tensor([x['peptide_length'] for x in batch],dtype=torch.long)
    protein_positions=torch.arange(protein_pad.shape[1]).unsqueeze(0)
    peptide_positions=torch.arange(peptide_pad.shape[1]).unsqueeze(0)
    protein_pad_mask=protein_positions>=protein_lengths.unsqueeze(1)
    peptide_pad_mask=peptide_positions>=peptide_lengths.unsqueeze(1)
    if not torch.equal(protein_pad_mask,label_pad.eq(-100)): raise ValueError('protein mask/label padding mismatch')
    if protein_pad.shape[:2]!=label_pad.shape: raise ValueError('padded protein/label mismatch')
    return {'record_ids':[x['record_id'] for x in batch],'dataset_indices':[x['dataset_index'] for x in batch],
            'protein':protein_pad,'peptide':peptide_pad,'labels':label_pad,
            'protein_pad_mask':protein_pad_mask,'peptide_pad_mask':peptide_pad_mask,
            'protein_lengths':protein_lengths,'peptide_lengths':peptide_lengths}
