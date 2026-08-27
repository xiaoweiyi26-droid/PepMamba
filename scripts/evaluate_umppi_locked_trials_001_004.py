#!/usr/bin/env python3
"""Exploratory fixed-threshold test evaluation for pre-locked trial_001-004."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from models import PepMamba
from umppi_dataset import UMppiProteinSiteDataset,umppi_collate
from umppi_reference_run import binary,collect,record_metrics,save_predictions,sha256,write_json
RUN=ROOT/'tuning_results/umppi_validation_tuning_20260808_125913'; OUT=RUN/'exploratory_test_trial_001_004'; LOCK=RUN/'MULTI_TRIAL_TEST_EVALUATION_LOCK.md'

def main():
 if OUT.exists(): raise RuntimeError('exploratory output already exists; refusing repeat evaluation')
 if not LOCK.exists(): raise RuntimeError('multi-trial pre-test lock missing')
 OUT.mkdir(parents=True,exist_ok=False); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 dataset=UMppiProteinSiteDataset(ROOT/'data/umppi_protein_site/test.csv',ROOT,ROOT/'embeddings/umppi_esm2_650m')
 if len(dataset)!=1310: raise RuntimeError('test record count changed')
 loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,collate_fn=umppi_collate,pin_memory=True); rows=[]
 for i in range(1,5):
  tid=f'trial_{i:03d}'; d=RUN/tid; dest=OUT/tid; dest.mkdir(); cfg=json.loads((d/'config.json').read_text()); sel=json.loads((d/'selected_validation_threshold.json').read_text()); ckpt=d/'best_validation_checkpoint.pth'; digest=sha256(ckpt)
  if digest!=sel['checkpoint_sha256'] or digest not in LOCK.read_text(): raise RuntimeError(f'{tid} checkpoint lock mismatch')
  threshold=float(sel['selected_threshold']); model=PepMamba(n_layers=cfg['n_layers'],d_model=cfg['d_model'],n_head=cfg['cross_attention_heads'],d_k=cfg['d_k'],d_v=cfg['d_v'],d_inner=cfg['d_model']*cfg['d_inner_multiplier'],dropout=cfg['dropout']).to(device); state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state['model_state_dict']); model.eval()
  y,p,records=collect(model,loader,device); save_predictions(dest/'test_predictions.npz',y,p,records); pooled={'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'threshold':threshold,**binary(y,p,threshold)}; recs,macro,unique=record_metrics(records,dataset.frame,threshold); pd.DataFrame(recs).to_csv(dest/'per_record_metrics.csv',index=False); write_json(dest/'pooled_test_metrics.json',pooled); write_json(dest/'macro_test_metrics.json',macro); write_json(dest/'unique_protein_metrics.json',unique)
  row={'trial_id':tid,'best_epoch':sel['best_epoch'],'fixed_validation_threshold':threshold,'mean_record_AUROC':macro['mean_record_AUROC'],'mean_record_MCC':macro['mean_record_MCC'],'median_record_AUROC':macro['median_record_AUROC'],'median_record_MCC':macro['median_record_MCC'],'pooled_AUROC':pooled['auroc'],'pooled_AUPRC':pooled['auprc'],'pooled_MCC':pooled['mcc'],'F1':pooled['f1'],'precision':pooled['precision'],'sensitivity':pooled['sensitivity'],'specificity':pooled['specificity'],'TP':pooled['tp'],'TN':pooled['tn'],'FP':pooled['fp'],'FN':pooled['fn'],'checkpoint_sha256':digest,'beats_UMPPI_AUROC_posthoc':macro['mean_record_AUROC']>0.914,'beats_UMPPI_MCC_posthoc':macro['mean_record_MCC']>0.489}; row['beats_UMPPI_both_posthoc']=row['beats_UMPPI_AUROC_posthoc'] and row['beats_UMPPI_MCC_posthoc']; rows.append(row); write_json(dest/'evaluation_receipt.json',{'evaluation_type':'exploratory; not valid for unbiased model selection','test_threshold_search_performed':False,'test_used_to_change_hyperparameters':False,**row}); print(json.dumps(row,indent=2),flush=True); del model; torch.cuda.empty_cache()
 df=pd.DataFrame(rows); df.to_csv(OUT/'exploratory_test_comparison.csv',index=False)
 md=['# Exploratory UMPPI trial_001–004 test comparison','','This comparison is post hoc and must not be used as an unbiased final-model selection procedure. Every checkpoint and threshold was locked beforehand; no test threshold scan was performed.','', '| Trial | Threshold | Mean-record AUROC | Mean-record MCC | Pooled AUROC | Pooled AUPRC | Pooled MCC | F1 | Precision | Sensitivity | Specificity |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for _,r in df.iterrows(): md.append(f"| {r.trial_id} | {r.fixed_validation_threshold:.3f} | {r.mean_record_AUROC:.6f} | {r.mean_record_MCC:.6f} | {r.pooled_AUROC:.6f} | {r.pooled_AUPRC:.6f} | {r.pooled_MCC:.6f} | {r.F1:.6f} | {r.precision:.6f} | {r.sensitivity:.6f} | {r.specificity:.6f} |")
 (OUT/'EXPLORATORY_TEST_REPORT.md').write_text('\n'.join(md)+'\n'); write_json(OUT/'evaluation_complete.json',{'trials':['trial_001','trial_002','trial_003','trial_004'],'each_checkpoint_evaluated_once':True,'test_threshold_search_performed':False,'valid_for_unbiased_model_selection':False,'completed_at':datetime.now(timezone.utc).isoformat()})
if __name__=='__main__':main()
