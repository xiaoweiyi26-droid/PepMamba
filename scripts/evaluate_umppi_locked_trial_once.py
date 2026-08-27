#!/usr/bin/env python3
"""One-time held-out evaluation of the durably locked UMPPI trial."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from models import PepMamba
from umppi_dataset import UMppiProteinSiteDataset,umppi_collate
from sklearn.metrics import average_precision_score,roc_auc_score
from umppi_reference_run import binary,collect,record_metrics,save_predictions,sha256,write_json

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--trial-id',default='trial_000'); args=ap.parse_args(); run=args.run_dir; trial=run/args.trial_id; output=run/f'final_test_{args.trial_id}'
 lock=run/'FINAL_SELECTION_BEFORE_TEST.md'
 if not lock.exists(): raise RuntimeError('selection lock missing')
 if output.exists(): raise RuntimeError('held-out output exists; refusing repeat evaluation')
 sel=json.loads((trial/'selected_validation_threshold.json').read_text()); cfg=json.loads((trial/'config.json').read_text()); ckpt=trial/'best_validation_checkpoint.pth'
 if args.trial_id!='trial_000' or f'selected_trial: `{args.trial_id}`' not in lock.read_text(): raise RuntimeError('requested trial does not match pre-test lock')
 digest=sha256(ckpt)
 if digest!=sel['checkpoint_sha256']: raise RuntimeError('checkpoint SHA mismatch')
 threshold=float(sel['selected_threshold']); output.mkdir(parents=True,exist_ok=False)
 write_json(output/'evaluation_protocol.json',{'selected_trial':args.trial_id,'checkpoint_sha256':digest,'validation_selected_threshold':threshold,'test_used_for_selection':False,'test_threshold_search_performed':False,'evaluation_started_at':datetime.now(timezone.utc).isoformat()})
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model=PepMamba(n_layers=cfg['n_layers'],d_model=cfg['d_model'],n_head=cfg['cross_attention_heads'],d_k=cfg['d_k'],d_v=cfg['d_v'],d_inner=cfg['d_model']*cfg['d_inner_multiplier'],dropout=cfg['dropout']).to(device)
 state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state['model_state_dict']); model.eval()
 dataset=UMppiProteinSiteDataset(ROOT/'data/umppi_protein_site/test.csv',ROOT,ROOT/'embeddings/umppi_esm2_650m')
 if len(dataset)!=1310: raise RuntimeError('held-out record count changed')
 loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,collate_fn=umppi_collate,pin_memory=True)
 y,p,records=collect(model,loader,device); save_predictions(output/'test_predictions.npz',y,p,records)
 pm={'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'threshold':threshold,**binary(y,p,threshold)}; rows,macro,unique=record_metrics(records,dataset.frame,threshold)
 pd.DataFrame(rows).to_csv(output/'per_record_metrics.csv',index=False); write_json(output/'pooled_test_metrics.json',pm); write_json(output/'macro_test_metrics.json',macro); write_json(output/'unique_protein_metrics.json',unique)
 comparison={'beats_UMPPI_AUROC':macro['mean_record_AUROC']>0.914,'beats_UMPPI_MCC':macro['mean_record_MCC']>0.489}; comparison['beats_UMPPI_both']=comparison['beats_UMPPI_AUROC'] and comparison['beats_UMPPI_MCC']
 receipt={'status':'complete','selected_trial':args.trial_id,'best_epoch':sel['best_epoch'],'checkpoint_sha256':digest,'fixed_validation_threshold':threshold,'n_test_records':len(dataset),'test_used_for_selection':False,'test_threshold_search_performed':False,'evaluated_exactly_once_by_this_entrypoint':True,'macro_record':macro,'pooled':pm,'unique_protein':unique,'posthoc_reported_target_comparison':comparison,'completed_at':datetime.now(timezone.utc).isoformat()}; write_json(output/'test_evaluation_receipt.json',receipt)
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
