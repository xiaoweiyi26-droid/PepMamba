#!/usr/bin/env python3
"""Two-epoch UMPPI PepMamba smoke test. Never evaluates the test split."""
from __future__ import annotations
import json, math, os, random, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Subset

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from models import PepMamba
from umppi_dataset import UMppiProteinSiteDataset, umppi_collate

def seed_all(seed=42):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def metrics(y,p,t):
 pred=(p>=t).astype(int); tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); tn=int(((pred==0)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
 precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); specificity=tn/max(tn+fp,1); f1=2*precision*recall/max(precision+recall,1e-12)
 return {'mcc':float(matthews_corrcoef(y,pred)),'f1':f1,'precision':precision,'recall':recall,'specificity':specificity,'tp':tp,'fp':fp,'tn':tn,'fn':fn}

def evaluate(model,loader,device):
 model.eval(); all_y=[]; all_p=[]; records=[]; padded_y=[]; padded_p=[]
 with torch.no_grad():
  for batch in loader:
   protein=batch['protein'].to(device); peptide=batch['peptide'].to(device); labels=batch['labels'].to(device); mask=batch['protein_pad_mask'].to(device)
   logits,_=model(peptide,protein,class_mask=mask)
   if logits.shape[:2]!=labels.shape: raise ValueError('prediction/label padded shape mismatch')
   prob=F.softmax(logits,dim=-1)[...,1]
   valid=~mask
   padded_y.extend(labels[valid].cpu().numpy()); padded_p.extend(prob[valid].cpu().numpy())
   for i,rid in enumerate(batch['record_ids']):
    length=int(batch['protein_lengths'][i]); y=labels[i,:length].cpu().numpy(); p=prob[i,:length].cpu().numpy()
    if len(y)!=length or len(p)!=length: raise ValueError('per-record prediction length mismatch')
    if len(np.unique(y))==2:
     records.append({'record_id':rid,'length':length,'auc':float(roc_auc_score(y,p)),'y':y,'p':p})
    all_y.extend(y); all_p.extend(p)
 y=np.asarray(all_y); p=np.asarray(all_p); py=np.asarray(padded_y); pp=np.asarray(padded_p)
 if not np.array_equal(y,py) or not np.allclose(p,pp): raise RuntimeError('padding changed pooled metric inputs')
 auc=float(roc_auc_score(y,p)); auprc=float(average_precision_score(y,p)); scans=[]
 for threshold in np.arange(.1,.901,.01): scans.append({'threshold':float(threshold),**metrics(y,p,float(threshold))})
 chosen=max(scans,key=lambda x:(x['mcc'],x['f1'],-x['threshold'])); rec=[]
 for r in records:
  rec.append({'record_id':r['record_id'],'length':r['length'],'auc':r['auc'],'mcc':metrics(r['y'],r['p'],chosen['threshold'])['mcc']})
 return {'auc':auc,'auprc':auprc,'threshold':chosen['threshold'],**{k:v for k,v in chosen.items() if k!='threshold'},
         'mean_record_auc':float(np.mean([r['auc'] for r in rec])),'mean_record_mcc':float(np.mean([r['mcc'] for r in rec])),
         'n_valid_records':len(rec),'records':rec,'threshold_scan':scans,'padding_metric_inputs_identical':True}

def main():
 seed_all(42); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 data=ROOT/'data/umppi_protein_site'; emb=ROOT/'embeddings/umppi_esm2_650m'
 train_ds=UMppiProteinSiteDataset(data/'train.csv',ROOT,emb); val_ds=UMppiProteinSiteDataset(data/'validation.csv',ROOT,emb)
 lengths=train_ds.frame.prot_seq.str.len(); longest=int(lengths.idxmax()); short=list(lengths.nsmallest(23).index); train_idx=short+[longest]
 val_idx=list(val_ds.frame.prot_seq.str.len().sort_values().iloc[np.linspace(0,len(val_ds)-1,24,dtype=int)].index)
 train_loader=DataLoader(Subset(train_ds,train_idx),batch_size=1,shuffle=True,num_workers=0,collate_fn=umppi_collate)
 val_loader=DataLoader(Subset(val_ds,val_idx),batch_size=1,shuffle=False,num_workers=0,collate_fn=umppi_collate)
 # Explicit two-record padding/mask check before training.
 padding_batch=umppi_collate([train_ds[short[0]],train_ds[short[-1]]])
 protein_mask_ok=bool(torch.equal(padding_batch['protein_pad_mask'],padding_batch['labels'].eq(-100)))
 peptide_mask_ok=bool((padding_batch['peptide'][padding_batch['peptide_pad_mask']].eq(0)).all())
 valid_peptide_nonzero=bool((padding_batch['peptide'][~padding_batch['peptide_pad_mask']].abs().sum(dim=-1)>0).all())
 model=PepMamba(n_layers=1,d_model=128,n_head=4,d_k=32,d_v=32,d_inner=256,dropout=.2).to(device)
 optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); weight=torch.tensor([1.,5.],device=device)
 history=[]; finite=True; long_trained=False; prediction_lengths=True
 for epoch in range(1,3):
  model.train(); losses=[]
  for batch in train_loader:
   protein=batch['protein'].to(device); peptide=batch['peptide'].to(device); labels=batch['labels'].to(device); mask=batch['protein_pad_mask'].to(device)
   optimizer.zero_grad(set_to_none=True); logits,_=model(peptide,protein,class_mask=mask)
   prediction_lengths &= logits.shape[:2]==labels.shape
   loss=F.cross_entropy(logits.reshape(-1,2),labels.reshape(-1),weight=weight,ignore_index=-100)
   if not torch.isfinite(loss): finite=False; raise RuntimeError('NaN/Inf loss')
   loss.backward(); grad_ok=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
   if not grad_ok: finite=False; raise RuntimeError('NaN/Inf gradient')
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach()))
   long_trained |= int(batch['protein_lengths'].max())>1022
  val=evaluate(model,val_loader,device); history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':{k:v for k,v in val.items() if k not in ('records','threshold_scan')}})
 out=ROOT/'smoke_results'/f"umppi_seed42_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True)
 (out/'history.json').write_text(json.dumps(history,indent=2)+'\n'); (out/'validation_threshold_scan.json').write_text(json.dumps(val['threshold_scan'],indent=2)+'\n'); (out/'validation_per_record_metrics.json').write_text(json.dumps(val['records'],indent=2)+'\n')
 checks={'forward_pass':True,'backward_pass':True,'finite_loss_and_gradients':finite,'loss_decreased':history[-1]['train_loss']<history[0]['train_loss'],
  'protein_mask_correct':protein_mask_ok,'peptide_mask_correct':peptide_mask_ok and valid_peptide_nonzero,'padding_does_not_change_metric_inputs':val['padding_metric_inputs_identical'],
  'prediction_label_lengths_identical':prediction_lengths,'long_protein_trained':long_trained,'validation_metrics_computed':math.isfinite(val['auc']) and math.isfinite(val['mcc']),
  'per_record_metrics_computed':val['n_valid_records']>0,'test_used_for_selection':False,'test_threshold_search_performed':False,'test_evaluated':False}
 positive_checks=[v for k,v in checks.items() if k not in ('test_used_for_selection','test_threshold_search_performed','test_evaluated')]
 ready=all(positive_checks) and not checks['test_used_for_selection'] and not checks['test_threshold_search_performed'] and not checks['test_evaluated']
 result={'seed':42,'epochs':2,'train_subset_records':len(train_idx),'validation_subset_records':len(val_idx),'longest_training_protein':int(lengths.max()),'history':history,'selected_validation_threshold':val['threshold'],'checks':checks,'READY_FOR_FULL_TRAINING':ready,'output_dir':str(out)}
 (out/'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n')
 report='# UMPPI PepMamba smoke test report\n\n'+f"- Seed: 42\n- Epochs: 2\n- Train/validation subset: {len(train_idx)}/{len(val_idx)} records\n- Longest protein trained: {int(lengths.max())}\n- Epoch losses: {[round(x['train_loss'],6) for x in history]}\n- Final validation AUROC/AUPRC/MCC: {val['auc']:.6f}/{val['auprc']:.6f}/{val['mcc']:.6f}\n- Validation-selected threshold: {val['threshold']:.2f}\n- Test evaluated: false\n\n## Checks\n\n"+'\n'.join(f"- {k}: **{str(v).lower()}**" for k,v in checks.items())+f"\n\n`READY_FOR_FULL_TRAINING = {str(ready).upper()}`\n"
 (ROOT/'UMPPI_SMOKE_TEST_REPORT.md').write_text(report); print(json.dumps(result,indent=2))

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description='PepMamba UMPPI smoke test or protocol-locked full reference run')
 parser.add_argument('--seed',type=int,default=42)
 parser.add_argument('--epochs',type=int,default=2)
 parser.add_argument('--full-run',action='store_true')
 args=parser.parse_args()
 if args.full_run:
  from umppi_reference_run import main as full_run
  full_run(seed=args.seed,epochs=args.epochs)
 else:
  if args.seed!=42 or args.epochs!=2: raise ValueError('Smoke mode is fixed to seed=42 and 2 epochs')
  main()
