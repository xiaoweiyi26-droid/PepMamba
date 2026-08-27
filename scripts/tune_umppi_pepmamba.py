#!/usr/bin/env python3
"""Validation-only UMPPI tuning. This program never reads the held-out split."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('PYTHONHASHSEED','42'); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NVIDIA_TF32_OVERRIDE','0'); os.environ.setdefault('PEPMAMBA_DETERMINISTIC_KERNELS','1'); os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from config import BEST_PARAMS
from models import PepMamba
from umppi_dataset import UMppiProteinSiteDataset,umppi_collate
from umppi_reference_run import HybridLoss,collect,save_predictions,seed_all,sha256,write_csv,write_json
TRAIN_FILE=ROOT/'data/umppi_protein_site/train.csv'
VALIDATION_FILE=ROOT/'data/umppi_protein_site/validation.csv'
EMBEDDING_DIR=ROOT/'embeddings/umppi_esm2_650m'
SEARCH_SPACE={'learning_rate':[6e-5,2.5e-4,'log-uniform'],'dropout':[.15,.35],'focal_alpha':[.55,.85],
 'focal_gamma':[1.2,3.2],'ce_weight':[.40,.80],'weight_decay':[3e-5,3e-3,'log-uniform'],
 'class_weight_power':[.25,.75],'max_pos_weight':[4.,10.],'batch_size':[4,8,12,16],
 'n_layers':[2,3,4],'d_model':[96,128,160],'cross_attention_heads':[4,8]}

def log_uniform(rng,low,high): return float(np.exp(rng.uniform(np.log(low),np.log(high))))
def make_configs(n,seed):
 rng=np.random.default_rng(seed); configs=[]
 for i in range(n):
  sampled_batch_size=int(rng.choice([4,8,12,16]))
  cfg={'trial_id':f'trial_{i:03d}','seed':seed,'epochs':35,
   'learning_rate':log_uniform(rng,6e-5,2.5e-4),'dropout':float(rng.uniform(.15,.35)),
   'focal_alpha':float(rng.uniform(.55,.85)),'focal_gamma':float(rng.uniform(1.2,3.2)),
   'ce_weight':float(rng.uniform(.40,.80)),'weight_decay':log_uniform(rng,3e-5,3e-3),
   'class_weight_power':float(rng.uniform(.25,.75)),'max_pos_weight':float(rng.uniform(4.,10.)),
   'batch_size':sampled_batch_size,'n_layers':int(rng.choice([2,3,4])),
   'd_model':int(rng.choice([96,128,160])),'cross_attention_heads':int(rng.choice([4,8])),
   'd_k':32,'d_v':32,'d_inner_multiplier':2,'effective_batch_size':(16 if sampled_batch_size==16 else 24),
   'scheduler':{'name':'CosineAnnealingWarmRestarts','T_0':10,'eta_min':1e-7},'grad_clip':1.0}
  configs.append(cfg)
 return configs

def mcc_counts(tp,tn,fp,fn):
 den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
 return float((tp*tn-fp*fn)/math.sqrt(den)) if den>0 else 0.0

def threshold_metrics(records,pooled_y,pooled_p):
 rows=[]
 for t in np.round(np.arange(.05,.9501,.005),3):
  record_mcc=[]
  for rec in records:
   y=rec['y']; pred=rec['p']>=t; tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
   if len(np.unique(y))==2: record_mcc.append(mcc_counts(tp,tn,fp,fn))
  pred=pooled_p>=t; tp=int(((pred==1)&(pooled_y==1)).sum()); tn=int(((pred==0)&(pooled_y==0)).sum()); fp=int(((pred==1)&(pooled_y==0)).sum()); fn=int(((pred==0)&(pooled_y==1)).sum())
  precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1=2*precision*recall/max(precision+recall,1e-15)
  rows.append({'threshold':float(t),'mean_record_MCC':float(np.mean(record_mcc)),'n_valid_records':len(record_mcc),'pooled_MCC':mcc_counts(tp,tn,fp,fn),'pooled_F1':f1,'pooled_precision':precision,'pooled_recall':recall,'tp':tp,'tn':tn,'fp':fp,'fn':fn})
 chosen=max(rows,key=lambda r:(r['mean_record_MCC'],r['pooled_MCC'],r['pooled_F1'],-r['threshold']))
 return rows,chosen

def summarize_validation(y,p,records,chosen):
 aucs=[roc_auc_score(r['y'],r['p']) for r in records if len(np.unique(r['y']))==2]
 return {'validation_mean_record_AUROC':float(np.mean(aucs)),'validation_mean_record_MCC':chosen['mean_record_MCC'],
  'validation_n_valid_records':len(aucs),'validation_pooled_AUROC':float(roc_auc_score(y,p)),
  'validation_pooled_AUPRC':float(average_precision_score(y,p)),'validation_pooled_MCC':chosen['pooled_MCC'],
  'validation_F1':chosen['pooled_F1'],'selected_threshold':chosen['threshold']}

def make_batches(lengths,nominal,layers,d_model,seed,effective_target):
 # Limit padded self-attention work to approximately the successful reference batch.
 budget=48_000_000*(3*128)/(layers*d_model); order=np.argsort(lengths); batches=[]; i=0
 while i<len(order):
  length=max(int(lengths[order[i]]),1); cap=max(1,min(nominal,int(budget/(length*length)))); batch=order[i:i+cap].tolist(); batches.append(batch); i+=len(batch)
 random.Random(seed).shuffle(batches)
 denominators=[0]*len(batches); step_after=[False]*len(batches); group=[]; count=0
 for idx,batch in enumerate(batches):
  if group and count+len(batch)>effective_target:
   for j in group: denominators[j]=count
   step_after[group[-1]]=True; group=[]; count=0
  group.append(idx); count+=len(batch)
 if group:
  for j in group: denominators[j]=count
  step_after[group[-1]]=True
 return batches,denominators,step_after

def train_trial(cfg,train_ds,val_ds,out,device,global_log):
 trial_dir=out/cfg['trial_id']; trial_dir.mkdir(parents=True,exist_ok=False); write_json(trial_dir/'config.json',cfg)
 def tlog(msg):
  line=f"[{datetime.now(timezone.utc).isoformat()}] {msg}"; print(line,flush=True)
  with (trial_dir/'train.log').open('a') as f: f.write(line+'\n')
  with global_log.open('a') as f: f.write(line+'\n')
 seed_all(cfg['seed']); labels=''.join(train_ds.frame.protein_residue_label.astype(str)); positives=labels.count('1'); negatives=labels.count('0')
 pos_weight=min((negatives/positives)**cfg['class_weight_power'],cfg['max_pos_weight']); cfg['resolved_positive_class_weight']=pos_weight; write_json(trial_dir/'config.json',cfg)
 model=PepMamba(n_layers=cfg['n_layers'],d_model=cfg['d_model'],n_head=cfg['cross_attention_heads'],d_k=cfg['d_k'],d_v=cfg['d_v'],d_inner=cfg['d_model']*cfg['d_inner_multiplier'],dropout=cfg['dropout']).to(device)
 weights=torch.tensor([1.,pos_weight],device=device); criterion=HybridLoss(cfg['ce_weight'],weights,cfg['focal_alpha'],cfg['focal_gamma'])
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']); scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,eta_min=1e-7)
 val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=umppi_collate,pin_memory=True)
 lengths=train_ds.frame.prot_seq.str.len().to_numpy(); history=[]; best_key=None; ckpt=trial_dir/'best_validation_checkpoint.pth'
 for epoch in range(1,cfg['epochs']+1):
  batches,denominators,step_after=make_batches(lengths,cfg['batch_size'],cfg['n_layers'],cfg['d_model'],cfg['seed']+epoch,cfg['effective_batch_size'])
  loader=DataLoader(train_ds,batch_sampler=batches,num_workers=0,collate_fn=umppi_collate,pin_memory=True)
  model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
  for step,b in enumerate(loader,1):
   protein=b['protein'].to(device,non_blocking=True); peptide=b['peptide'].to(device,non_blocking=True); targets=b['labels'].to(device); mask=b['protein_pad_mask'].to(device)
   logits,_=model(peptide,protein,class_mask=mask); loss=criterion(logits,targets)
   if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss {cfg['trial_id']} epoch={epoch} step={step}")
   batch_records=len(b['record_ids']); (loss*batch_records/denominators[step-1]).backward(); losses.append(float(loss.detach()))
   if step_after[step-1]:
    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']); optimizer.step(); optimizer.zero_grad(set_to_none=True)
  vy,vp,vrecords=collect(model,val_loader,device); scan_rows,chosen=threshold_metrics(vrecords,vy,vp); summary=summarize_validation(vy,vp,vrecords,chosen)
  row={'epoch':epoch,'learning_rate':optimizer.param_groups[0]['lr'],'train_loss':float(np.mean(losses)),**summary}; history.append(row)
  key=(summary['validation_mean_record_AUROC'],summary['validation_mean_record_MCC'],summary['validation_pooled_AUPRC'])
  if best_key is None or key>best_key:
   best_key=key; torch.save({'model_state_dict':model.state_dict(),'best_epoch':epoch,'selection_key':key,'config':cfg},ckpt)
  write_csv(trial_dir/'training_history.csv',history); tlog(f"{cfg['trial_id']} epoch {epoch:02d}/35 loss={row['train_loss']:.6f} mean_record_auc={summary['validation_mean_record_AUROC']:.6f} mean_record_mcc={summary['validation_mean_record_MCC']:.6f} pooled_auprc={summary['validation_pooled_AUPRC']:.6f}"); scheduler.step(epoch)
 state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state['model_state_dict']); model.eval(); vy,vp,vrecords=collect(model,val_loader,device); scan_rows,chosen=threshold_metrics(vrecords,vy,vp); summary=summarize_validation(vy,vp,vrecords,chosen)
 write_csv(trial_dir/'validation_threshold_scan.csv',scan_rows); save_predictions(trial_dir/'validation_predictions.npz',vy,vp,vrecords); digest=sha256(ckpt)
 selected={'trial_id':cfg['trial_id'],'best_epoch':state['best_epoch'],**summary,'checkpoint_sha256':digest,'threshold_selection':'maximum validation mean-record MCC; ties pooled MCC, validation F1, lower threshold','held_out_split_loaded':False,'held_out_metrics_used':False}; write_json(trial_dir/'selected_validation_threshold.json',selected); tlog(f"{cfg['trial_id']} COMPLETE best_epoch={state['best_epoch']} threshold={summary['selected_threshold']:.3f}")
 del model,optimizer,criterion; torch.cuda.empty_cache(); return {**selected,'checkpoint_path':str(ckpt),'config_path':str(trial_dir/'config.json'),**{k:v for k,v in cfg.items() if not isinstance(v,dict)}}

def normalize(values):
 values=np.asarray(values,float); lo=float(values.min()); hi=float(values.max()); return np.ones(len(values)) if hi==lo else (values-lo)/(hi-lo)
def pareto(rows):
 result=[]
 for a in rows:
  dominated=any((b['validation_mean_record_AUROC']>=a['validation_mean_record_AUROC'] and b['validation_mean_record_MCC']>=a['validation_mean_record_MCC'] and (b['validation_mean_record_AUROC']>a['validation_mean_record_AUROC'] or b['validation_mean_record_MCC']>a['validation_mean_record_MCC'])) for b in rows)
  if not dominated: result.append(a)
 return result
def finalize_tables(out,rows):
 nmcc=normalize([r['validation_mean_record_MCC'] for r in rows]); nauc=normalize([r['validation_mean_record_AUROC'] for r in rows]); nap=normalize([r['validation_pooled_AUPRC'] for r in rows])
 for i,r in enumerate(rows): r['composite_score']=.45*nmcc[i]+.35*nauc[i]+.20*nap[i]
 write_csv(out/'all_trials.csv',rows); ranked=sorted(rows,key=lambda r:(r['validation_mean_record_MCC'],r['validation_mean_record_AUROC'],r['validation_pooled_AUPRC']),reverse=True)
 for i,r in enumerate(ranked,1): r['validation_rank']=i
 write_csv(out/'validation_rank.csv',ranked); write_csv(out/'pareto_front.csv',pareto(rows))
 top_n=max(1,math.ceil(len(rows)*.25)); auc_top=sorted(rows,key=lambda r:r['validation_mean_record_AUROC'],reverse=True)[:top_n]; max_mcc=max(r['validation_mean_record_MCC'] for r in auc_top); near=[r for r in auc_top if max_mcc-r['validation_mean_record_MCC']<.005]
 selected=max(near,key=lambda r:(r['validation_mean_record_AUROC'],r['validation_pooled_AUPRC']))
 stamp=datetime.now(timezone.utc).isoformat(); selection=f"""# FINAL SELECTION BEFORE HELD-OUT EVALUATION

- Selected trial: `{selected['trial_id']}`
- Locked at: {stamp}
- Selection rule: restrict to the top 25% by validation mean-record AUROC ({top_n}/{len(rows)} trials); identify the highest validation mean-record MCC; when the MCC gap is <0.005, prefer higher mean-record AUROC; then pooled AUPRC.
- Validation mean-record AUROC: {selected['validation_mean_record_AUROC']:.9f}
- Validation mean-record MCC: {selected['validation_mean_record_MCC']:.9f}
- Validation pooled AUPRC: {selected['validation_pooled_AUPRC']:.9f}
- Validation pooled AUROC/MCC/F1: {selected['validation_pooled_AUROC']:.9f} / {selected['validation_pooled_MCC']:.9f} / {selected['validation_F1']:.9f}
- Locked threshold: {selected['selected_threshold']:.3f}
- Best epoch: {selected['best_epoch']}
- Checkpoint SHA-256: `{selected['checkpoint_sha256']}`
- Checkpoint: `{selected['checkpoint_path']}`
- Held-out split accessed during tuning: false
- Held-out metrics used for selection: false

This file is the irreversible selection lock. No alternative trial may be selected after held-out evaluation.
"""; (out/'FINAL_SELECTION_BEFORE_TEST.md').write_text(selection)
 report=f"# UMPPI validation-only tuning report\n\n- Trials: {len(rows)}\n- Seed: 42 for every trial\n- Epochs: 35\n- Ranking: validation mean-record MCC, then mean-record AUROC, then pooled AUPRC\n- Selected final trial: `{selected['trial_id']}`\n- Held-out split accessed: false\n- Final selection lock: `FINAL_SELECTION_BEFORE_TEST.md`\n"; (out/'tuning_report.md').write_text(report)
 write_json(out/'tuning_complete.json',{'completed':True,'trials':len(rows),'selected_trial':selected['trial_id'],'selection_locked_at':stamp,'held_out_split_loaded':False,'held_out_metrics_used':False})
 return selected

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=42); parser.add_argument('--trials',type=int,default=6); parser.add_argument('--epochs',type=int,default=35); parser.add_argument('--dry-run',action='store_true'); parser.add_argument('--output-dir',type=Path); args=parser.parse_args()
 if args.seed!=42 or args.trials!=6 or args.epochs!=35: raise ValueError('Formal tuning is locked to seed=42, MAX_TRIALS=6, 35 epochs')
 configs=make_configs(args.trials,args.seed)
 if args.dry_run:
  print(json.dumps({'protocol':'validation-only','train_file':str(TRAIN_FILE),'validation_file':str(VALIDATION_FILE),'search_space':SEARCH_SPACE,'configs':configs},indent=2)); return
 seed_all(args.seed); train_ds=UMppiProteinSiteDataset(TRAIN_FILE,ROOT,EMBEDDING_DIR); val_ds=UMppiProteinSiteDataset(VALIDATION_FILE,ROOT,EMBEDDING_DIR)
 if (len(train_ds),len(val_ds))!=(4726,539): raise RuntimeError('locked split counts changed')
 out=args.output_dir or ROOT/'tuning_results'/f"umppi_validation_tuning_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True,exist_ok=False); write_json(out/'search_protocol.json',{'seed':42,'trials':24,'epochs':35,'search_space':SEARCH_SPACE,'ranking':['validation mean-record MCC','validation mean-record AUROC','validation pooled AUPRC'],'checkpoint_selection':['validation mean-record AUROC','validation mean-record MCC','validation pooled AUPRC'],'held_out_split_loaded':False,'held_out_metrics_used':False,'configs':configs})
 global_log=out/'tuning.log'; rows=[]; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 for cfg in configs:
  rows.append(train_trial(cfg,train_ds,val_ds,out,device,global_log)); write_csv(out/'all_trials.partial.csv',rows)
 selected=finalize_tables(out,rows); print(json.dumps({'output_dir':str(out),'selected_trial':selected['trial_id'],'held_out_evaluation_started':False},indent=2))
if __name__=='__main__': main()
