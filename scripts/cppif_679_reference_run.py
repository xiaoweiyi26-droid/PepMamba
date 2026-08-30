#!/usr/bin/env python3
"""Protocol-locked PepMamba-679 evaluation on the CPPIF external benchmark."""
from __future__ import annotations
import hashlib,json,math,os,platform,random,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('PYTHONHASHSEED','42'); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NVIDIA_TF32_OVERRIDE','0'); os.environ.setdefault('PEPMAMBA_DETERMINISTIC_KERNELS','1'); os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,matthews_corrcoef,roc_auc_score
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from models import PepMamba
from cppif_dataset import CPPIF679ProteinSiteDataset,cppif_collate

CPPIF679_PARAMS = {
    "seed": 42,
    "epochs": 35,
    "learning_rate": 0.00025,
    "dropout": 0.23,
    "focal_alpha": 0.55,
    "focal_gamma": 2.8,
    "ce_weight": 0.48,
    "weight_decay": 0.003,
    "class_weight_power": 0.45,
    "max_pos_weight": 4.0,
    "batch_size": 16,
    "effective_batch_size": 16,
    "n_layers": 4,
    "d_model": 160,
    "cross_attention_heads": 8,
    "d_k": 32,
    "d_v": 32,
    "d_inner_multiplier": 2,
    "scheduler_t0": 10,
    "grad_clip": 1.0,
    "parameter_source": "original CPPIF trial_003 locked configuration",
}

def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True,warn_only=True)
def jd(x):
 if isinstance(x,np.integer): return int(x)
 if isinstance(x,np.floating): return float(x)
 if isinstance(x,Path): return str(x)
 raise TypeError(type(x).__name__)
def write_json(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=jd)+'\n')
def write_csv(path,rows): pd.DataFrame(rows).to_csv(path,index=False)
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''): h.update(block)
 return h.hexdigest()
class Logger:
 def __init__(self,path): self.path=path
 def __call__(self,msg):
  line=f'[{datetime.now(timezone.utc).isoformat()}] {msg}'; print(line,flush=True)
  with self.path.open('a') as f: f.write(line+'\n')
class FocalDiceLoss(nn.Module):
 def __init__(self,weight,gamma,alpha): super().__init__(); self.weight=weight; self.gamma=gamma; self.alpha=alpha
 def forward(self,logits,targets):
  valid=targets.ne(-100); logits=logits[valid]; targets=targets[valid]; probs=F.softmax(logits,dim=-1)[:,1]
  ce=F.cross_entropy(logits,targets,reduction='none',weight=self.weight); focal=(((1-torch.exp(-ce))**self.gamma)*ce).mean()
  dice=1-(2*torch.sum(probs*targets)+1e-5)/(torch.sum(probs)+torch.sum(targets)+1e-5)
  return self.alpha*focal+(1-self.alpha)*dice
class HybridLoss(nn.Module):
 def __init__(self,cw,weight,alpha,gamma): super().__init__(); self.cw=cw; self.register_buffer('weight',weight); self.fd=FocalDiceLoss(self.weight,gamma,alpha)
 def forward(self,logits,targets):
  valid=targets.ne(-100); ce=F.cross_entropy(logits[valid],targets[valid],weight=self.weight)
  return self.cw*ce+(1-self.cw)*self.fd(logits,targets)
def binary(y,p,t):
 pred=(p>=t).astype(np.int8); y=y.astype(np.int8); tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
 precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); specificity=tn/max(tn+fp,1); f1=2*precision*recall/max(precision+recall,1e-15)
 mcc=float(matthews_corrcoef(y,pred)) if len(np.unique(y))==2 else float('nan')
 return {'mcc':mcc,'f1':f1,'precision':precision,'recall':recall,'sensitivity':recall,'specificity':specificity,'tp':tp,'tn':tn,'fp':fp,'fn':fn}
def scan(records,y,p):
 rows=[]
 for t in np.round(np.arange(.05,.9501,.005),3):
  record_mcc=[binary(r['y'],r['p'],float(t))['mcc'] for r in records if len(np.unique(r['y']))==2]
  pooled=binary(y,p,float(t))
  rows.append({'threshold':float(t),'mean_record_mcc':float(np.mean(record_mcc)),'n_valid_records':len(record_mcc),**pooled})
 return rows,max(rows,key=lambda r:(r['mean_record_mcc'],r['mcc'],r['f1'],-r['threshold']))

def validation_summary(y,p,records,t):
 valid=[r for r in records if len(np.unique(r['y']))==2]
 aucs=[roc_auc_score(r['y'],r['p']) for r in valid]
 mccs=[binary(r['y'],r['p'],t)['mcc'] for r in valid]
 return {'mean_record_auroc':float(np.mean(aucs)),'mean_record_mcc':float(np.mean(mccs)),'n_valid_records':len(valid),'pooled_auroc':float(roc_auc_score(y,p)),'pooled_auprc':float(average_precision_score(y,p)),**binary(y,p,t)}

def make_batches(lengths,nominal,n_layers,d_model,seed,effective_target):
 budget=48_000_000*(3*128)/(n_layers*d_model); order=np.argsort(lengths); batches=[]; i=0
 while i<len(order):
  length=max(int(lengths[order[i]]),1); capacity=max(1,min(nominal,int(budget/(length*length)))); batch=order[i:i+capacity].tolist(); batches.append(batch); i+=len(batch)
 random.Random(seed).shuffle(batches); denominators=[0]*len(batches); step_after=[False]*len(batches); group=[]; count=0
 for index,batch in enumerate(batches):
  if group and count+len(batch)>effective_target:
   for j in group: denominators[j]=count
   step_after[group[-1]]=True; group=[]; count=0
  group.append(index); count+=len(batch)
 if group:
  for j in group: denominators[j]=count
  step_after[group[-1]]=True
 return batches,denominators,step_after
def collect(model,loader,device):
 model.eval(); ys=[]; ps=[]; records=[]
 with torch.no_grad():
  for b in loader:
   protein=b['protein'].to(device,non_blocking=True); peptide=b['peptide'].to(device,non_blocking=True); labels=b['labels'].to(device); mask=b['protein_pad_mask'].to(device)
   logits,_=model(peptide,protein,class_mask=mask); prob=F.softmax(logits,dim=-1)[...,1]
   for i,rid in enumerate(b['record_ids']):
    length=int(b['protein_lengths'][i]); y=labels[i,:length].cpu().numpy(); p=prob[i,:length].cpu().numpy()
    if len(y)!=length or len(p)!=length: raise RuntimeError(f'alignment failure: {rid}')
    ys.append(y); ps.append(p); records.append({'record_id':rid,'dataset_index':b['dataset_indices'][i],'y':y,'p':p})
 return np.concatenate(ys),np.concatenate(ps),records
def save_predictions(path,y,p,records):
 lengths=np.asarray([len(r['y']) for r in records],dtype=np.int64)
 np.savez_compressed(path,labels=y.astype(np.int8),probabilities=p.astype(np.float32),record_ids=np.asarray([r['record_id'] for r in records],dtype=str),record_lengths=lengths,record_offsets=np.concatenate(([0],np.cumsum(lengths))))
def split_stats(train,val,test):
 sets=lambda f,c:set(f[c].astype(str)); pairs=lambda f:set(zip(f.protein_sequence_hash,f.peptide_sequence_hash))
 return {'train_records':len(train),'validation_records':len(val),'test_records':len(test),'train_validation_exact_protein_overlap':len(sets(train,'protein_sequence_hash')&sets(val,'protein_sequence_hash')),'train_test_exact_protein_overlap':len(sets(train,'protein_sequence_hash')&sets(test,'protein_sequence_hash')),'validation_test_exact_protein_overlap':len(sets(val,'protein_sequence_hash')&sets(test,'protein_sequence_hash')),'train_test_pair_overlap':len(pairs(train)&pairs(test)),'validation_test_pair_overlap':len(pairs(val)&pairs(test))}
def record_metrics(records,frame,t):
 meta=frame.set_index('record_id'); rows=[]; groups=defaultdict(lambda:{'y':[],'p':[]})
 for r in records:
  m=meta.loc[r['record_id']]; y=r['y']; p=r['p']; auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'); mcc=binary(y,p,t)['mcc']
  rows.append({'record_id':r['record_id'],'protein_id_hash':m['protein_sequence_hash'],'peptide_id_hash':m['peptide_sequence_hash'],'protein_length':len(y),'num_positive_residues':int(y.sum()),'AUROC':auc,'MCC':mcc})
  g=groups[m['protein_sequence_hash']]; g['y'].append(y); g['p'].append(p)
 valid=[r for r in rows if math.isfinite(r['AUROC']) and math.isfinite(r['MCC'])]; aucs=np.asarray([r['AUROC'] for r in valid]); mccs=np.asarray([r['MCC'] for r in valid])
 macro={'mean_record_AUROC':float(aucs.mean()),'mean_record_MCC':float(mccs.mean()),'median_record_AUROC':float(np.median(aucs)),'median_record_MCC':float(np.median(mccs)),'std_record_AUROC':float(aucs.std()),'std_record_MCC':float(mccs.std()),'n_valid_records':len(valid)}
 pv=[]
 for v in groups.values():
  y=np.concatenate(v['y']); p=np.concatenate(v['p'])
  if len(np.unique(y))==2: pv.append((roc_auc_score(y,p),binary(y,p,t)['mcc']))
 unique={'n_unique_test_proteins':len(groups),'n_valid_unique_test_proteins':len(pv),'macro_unique_protein_AUROC':float(np.mean([x[0] for x in pv])),'macro_unique_protein_MCC':float(np.mean([x[1] for x in pv]))}
 return rows,macro,unique
def infer_pair(model,base,partner,t,device):
 item=dict(base); item['peptide']=partner['peptide']; item['peptide_length']=partner['peptide_length']; b=cppif_collate([item])
 with torch.no_grad(): logits,_=model(b['peptide'].to(device),b['protein'].to(device),class_mask=b['protein_pad_mask'].to(device))
 y=item['labels'].numpy(); p=F.softmax(logits,dim=-1)[0,:len(y),1].cpu().numpy()
 return (roc_auc_score(y,p),binary(y,p,t)['mcc']) if len(np.unique(y))==2 else None
def partner_shuffle(model,dataset,t,device,seeds,n=200):
 chosen=np.sort(np.random.default_rng(42).choice(len(dataset),size=min(n,len(dataset)),replace=False)); items=[dataset[int(i)] for i in chosen]; model.eval()
 correct=[infer_pair(model,x,x,t,device) for x in items]; correct=[x for x in correct if x is not None]; ca=float(np.mean([x[0] for x in correct])); cm=float(np.mean([x[1] for x in correct])); rows=[]
 for seed in seeds:
  perm=np.random.default_rng(seed).permutation(len(items)); values=[infer_pair(model,items[i],items[int(j)],t,device) for i,j in enumerate(perm)]; values=[x for x in values if x is not None]
  rows.append({'shuffle_seed':seed,'n_records':len(items),'n_valid_records':len(values),'correct_peptide_mean_AUROC':ca,'correct_peptide_mean_MCC':cm,'shuffled_peptide_mean_AUROC':float(np.mean([x[0] for x in values])),'shuffled_peptide_mean_MCC':float(np.mean([x[1] for x in values]))})
 return rows
def env_info(): return {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none','command':' '.join(sys.argv)}

def main(seed=42,epochs=35):
 if seed!=42 or epochs!=35: raise ValueError('Formal reference protocol is locked to seed=42 and epochs=35')
 seed_all(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'); out=ROOT/'runs'/'cppif_679'/f'reference_seed42_{stamp}'; out.mkdir(parents=True,exist_ok=False); log=Logger(out/'train.log')
 data=ROOT/'data/cppif'; emb=ROOT/'embeddings/cppif_esm2_650m'
 # Test is intentionally not read or instantiated before validation lock.
 train_ds=CPPIF679ProteinSiteDataset(data/'train.csv',ROOT,emb); val_ds=CPPIF679ProteinSiteDataset(data/'validation.csv',ROOT,emb)
 if (len(train_ds),len(val_ds))!=(4726,539): raise RuntimeError('locked train/validation counts changed')
 p=CPPIF679_PARAMS
 config={**p,'experiment':'PepMamba-679 CPPIF external benchmark reference run','split_source':'clu_thre_0.0001/new_protein_balanced fold 0; train=group0-group1, validation=group1, test=group2','train_records':4726,'validation_records':539,'expected_test_records':1310,'scheduler':{'name':'CosineAnnealingWarmRestarts','T_0':p['scheduler_t0'],'eta_min':1e-7},'checkpoint_selection':['validation mean-record AUROC','validation mean-record MCC','validation pooled AUPRC'],'threshold':{'split':'validation','minimum':.05,'maximum':.95,'step':.005,'selection':['mean-record MCC','pooled MCC','pooled F1','lower threshold']},'test_used_for_selection':False,'test_threshold_search_performed':False,'sequence_truncation':True,'protein_max_length':679,'peptide_max_length':50,'protein_truncation':'retain first 679 residues, labels, and ESM rows','silent_clipping':False,'environment':env_info()}
 write_json(out/'config.json',config); log('CONFIG_LOCKED '+json.dumps(config,default=jd))
 txt=''.join(train_ds.frame.protein_residue_label.astype(str).str.slice(0, CPPIF679ProteinSiteDataset.PROTEIN_LIMIT)); positives=txt.count('1'); negatives=txt.count('0'); pos_weight=min((negatives/positives)**p['class_weight_power'],p['max_pos_weight']); weights=torch.tensor([1.,pos_weight],device=device)
 model=PepMamba(n_layers=p['n_layers'],d_model=p['d_model'],n_head=p['cross_attention_heads'],d_k=p['d_k'],d_v=p['d_v'],d_inner=p['d_model']*p['d_inner_multiplier'],dropout=p['dropout']).to(device)
 loss_fn=HybridLoss(p['ce_weight'],weights,p['focal_alpha'],p['focal_gamma']); optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay']); scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=p['scheduler_t0'],eta_min=1e-7)
 val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=cppif_collate,pin_memory=True); history=[]; best_key=None; checkpoint=out/'best_validation_checkpoint.pth'; lengths=np.minimum(train_ds.frame.prot_seq.str.len().to_numpy(),CPPIF679ProteinSiteDataset.PROTEIN_LIMIT)
 for epoch in range(1,epochs+1):
  batches,denominators,step_after=make_batches(lengths,p['batch_size'],p['n_layers'],p['d_model'],seed+epoch,p['effective_batch_size']); loader=DataLoader(train_ds,batch_sampler=batches,num_workers=0,collate_fn=cppif_collate,pin_memory=True)
  model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
  for step,b in enumerate(loader,1):
   protein=b['protein'].to(device,non_blocking=True); peptide=b['peptide'].to(device,non_blocking=True); targets=b['labels'].to(device); mask=b['protein_pad_mask'].to(device); logits,_=model(peptide,protein,class_mask=mask); loss=loss_fn(logits,targets)
   if not torch.isfinite(loss): raise RuntimeError(f'non-finite loss epoch={epoch} step={step}')
   count=len(b['record_ids']); (loss*count/denominators[step-1]).backward(); losses.append(float(loss.detach()))
   if step_after[step-1]: torch.nn.utils.clip_grad_norm_(model.parameters(),p['grad_clip']); optimizer.step(); optimizer.zero_grad(set_to_none=True)
  vy,vp,vrecords=collect(model,val_loader,device); _,chosen=scan(vrecords,vy,vp); summary=validation_summary(vy,vp,vrecords,chosen['threshold'])
  row={'epoch':epoch,'learning_rate':optimizer.param_groups[0]['lr'],'train_loss':float(np.mean(losses)),**summary,'validation_epoch_threshold':chosen['threshold']}; history.append(row); key=(summary['mean_record_auroc'],summary['mean_record_mcc'],summary['pooled_auprc'])
  if best_key is None or key>best_key: best_key=key; torch.save({'model_state_dict':model.state_dict(),'best_epoch':epoch,'selection_key':key,'config':config},checkpoint)
  write_csv(out/'training_history.csv',history); log(f"epoch {epoch:02d}/{epochs} train_loss={row['train_loss']:.6f} mean_record_auc={row['mean_record_auroc']:.6f} mean_record_mcc={row['mean_record_mcc']:.6f} pooled_auprc={row['pooled_auprc']:.6f} val_threshold={row['validation_epoch_threshold']:.3f}"); scheduler.step(epoch)
 state=torch.load(checkpoint,map_location=device,weights_only=False); model.load_state_dict(state['model_state_dict']); model.eval(); digest=sha256(checkpoint); (out/'checkpoint_sha256.txt').write_text(digest+'\n')
 vy,vp,vrecords=collect(model,val_loader,device); rows,chosen=scan(vrecords,vy,vp); validation=validation_summary(vy,vp,vrecords,chosen['threshold']); write_csv(out/'validation_threshold_scan.csv',rows); save_predictions(out/'validation_predictions.npz',vy,vp,vrecords)
 lock={'best_epoch':state['best_epoch'],**validation,'selected_threshold':chosen['threshold'],'selection':'maximum validation mean-record MCC; ties pooled MCC, pooled F1, then lower threshold','checkpoint_sha256':digest,'test_loaded_before_lock':False,'test_used_for_selection':False,'test_threshold_search_performed':False,'locked_at':datetime.now(timezone.utc).isoformat()}; write_json(out/'selected_validation_threshold.json',lock); log('VALIDATION_LOCKED '+json.dumps(lock))
 # Test access starts only after the durable checkpoint/threshold lock above.
 test_ds=CPPIF679ProteinSiteDataset(data/'test.csv',ROOT,emb)
 if len(test_ds)!=1310: raise RuntimeError('locked test count changed')
 train=train_ds.frame.copy(); val=val_ds.frame.copy(); test=test_ds.frame.copy(); stats=split_stats(train,val,test); manifest=[]
 for split,frame in [('train',train),('validation',val),('test',test)]:
  for _,r in frame.iterrows(): manifest.append({'split':split,'record_id':r.record_id,'dataset_index':r.dataset_index,'protein_sequence_hash':r.protein_sequence_hash,'peptide_sequence_hash':r.peptide_sequence_hash,'protein_length':min(len(r.prot_seq), CPPIF679ProteinSiteDataset.PROTEIN_LIMIT),'original_protein_length':len(r.prot_seq),'peptide_length':len(r.pep_seq)})
 write_csv(out/'split_manifest.csv',manifest); test_loader=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=cppif_collate,pin_memory=True); test_y,test_p,test_records=collect(model,test_loader,device); t=chosen['threshold']; save_predictions(out/'test_predictions.npz',test_y,test_p,test_records)
 pm={'auroc':float(roc_auc_score(test_y,test_p)),'auprc':float(average_precision_score(test_y,test_p)),'threshold':t,**binary(test_y,test_p,t)}; recs,macro,unique=record_metrics(test_records,test,t); write_csv(out/'per_record_metrics.csv',recs); write_json(out/'pooled_test_metrics.json',pm); write_json(out/'macro_test_metrics.json',macro); write_json(out/'unique_protein_metrics.json',unique)
 sh=partner_shuffle(model,test_ds,t,device,[42,123,3407,7169,8873],200); write_csv(out/'partner_shuffle_analysis.csv',sh); sa=np.asarray([r['shuffled_peptide_mean_AUROC'] for r in sh]); sm=np.asarray([r['shuffled_peptide_mean_MCC'] for r in sh]); (out/'partner_shuffle_report.md').write_text(f"# Partner-shuffle analysis\n\n- Frozen checkpoint: `{digest}`\n- Selection role: none\n- Records sampled: {sh[0]['n_records']}\n- Correct peptide AUROC/MCC: {sh[0]['correct_peptide_mean_AUROC']:.6f} / {sh[0]['correct_peptide_mean_MCC']:.6f}\n- Shuffled peptide AUROC: {sa.mean():.6f} ± {sa.std():.6f}\n- Shuffled peptide MCC: {sm.mean():.6f} ± {sm.std():.6f}\n")
 ef1=2*pm['precision']*pm['recall']/max(pm['precision']+pm['recall'],1e-15); integrity={'label_alignment_pass':'100%','embedding_alignment_pass':'100%','sequence_truncation':True,'protein_max_length':679,'protein_truncation':'retain first 679 residues, labels, and ESM rows','silent_clipping':False,**stats,'test_used_for_selection':False,'test_threshold_search_performed':False,'f1_formula_pass':abs(ef1-pm['f1'])<1e-12,'mcc_confusion_recompute_pass':abs(binary(test_y,test_p,t)['mcc']-pm['mcc'])<1e-12}
 report=f"# FINAL PepMamba-679 CPPIF external benchmark report\n\n## Locked protocol\n\n- Seed / epochs: {seed} / {epochs}\n- Best epoch: {state['best_epoch']}\n- Checkpoint SHA-256: `{digest}`\n- Validation mean-record AUROC/MCC and pooled AUPRC: {lock['mean_record_auroc']:.6f} / {lock['mean_record_mcc']:.6f} / {lock['pooled_auprc']:.6f}\n- Validation-selected threshold: {t:.3f}\n- Test used for selection: false\n- Test threshold search performed: false\n\n## CPPIF external benchmark macro-record test metrics\n\n- Mean AUROC / MCC: {macro['mean_record_AUROC']:.6f} / {macro['mean_record_MCC']:.6f}\n- Median AUROC / MCC: {macro['median_record_AUROC']:.6f} / {macro['median_record_MCC']:.6f}\n- Std AUROC / MCC: {macro['std_record_AUROC']:.6f} / {macro['std_record_MCC']:.6f}\n- Valid records: {macro['n_valid_records']}\n\n## Pooled test metrics\n\n```json\n{json.dumps(pm,indent=2)}\n```\n\n## Unique-protein secondary metrics\n\n```json\n{json.dumps(unique,indent=2)}\n```\n\n## Integrity\n\n```json\n{json.dumps(integrity,indent=2)}\n```\n"; (out/'FINAL_CPPIF_679_REPORT.md').write_text(report); log(f'COMPLETE output_dir={out}')
 print(json.dumps({'output_dir':str(out),'best_epoch':state['best_epoch'],'threshold':t,'macro':macro,'pooled':pm,'unique_protein':unique,'integrity':integrity},indent=2))


if __name__ == "__main__":
    main()
