#!/usr/bin/env python3
"""Run only locked deterministic UMPPI trial_001 through trial_004; validation-only."""
from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from tune_umppi_pepmamba import EMBEDDING_DIR,TRAIN_FILE,VALIDATION_FILE,train_trial
from umppi_dataset import UMppiProteinSiteDataset
from umppi_reference_run import write_csv,write_json
RUN=ROOT/'tuning_results/umppi_validation_tuning_20260808_125913'
LOCK=RUN/'deterministic_grid_lock/trial_000_005_locked_configs.json'
SCHEDULED=['trial_001','trial_002','trial_003','trial_004']

def verify_lock(payload):
 expected=payload['config_payload_sha256']; bare=dict(payload); bare.pop('config_payload_sha256')
 actual=hashlib.sha256(json.dumps(bare,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if actual!=expected: raise RuntimeError('locked configuration payload SHA mismatch')
 if payload['scheduled_trials']!=SCHEDULED or payload['held_trial']!='trial_005': raise RuntimeError('locked run budget changed')

def trial0_row():
 df=pd.read_csv(RUN/'all_trials.partial.csv'); row=df[df.trial_id=='trial_000']
 if len(row)!=1: raise RuntimeError('completed trial_000 row missing or duplicated')
 return row.iloc[0].to_dict()

def outputs(rows,payload):
 all_rows=[trial0_row(),*rows]; pd.DataFrame(all_rows).to_csv(RUN/'deterministic_trial_000_004_validation_comparison.csv',index=False)
 configs=pd.DataFrame(payload['configs']); configs[configs.trial_id.isin(['trial_000',*SCHEDULED])].to_csv(RUN/'deterministic_trial_000_004_parameter_table.csv',index=False)
 cols=['trial_id','best_epoch','validation_mean_record_AUROC','validation_mean_record_MCC','validation_pooled_AUROC','validation_pooled_AUPRC','validation_pooled_MCC','validation_F1','selected_threshold','checkpoint_sha256']
 ranking=pd.DataFrame(all_rows)[cols].sort_values(['validation_mean_record_MCC','validation_mean_record_AUROC','validation_pooled_AUPRC'],ascending=False); ranking.insert(0,'validation_rank',range(1,len(ranking)+1)); ranking.to_csv(RUN/'deterministic_trial_000_004_validation_ranking.csv',index=False)
 lines=['# UMPPI deterministic trial_000–004 validation report','',f'- Completed at: {datetime.now(timezone.utc).isoformat()}','- Completed scheduled batch: trial_001–trial_004','- trial_005 started: false','- Held-out dataset loaded by this runner: false','- Held-out metrics produced by this runner: false','- Checkpoint selection: validation mean-record AUROC; then mean-record MCC; then pooled AUPRC','- Threshold selection: validation mean-record MCC; ties pooled MCC, F1, lower threshold','','## Validation comparison','', '| Rank | Trial | Best epoch | Mean-record AUROC | Mean-record MCC | Pooled AUPRC | Threshold |','|---:|---|---:|---:|---:|---:|---:|']
 for _,r in ranking.iterrows(): lines.append(f"| {int(r.validation_rank)} | {r.trial_id} | {int(r.best_epoch)} | {r.validation_mean_record_AUROC:.6f} | {r.validation_mean_record_MCC:.6f} | {r.validation_pooled_AUPRC:.6f} | {r.selected_threshold:.3f} |")
 lines += ['','No final candidate is selected here because trial_005 is explicitly held outside this scheduled batch.','Historical note: trial_000 had already been evaluated on the held-out set before this new validation-only continuation; that prior access is preserved in the audit trail and is not used by this runner.','']
 (RUN/'DETERMINISTIC_TRIAL_000_004_REPORT.md').write_text('\n'.join(lines)); write_json(RUN/'deterministic_batch_complete.json',{'completed_trials':SCHEDULED,'trial_005_started':False,'stopped_after':'trial_004','test_loaded_by_runner':False,'test_metrics_used_by_runner':False})

def main():
 payload=json.loads(LOCK.read_text()); verify_lock(payload); by_id={c['trial_id']:c for c in payload['configs']}
 for trial_id in SCHEDULED:
  if (RUN/trial_id).exists(): raise FileExistsError(f'refusing to overwrite {RUN/trial_id}')
 train=UMppiProteinSiteDataset(TRAIN_FILE,ROOT,EMBEDDING_DIR); val=UMppiProteinSiteDataset(VALIDATION_FILE,ROOT,EMBEDDING_DIR)
 if (len(train),len(val))!=(4726,539): raise RuntimeError('locked train/validation counts changed')
 rows=[]; global_log=RUN/'deterministic_tuning_001_004.log'; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 for trial_id in SCHEDULED:
  rows.append(train_trial(dict(by_id[trial_id]),train,val,RUN,device,global_log)); pd.DataFrame([trial0_row(),*rows]).to_csv(RUN/'deterministic_trial_000_004.partial.csv',index=False)
 outputs(rows,payload); print(json.dumps({'status':'complete','completed_trials':SCHEDULED,'trial_005_started':False,'test_loaded':False},indent=2))
if __name__=='__main__':main()
