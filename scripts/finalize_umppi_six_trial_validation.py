#!/usr/bin/env python3
"""Wait for six validation-only trials, summarize, lock one candidate, and stop without held-out evaluation."""
from __future__ import annotations
import argparse,json,math,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from tune_umppi_pepmamba import pareto,summarize_validation,threshold_metrics
from umppi_reference_run import sha256,write_csv,write_json
REFERENCE=ROOT/'tuning_results/umppi_reference_seed42_20260808_082256'

def reference_validation():
 checkpoint=REFERENCE/'best_validation_checkpoint.pth'; expected=(REFERENCE/'checkpoint_sha256.txt').read_text().strip(); actual=sha256(checkpoint)
 if actual!=expected: raise RuntimeError('sealed reference checkpoint SHA-256 mismatch')
 z=np.load(REFERENCE/'validation_predictions.npz'); y=z['labels']; p=z['probabilities']; ids=z['record_ids']; offsets=z['record_offsets']; records=[]
 for i,rid in enumerate(ids): records.append({'record_id':str(rid),'y':y[offsets[i]:offsets[i+1]],'p':p[offsets[i]:offsets[i+1]]})
 rows,chosen=threshold_metrics(records,y,p); summary=summarize_validation(y,p,records,chosen)
 return {'reference_checkpoint':str(checkpoint),'reference_checkpoint_sha256':actual,'reference_best_epoch':34,**summary,'held_out_split_loaded':False}

def load_six(run_dir):
 partial=run_dir/'all_trials.partial.csv'
 if not partial.exists(): return None
 df=pd.read_csv(partial)
 if len(df)<6: return None
 df=df[df.trial_id.isin([f'trial_{i:03d}' for i in range(6)])].copy()
 if len(df)!=6: raise RuntimeError('expected exactly trial_000 through trial_005')
 return df

def finalize(run_dir):
 marker=run_dir/'SIX_TRIAL_SELECTION_REPORT.md'
 if marker.exists(): return
 df=load_six(run_dir)
 if df is None: raise RuntimeError('six completed trials are not available')
 ref=reference_validation(); write_json(run_dir/'reference_validation_metrics.json',ref)
 df['delta_mean_record_AUROC_vs_reference']=df.validation_mean_record_AUROC-ref['validation_mean_record_AUROC']
 df['delta_mean_record_MCC_vs_reference']=df.validation_mean_record_MCC-ref['validation_mean_record_MCC']
 df['delta_pooled_AUPRC_vs_reference']=df.validation_pooled_AUPRC-ref['validation_pooled_AUPRC']
 configs=[]
 for trial_id in df.trial_id:
  cfg=json.loads((run_dir/trial_id/'config.json').read_text()); configs.append(json.dumps(cfg,sort_keys=True,separators=(',',':')))
 df['config']=configs
 cols=['trial_id','best_epoch','validation_mean_record_AUROC','validation_mean_record_MCC','validation_pooled_AUROC','validation_pooled_AUPRC','validation_pooled_MCC','validation_F1','selected_threshold','checkpoint_sha256','checkpoint_path','delta_mean_record_AUROC_vs_reference','delta_mean_record_MCC_vs_reference','delta_pooled_AUPRC_vs_reference','config']
 summary=df[cols].copy(); summary.to_csv(run_dir/'six_trial_validation_summary.csv',index=False)
 ranking=summary.sort_values(['validation_mean_record_MCC','validation_mean_record_AUROC','validation_pooled_AUPRC'],ascending=False).reset_index(drop=True); ranking.insert(0,'validation_rank',np.arange(1,7)); ranking.to_csv(run_dir/'six_trial_validation_ranking.csv',index=False)
 front=pd.DataFrame(pareto(summary.to_dict('records'))); front[cols].sort_values(['validation_mean_record_AUROC','validation_mean_record_MCC'],ascending=False).to_csv(run_dir/'six_trial_pareto_front.csv',index=False)
 selected=ranking.iloc[0].to_dict(); cfg=json.loads((run_dir/selected['trial_id']/'config.json').read_text()); locked=datetime.now(timezone.utc).isoformat()
 lines=['# SIX-TRIAL VALIDATION SELECTION REPORT','',f'- Completed trials: 6 (`trial_000`–`trial_005`)','- Maximum trials: 6','- Ranking: validation mean-record MCC descending; mean-record AUROC descending; pooled AUPRC descending.','- Held-out split loaded: false','- Held-out evaluation during tuning: false','', '## Sealed reference validation metrics','',f"- Mean-record AUROC: {ref['validation_mean_record_AUROC']:.9f}",f"- Mean-record MCC: {ref['validation_mean_record_MCC']:.9f}",f"- Pooled AUPRC: {ref['validation_pooled_AUPRC']:.9f}",'','## Six-trial ranking','', '| Rank | Trial | Best epoch | Mean-record AUROC | Mean-record MCC | Pooled AUPRC | Threshold |','|---:|---|---:|---:|---:|---:|---:|']
 for _,r in ranking.iterrows(): lines.append(f"| {int(r.validation_rank)} | {r.trial_id} | {int(r.best_epoch)} | {r.validation_mean_record_AUROC:.6f} | {r.validation_mean_record_MCC:.6f} | {r.validation_pooled_AUPRC:.6f} | {r.selected_threshold:.3f} |")
 lines += ['',f"Selected candidate: `{selected['trial_id']}`",'', '`test_used_for_selection = false`  ', '`test_threshold_search_performed = false`  ', '`test_evaluated_during_tuning = false`','']
 marker.write_text('\n'.join(lines))
 lock=f"""# FINAL SELECTION BEFORE TEST

- selected_trial: `{selected['trial_id']}`
- selection_rule: validation mean-record MCC descending; then mean-record AUROC; then pooled AUPRC
- locked_at: {locked}
- best_epoch: {int(selected['best_epoch'])}
- validation mean_record_AUROC: {selected['validation_mean_record_AUROC']:.9f}
- validation mean_record_MCC: {selected['validation_mean_record_MCC']:.9f}
- validation pooled_AUPRC: {selected['validation_pooled_AUPRC']:.9f}
- validation threshold: {selected['selected_threshold']:.3f}
- checkpoint path: `{selected['checkpoint_path']}`
- checkpoint SHA256: `{selected['checkpoint_sha256']}`
- test_used_for_selection = false
- test_threshold_search_performed = false
- test_evaluated_during_tuning = false

## Full config

```json
{json.dumps(cfg,indent=2)}
```

The process stops here pending explicit human confirmation. No held-out evaluation was started.
"""; (run_dir/'FINAL_SELECTION_BEFORE_TEST.md').write_text(lock)
 write_json(run_dir/'six_trial_tuning_complete.json',{'MAX_TRIALS':6,'completed_trials':[f'trial_{i:03d}' for i in range(6)],'selected_trial':selected['trial_id'],'locked_at':locked,'test_used_for_selection':False,'test_threshold_search_performed':False,'test_evaluated_during_tuning':False})

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--watch-pid',type=int); ap.add_argument('--check-only',action='store_true'); args=ap.parse_args(); ref=reference_validation(); print(json.dumps(ref,indent=2))
 if args.check_only:return
 while load_six(args.run_dir) is None:
  if args.watch_pid:
   try: os.kill(args.watch_pid,0)
   except ProcessLookupError: raise RuntimeError('tuner exited before six trials completed')
  time.sleep(30)
 if args.watch_pid:
  for _ in range(20):
   try: os.kill(args.watch_pid,0)
   except ProcessLookupError: break
   time.sleep(.25)
  else: raise RuntimeError('budget gate did not stop original tuner after trial_005')
 gate=args.run_dir/'trial_006'
 if not (gate/'BUDGET_STOP_GATE.json').exists(): raise RuntimeError('budget gate marker missing')
 gate.rename(args.run_dir/'BUDGET_STOP_GATE')
 finalize(args.run_dir)
if __name__=='__main__':main()
