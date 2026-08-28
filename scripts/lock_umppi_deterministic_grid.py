#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'tuning_results/umppi_validation_tuning_20260808_125913'
CONTINUOUS={
 'learning_rate':('log',6e-5,2.5e-4),'dropout':('linear',.15,.35),'focal_alpha':('linear',.55,.85),
 'focal_gamma':('linear',1.2,3.2),'ce_weight':('linear',.40,.80),'weight_decay':('log',3e-5,3e-3),
 'class_weight_power':('linear',.25,.75),'max_pos_weight':('linear',4.,10.)}
PATTERNS=[[0,2,4,1,3],[3,0,2,4,1],[1,4,0,3,2],[4,1,3,0,2],[2,4,1,3,0]]
DISCRETE={
 'batch_size':[8,12,16,4,16],
 'n_layers':[2,3,4,2,3],
 'd_model':[128,96,160,96,128],
 'cross_attention_heads':[8,4,8,4,8]}

def levels(kind,lo,hi): return np.geomspace(lo,hi,6) if kind=='log' else np.linspace(lo,hi,6)
def main():
 out=RUN/'deterministic_grid_lock'; out.mkdir(parents=True,exist_ok=False)
 base=json.loads((RUN/'trial_000/config.json').read_text()); configs=[dict(base)]; level_rows=[]
 trial_cfg={i:{'trial_id':f'trial_{i:03d}','seed':42,'epochs':35} for i in range(1,6)}
 for j,(name,(kind,lo,hi)) in enumerate(CONTINUOUS.items()):
  vals=levels(kind,lo,hi); distance=np.abs(np.log(vals/base[name])) if kind=='log' else np.abs(vals-base[name]); base_level=int(distance.argmin())
  remaining=[i for i in range(6) if i!=base_level]; pattern=PATTERNS[j%len(PATTERNS)]; assigned=[remaining[k] for k in pattern]
  level_rows.append({'parameter':name,'scale':kind,'levels':json.dumps([float(x) for x in vals]),'trial_000_actual':base[name],'trial_000_nearest_level':base_level,**{f'trial_{i:03d}_level':assigned[i-1] for i in range(1,6)}})
  for i in range(1,6): trial_cfg[i][name]=float(vals[assigned[i-1]])
 for name,values in DISCRETE.items():
  for i in range(1,6): trial_cfg[i][name]=values[i-1]
 for i in range(1,6):
  c=trial_cfg[i]; c.update({'d_k':32,'d_v':32,'d_inner_multiplier':2,'effective_batch_size':16 if c['batch_size']==16 else 24,'scheduler':{'name':'CosineAnnealingWarmRestarts','T_0':10,'eta_min':1e-7},'grad_clip':1.0,'parameter_source':'locked deterministic six-level crossed coverage'})
  configs.append(c)
 payload={'locked_at':datetime.now(timezone.utc).isoformat(),'protocol':'deterministic six-level crossed coverage; trial_000 actual values anchor nearest levels; trial_001-005 receive each remaining level exactly once per continuous parameter','seed':42,'epochs':35,'scheduled_trials':['trial_001','trial_002','trial_003','trial_004'],'held_trial':'trial_005','test_access_allowed':False,'configs':configs}
 canonical=json.dumps(payload,sort_keys=True,separators=(',',':')).encode(); payload['config_payload_sha256']=hashlib.sha256(canonical).hexdigest()
 (out/'trial_000_005_locked_configs.json').write_text(json.dumps(payload,indent=2)+'\n'); pd.DataFrame(configs).to_csv(out/'trial_000_005_locked_parameter_table.csv',index=False); pd.DataFrame(level_rows).to_csv(out/'continuous_level_assignment.csv',index=False)
 readme=f"# UMPPI deterministic grid lock\n\n- Locked: {payload['locked_at']}\n- Payload SHA-256: `{payload['config_payload_sha256']}`\n- Scheduled batch: trial_001–trial_004\n- Held: trial_005\n- Test access: false\n"; (out/'GRID_LOCK.md').write_text(readme)
 print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
