#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/umppi_protein_site'
EMB=ROOT/'embeddings/umppi_esm2_650m'

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 raw=pd.read_csv(EMB/'manifest.csv')
 manifest=pd.DataFrame({
  'sequence_hash':raw.sequence_sha256,'sequence_type':raw.sequence_type,
  'sequence_length':raw.original_length,'embedding_length':raw.embedding_length,
  'embedding_dim':raw.embedding_dimension,
  'contains_noncanonical':raw[['contains_X','contains_B','contains_U','contains_Z']].any(axis=1),
  'num_windows':raw.window_count,'file_path':raw.file_path,'sha256':raw.file_sha256,
 })
 if manifest.sequence_hash.duplicated().any(): raise RuntimeError('duplicate sequence hash')
 checks=[]
 for row in manifest.itertuples(index=False):
  path=ROOT/row.file_path; value=np.load(path,mmap_mode='r')
  checks.append(value.shape==(row.sequence_length,1280) and digest(path)==row.sha256)
 if not all(checks): raise RuntimeError('manifest file/shape/hash check failed')
 manifest.to_csv(EMB/'manifest.csv',index=False)
 lookup=manifest.set_index('sequence_hash').to_dict('index'); rows=[]
 for split in ('train','validation','test'):
  frame=pd.read_csv(DATA/f'{split}.csv',dtype=str,keep_default_na=False)
  for r in frame.itertuples(index=False):
   pm=lookup[r.protein_sequence_hash]; qm=lookup[r.peptide_sequence_hash]
   plen=len(r.prot_seq); qlen=len(r.pep_seq); llen=len(r.protein_residue_label)
   rows.append({'record_id':r.record_id,'dataset_index':int(r.dataset_index),'split':split,
    'protein_sequence_hash':r.protein_sequence_hash,'peptide_sequence_hash':r.peptide_sequence_hash,
    'protein_sequence_length':plen,'protein_embedding_length':pm['embedding_length'],
    'protein_label_length':llen,'peptide_sequence_length':qlen,'peptide_embedding_length':qm['embedding_length'],
    'protein_embedding_matches_sequence':pm['embedding_length']==plen,
    'protein_label_matches_sequence':llen==plen,'peptide_embedding_matches_sequence':qm['embedding_length']==qlen,
    'all_alignment_checks_pass':pm['embedding_length']==plen and llen==plen and qm['embedding_length']==qlen})
 out=pd.DataFrame(rows); out.to_csv(ROOT/'umppi_embedding_alignment_final.csv',index=False)
 rate=float(out.all_alignment_checks_pass.mean())
 if rate!=1.0: raise RuntimeError(f'alignment pass rate {rate}')
 summary={'records':len(out),'alignment_pass_rate':rate,'manifest_sequences':len(manifest),
  'protein_embeddings':int((manifest.sequence_type=='protein').sum()),'peptide_embeddings':int((manifest.sequence_type=='peptide').sum()),
  'multiwindow_proteins':int(((manifest.sequence_type=='protein')&(manifest.num_windows>1)).sum()),
  'max_protein_length':int(manifest.loc[manifest.sequence_type=='protein','sequence_length'].max()),
  'noncanonical_sequences':int(manifest.contains_noncanonical.sum()),'embedding_dim':1280,'storage_dtype':'float16'}
 report=f'''# UMPPI ESM-2 embedding generation report\n\n- Model: ESM-2 t33 650M, layer 33, 1280 dimensions.\n- Eligible records checked: **{len(out):,}**.\n- Unique protein embeddings: **{summary['protein_embeddings']:,}**.\n- Unique peptide embeddings: **{summary['peptide_embeddings']:,}**.\n- Multi-window proteins: **{summary['multiwindow_proteins']:,}**.\n- Maximum protein length: **{summary['max_protein_length']:,}**.\n- Window/overlap: 1022/128; overlap merge: arithmetic mean.\n- X/B/U/Z: preserved as native ESM tokens; no residue removed.\n- Alignment pass rate: **{rate:.1%}**.\n- Silent truncation/clipping: **not used**.\n- Training performed: **false**.\n\nEvery manifest file was reopened with mmap and checked for `(sequence_length, 1280)` shape and SHA256 integrity. Every record passed protein-embedding, protein-label, and peptide-embedding length checks.\n'''
 (ROOT/'umppi_embedding_generation_report.md').write_text(report)
 (ROOT/'umppi_embedding_generation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
