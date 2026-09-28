import json,sys,time,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import OUT,write,sha

def main():
 allrecords=[]
 for dirname in ['step1','step4','step5']:
  entries=[]
  for p in (OUT/dirname).rglob('*.json'):
   r=json.loads(p.read_text())
   if isinstance(r,dict) and 'response' in r and 'seconds' in r:
    entries.append({'path':str(p.relative_to(ROOT)),'seconds':r['seconds'],'peak_allocated_gib':r.get('peak_allocated_gib'),'peak_reserved_gib':r.get('peak_reserved_gib'),'reused':bool(r.get('reused_source')),'prompt_sha256':hashlib.sha256(r['prompt'].encode()).hexdigest(),'response':r['response'],'prob':r.get('prob'),'dec':r.get('dec')})
  if entries:
   write(OUT/dirname/'runtime_summary.json',{'calls':len(entries),'reused_calls':sum(r['reused'] for r in entries),'inference_seconds':sum(r['seconds'] for r in entries),'peak_allocated_gib':max((r['peak_allocated_gib'] or 0) for r in entries),'records':entries});allrecords.extend(entries)
 extra=[]
 for p in (OUT/'diagnostics/unseeded_window_attempt/inference').rglob('*.json'):
  r=json.loads(p.read_text());extra.append({'category':'interrupted_window','seconds':r['seconds']})
 for name in ['validation_replay.jsonl','explanation_replay.jsonl']:
  p=OUT/'seed0_verification'/name
  if p.exists():extra.extend({'category':'seed0_replay','seconds':json.loads(line)['seconds']} for line in p.read_text().splitlines() if line.strip())
 write(OUT/'runtime_summary.json',{'canonical_recorded_vlm_calls':len(allrecords),'canonical_model_seconds':sum(r['seconds'] for r in allrecords),'additional_diagnostic_calls':len(extra),'additional_model_seconds':sum(r['seconds'] for r in extra),'total_logged_vlm_model_seconds':sum(r['seconds'] for r in allrecords+extra),'includes_training_preflight_validation_explanations':True,'not_end_to_end_wall_time':'Command logs separately contain elapsed wall time, loading, statistics and feature extraction.'})
if __name__=='__main__':main()
