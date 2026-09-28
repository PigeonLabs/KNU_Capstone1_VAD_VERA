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
 write(OUT/'runtime_summary.json',{'recorded_vlm_calls':len(allrecords),'inference_seconds':sum(r['seconds'] for r in allrecords),'includes_training_preflight_validation_explanations':True})
if __name__=='__main__':main()
