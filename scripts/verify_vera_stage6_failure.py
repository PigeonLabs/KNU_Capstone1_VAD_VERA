"""Verify the actual fail-stop Stage6 run without opening evaluation labels."""
import json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage6_common import OUT,sha,digest,protected,input_identity
from scripts.vera_stage4_common import readlines,write
from scripts.train_vera_questions import parse_response

def main():
 f=json.loads((OUT/'frozen.json').read_text());assert f['fingerprint']==digest({k:v for k,v in f.items() if k!='fingerprint'})
 for p,h in f['source_sha256'].items():assert sha(ROOT/p)==h,p
 assert json.loads((OUT/'status.json').read_text())['status']=='failed'
 pre=[json.loads(p.read_text()) for p in (OUT/'preflight').rglob('*.json')];ev=[json.loads(p.read_text()) for p in (OUT/'inference').rglob('*.json')]
 assert len(pre)==60 and len(ev)==78
 assert all(r['parse_status']=='valid' and parse_response(r['response'])==r['prediction'] for r in pre)
 bad=[r for r in ev if r['parse_status']!='valid'];assert len(bad)==1
 assert all(parse_response(r['response'])==r['prediction'] for r in ev if r['parse_status']=='valid')
 r=bad[0];assert r['prediction'] is None and 'Output' not in r['response'] and r['response_retokenized_tokens']==1024
 assert r['request']['video_id']=='R01/training/27' and r['request']['segment']['center']==64
 assert all(r['request']['condition']=='C0' and r['request']['video_id'].startswith('R01/') for r in ev)
 calls=readlines(OUT/'calls.jsonl');assert len(calls)==138 and {r['cache_key'] for r in calls}=={r['cache_key'] for r in pre+ev}
 for r in calls:assert r['cache_key']==digest(r['request']) and r['request']['fingerprint']==f['fingerprint']
 assert len(readlines(OUT/'parsing_failures.jsonl'))==1
 events=readlines(OUT/'events.jsonl');assert [r['event'] for r in events]==['invocation','model_loaded','invocation','model_loaded']
 for name in ['metrics.json','frame_scores.csv','window_diagnostics.json','postprocessing']:
  assert not (OUT/name).exists(),name
 log=next((OUT/'execution/regression_final').glob('*/output.log')).read_text();assert '106 passed' in log
 old=subprocess.check_output(['git','show','9c351438e36b997af7d89664bb2bac00ed18e8b2:README.md'],text=True)
 now=(ROOT/'README.md').read_text();assert now.split('## 1단계 — 개요와 진행 방법')[1].split('\n## 6단계')[0].rstrip()==old.split('## 1단계 — 개요와 진행 방법')[1].rstrip()
 result={'status':'passed','experiment_status':'failed','preflight_valid':60,'evaluation_valid':77,'evaluation_failed':1,'evaluation_expected':3216,'total_model_calls':138,'failure_video':r['request']['video_id'] if False else bad[0]['request']['video_id'],'failure_center':64,'response_retokenized_tokens':1024,'actual_generated_tokens_and_stop_reason':'not exposed by frozen runner','evaluation_label_gate_reached':False,'metrics_computed':False,'evidence_for_no_label_access':'frozen source control flow, event sequence, absent downstream outputs','no_repair_retry_or_imputation':True,'protected_stage4_stage5_files':protected(),'repository_tests_passed':106,'source_hashes_verified':len(f['source_sha256']),'fingerprint':f['fingerprint'],'model_seconds':sum(r['seconds'] for r in calls),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in calls)}
 write(OUT/'failure_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
