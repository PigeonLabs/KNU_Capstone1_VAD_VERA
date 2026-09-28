"""Final publication gate: complete actual results, immutable evaluators, safe files."""
import json,sys,csv,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import OUT,sha,write,protected,evaluator_guard
from scripts.publish_vera_stage import publication_files

def main():
 evaluator_guard();prior=protected();assert json.loads((OUT/'seed0_verification/status.json').read_text())['status']=='complete'
 for step in ['step0','step1','step4','step5']:
  s=json.loads((OUT/step/'status.json').read_text());assert s['status'] in ['complete','diagnostic']
  v=json.loads((OUT/step/'independent_verification.json').read_text());assert v['status']=='passed'
  rr=list(csv.DictReader((OUT/step/'frame_scores.csv').open()))
  for condition in {r['condition'] for r in rr}:
   p=[r for r in rr if r['condition']==condition];assert len(p)==16862 and len({(r['scene'],r['original_split'],r['video']) for r in p})==37
 for step in ['step1']:
  frozen=json.loads((OUT/step/'frozen.json').read_text())
  for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h
 for step,script in [('step4','scripts/run_vera_stage8_windows.py'),('step5','scripts/run_vera_stage8_hybrid.py'),('dino','scripts/run_vera_stage8_dino.py')]:assert sha(ROOT/script)==json.loads((OUT/step/'frozen.json').read_text())['source_sha256']
 assert json.loads((OUT/'step4/frozen.json').read_text())['torch_initial_seed']==0
 assert json.loads((OUT/'step5/explanations/verification.json').read_text())['calls']==37
 inventory=publication_files();unsafe=[r['path'] for r in inventory if any(part in Path(r['path']).parts for part in ['IPAD_dataset','cache','.venv'])];assert not unsafe
 write(OUT/'publication_verification.json',{'status':'passed','protected_prior_files':prior,'audited_files':len(inventory),'bytes':sum(r['bytes'] for r in inventory),'raw_images_weights_features_published':False,'all_37_video_supports_complete':True,'source_fingerprints_valid':True,'evaluation_code_unchanged':True,'seed0_replay_verified':True,'time':time.time()});print('Stage8 publication verification passed',len(inventory),'files')
if __name__=='__main__':main()
