"""Independent artifact verification, recomputing sklearn metrics and raw logits."""
import argparse,csv,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from scripts.vera_stage8_common import OUT,ROOT,sha,write,protected,evaluator_guard

def main(step):
 evaluator_guard();old=protected();base=OUT/step
 files=list(csv.DictReader((base/'frame_scores.csv').open()));metrics=json.loads((base/'metrics.json').read_text())
 keys=[(r['condition'],r['scene'],r['original_split'],r['video'],r['frame']) for r in files];assert len(keys)==len(set(keys))
 checks=0
 for condition,d in metrics.items():
  for scene,m in d.items():
   if scene=='macro':continue
   part=[r for r in files if r['condition']==condition and (scene=='pooled' or r['scene']==scene)];y=[int(r['label']) for r in part]
   for stage,v in m.items():
    p=[float(r[stage]) for r in part]
    for k,f in [('auroc',roc_auc_score),('ap',average_precision_score)]:assert abs(f(y,p)*100-v[k])<1e-10;checks+=1
  for stage in d['macro']:
   for k in ['auroc','ap']:assert abs(np.mean([d[s][stage][k] for s in ['R01','R02','R03','R04']])-d['macro'][stage][k])<1e-10
 result={'status':'passed','protected_previous_files':old,'metric_values_recomputed':checks,'unique_frame_rows':len(keys),'evaluation_code_sha256':sha(ROOT/'scripts/vera_stage8_eval.py'),'time':time.time()}
 if step=='step1':
  frozen=json.loads((base/'frozen.json').read_text())
  for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h
  assert hashlib.sha256(frozen['prompt'].encode()).hexdigest()==frozen['prompt_sha256']
  count=0;image_hashes={};lookup={(r['scene'],r['original_split'],r['video'],int(r['frame'])):r for r in files if r['condition']=='PROB'}
  for path in (base/'inference').rglob('*.json'):
   r=json.loads(path.read_text());q=r['request'];assert q['fingerprint']==frozen['fingerprint'];assert r['prompt']==frozen['prompt']
   expected=1/(1+np.exp(r['no_logit']-r['yes_logit']));assert abs(expected-r['prob'])<1e-14
   assert r['generated_token_ids']==[r['vocabulary_argmax']]
   assert r['dec']==({'Yes':1,'No':0}.get(r['response'].strip()))
   sc,split,vid=q['video_id'].split('/');seg=q['segment']
   for frame in range(seg['center'],seg['score_end']):assert float(lookup[(sc,split,vid,frame)]['initial'])==r['prob']
   for im in q['images']:image_hashes[(im['relative_path'],im['frame_id'])]=im['sha256']
   count+=1
  from scripts.vera_stage8_common import DATA
  for (rel,fid),h in image_hashes.items():
   p=DATA/rel/f'{fid:03d}.jpg'
   if not p.exists():p=next(x for x in (DATA/rel).glob('*.jpg') if int(x.stem)==fid)
   assert sha(p)==h
  result.update(raw_calls_verified=count,unique_raw_images_verified=len(image_hashes),no_synthetic_decisions=True)
 for p in (base/'bootstrap').glob('*.json'):
  b=json.loads(p.read_text());assert b['n']==2000 and len(b['resample_counts'])==2000 and b['valid']+b['undefined_single_class']==2000
 write(base/'independent_verification.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',choices=['step0','step1'],required=True);main(p.parse_args().step)
