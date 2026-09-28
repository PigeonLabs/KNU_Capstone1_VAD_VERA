"""Recompute stored follow-up scores/metrics and prove split/fingerprint protection."""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from dataclasses import replace
from sklearn.metrics import roc_auc_score,average_precision_score
from scripts.vera_stage8_common import *
from scripts.vera_stage8_eval import refine,STAGES
from scripts.analyze_vera_stage8 import load

def main(step):
 evaluator_guard();prior=protected();rr=rows();evaluation=[r for r in rr if r['split']=='evaluation'];out=OUT/step;data=load(out/'frame_scores.csv');m=json.loads((out/'metrics.json').read_text());lookup={(r['condition'],r['scene'],r['original_split'],r['video'],r['frame']):r for r in data};assert len(lookup)==len(data);checks=0
 for c,scenes in m.items():
  for sc,stages in scenes.items():
   if sc=='macro':continue
   part=[r for r in data if r['condition']==c and (sc=='pooled' or r['scene']==sc)];y=[r['label'] for r in part]
   for s,metrics in stages.items():
    p=[r[s] for r in part]
    assert abs(roc_auc_score(y,p)*100-metrics['auroc'])<1e-10;assert abs(average_precision_score(y,p)*100-metrics['ap'])<1e-10;checks+=2
  for s in STAGES:
   for k in ['auroc','ap']:assert abs(np.mean([scenes[sc][s][k] for sc in SCENES])-scenes['macro'][s][k])<1e-10
 if step=='step5':
  from scripts.run_vera_stage8_hybrid import dino,vlm
  lock=json.loads((out/'selection_lock.json').read_text());assert set(lock['validation_ids']).isdisjoint(lock['evaluation_ids']);assert lock['selected_lambda']==max(lock['selection'],key=lambda r:(r['macro_auroc'],-r['lambda']))['lambda']
  events=readlines(OUT/'events.jsonl');assert any(r['event']=='step5_selection_locked_before_evaluation_labels' and r['selection_sha256']==sha(out/'selection_lock.json') for r in events)
  normal={r['id'] for r in rr if r['split']=='train' and r['video_label']==0};memorycount=0
  for p in (OUT/'dino').glob('*/memory_inputs.json'):
   d=json.loads(p.read_text());assert set(d['normal_train_ids'])<=normal;assert {v['video_id'] for v in d['selected']}<=normal;memorycount+=len(d['selected'])
  for r in evaluation:
   ns=lock['normalization'][r['scene']];dd=(dino(r)-ns['dino_mean'])/ns['dino_std'];vv=(np.repeat(vlm(r,'evaluation'),16)[:r['length']]-ns['vlm_mean'])/ns['vlm_std'];ss=segments(r['length']);f=features(r)
   for c,x in [('DINO',dd),('HYBRID',dd+lock['selected_lambda']*vv)]:
    score=refine([float(x[s['center']:s['score_end']].mean()) for s in ss],f,r['length']);score['initial']=x
    for s in STAGES:np.testing.assert_allclose([lookup[(c,r['scene'],r['original_split'],r['video'],i)][s] for i in range(r['length'])],score[s],rtol=0,atol=1e-14)
  extra={'memory_train_samples':memorycount,'disjoint_memory':True,'selected_lambda':lock['selected_lambda'],'frame_formula_and_postprocessing_verified':True}
 else:
  inv=json.loads((out/'feature_inventory.json').read_text())
  for record in inv:assert sha(ROOT/record['path'])==record['sha256']
  for r in evaluation:
   for sec in [2,4]:
    config=replace(CONFIG,window_seconds=sec);ss=segments(r['length'],config);fr=readlines(ROOT/'cache/stage8/window_features'/f'{sec}s'/r['scene']/r['original_split']/(r['video']+'.jsonl'));xx=[]
    for s,f in zip(ss,fr):
     assert all(f[k]==v for k,v in s.items());v=json.loads((out/'inference'/f'{sec}s'/r['scene']/r['original_split']/r['video']/f"{s['center']:06d}.json").read_text());assert v['request']['segment']==s
     assert abs(1/(1+np.exp(v['no_logit']-v['yes_logit']))-v['prob'])<1e-14
     if v.get('reused_source'):assert sha(ROOT/v['reused_source'])==v['reused_source_sha256']
     xx.append(v['prob'])
    scores=refine(xx,[x['feature'] for x in fr],r['length'],config)
    for s in STAGES:np.testing.assert_allclose([lookup[(f'PROB_{sec}s',r['scene'],r['original_split'],r['video'],i)][s] for i in range(r['length'])],scores[s],rtol=0,atol=1e-14)
  extra={'feature_window_alignment':True,'all_stages_recomputed':True}
 write(out/'independent_verification.json',{'status':'passed','protected_previous_files':prior,'metric_values_recomputed':checks,'frames':len(data),'time':time.time(),**extra});print('Independent verification passed',step)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',choices=['step4','step5'],required=True);main(p.parse_args().step)
