"""Stage5 label isolation, frozen cache identity and unchanged baseline support."""
import csv,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import scripts.run_vera_stage5 as run
from ipad.vera import segments,refine_scores

def synthetic(monkeypatch,tmp_path,invalid=False):
 root=tmp_path;out=root/'experiments/stage5';out.mkdir(parents=True)
 rows=[dict(id=f'{s}/testing/01',scene=s,original_split='testing',video='01',length=17,relative_path='unused',frames_sha256=['hash']*17,frame_ids=list(range(17))) for s in run.SCENES]
 prompts={s:{c:f'{s} {c} '+('<image>\n'*8) for c in run.CONDITIONS} for s in run.SCENES}
 frozen={'fingerprint':'frozen'}
 run.write(out/'preflight_summary.json',{'status':'complete'})
 monkeypatch.setattr(run,'ROOT',root);monkeypatch.setattr(run,'OUT',out)
 monkeypatch.setattr(run,'setup',lambda:(rows,rows,[],prompts,frozen));monkeypatch.setattr(run,'protected',lambda:99)
 calls=[];labels=[]
 class Fake:
  def __init__(self,_):self.adapter=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text,**kwargs:text.split()))
  def recorded(self,path,prompt,row,seg):
   if path.exists():return json.loads(path.read_text())
   response='Output: 1' if seg['center']==16 else 'Output: 0'
   if invalid and row['scene']=='R01' and ' P1 ' in prompt:response='uncertain'
   value={'response':response,'seconds':.1,'peak_allocated_gib':1.,'peak_reserved_gib':1.,'prompt':prompt,'segment':seg,'video_id':row['id'],'request_hash':run.digest({'prompt':prompt,'video_id':row['id'],'segment':seg})}
   calls.append(value);run.append(out/'calls.jsonl',value);run.write(path,value);return value
 monkeypatch.setattr(run,'Engine',Fake)
 def load(row,allrows):
  assert len(list((out/'inference').rglob('*.json')))==24
  labels.append(row['id']);return np.array([0]*16+[1])
 monkeypatch.setattr(run,'load_labels',load)
 inventory=[]
 for row in rows:
  scene=row['scene'];feature=root/f'cache/stage3/features/{scene}/testing/01.jsonl'
  for seg in segments(17):run.append(feature,{**seg,'feature':[1.,1.]})
  inventory.append({'path':str(feature.relative_to(root)),'local_only':True,'sha256':run.sha(feature)})
  baseline=root/f'experiments/stage4/{scene}/evaluation/frame_scores.csv';baseline.parent.mkdir(parents=True)
  with baseline.open('w') as f:
   writer=csv.DictWriter(f,fieldnames=['condition','original_split','video','frame','label','initial','retrieved','smoothed','final']);writer.writeheader()
   for c in ['A','B','C']:
    for i in range(17):writer.writerow(dict(condition=c,original_split='testing',video='01',frame=i,label=int(i==16),initial=0,retrieved=0,smoothed=0,final=0))
 run.write(root/'experiments/stage3/inference_inventory.json',inventory)
 return out,calls,labels,rows,prompts

def test_all_four_scenes_inferred_before_labels_and_exact_support(monkeypatch,tmp_path):
 out,calls,labels,rows,prompts=synthetic(monkeypatch,tmp_path)
 run.main('evaluation')
 assert len(calls)==24 and len(labels)==4
 status=json.loads((out/'status.json').read_text());assert status['status']=='complete' and status['frames_per_condition']==68
 result=list(csv.DictReader((out/'frame_scores.csv').open()));assert len(result)==6*68
 metrics=json.loads((out/'metrics.json').read_text());assert metrics['R01']['P1']['initial_binary']['tp']==1
 assert metrics['R01']['A']['initial_binary']['tp']==0
 events=run.readlines(out/'events.jsonl');assert events[0]['event']=='all_evaluation_inference_validated_before_labels'
 assert all(e['event']=='evaluation_labels_open' for e in events[1:])

def test_invalid_output_preserved_without_imputation_or_label_access(monkeypatch,tmp_path):
 out,calls,labels,*_=synthetic(monkeypatch,tmp_path,invalid=True)
 with pytest.raises(RuntimeError,match='invalid model outputs'):run.main('evaluation')
 assert not labels and len(calls)==24
 assert not (out/'frame_scores.csv').exists()
 invalid=run.readlines(out/'parsing_failures.jsonl');assert invalid and all(r['prediction'] is None and r['parse_status']=='failed' for r in invalid)

def test_cache_cannot_be_relabelled_with_new_fingerprint(monkeypatch,tmp_path):
 out,calls,labels,rows,prompts=synthetic(monkeypatch,tmp_path)
 engine=run.Engine(out);row=rows[0];seg=segments(17)[0];path=out/'cached.json'
 run.predict(engine,path,prompts['R01']['P1'],row,seg,'original')
 with pytest.raises(AssertionError,match='cache identity'):run.predict(engine,path,prompts['R01']['P1'],row,seg,'different')
 assert len(calls)==1

def test_current_stage4_protection():
 assert run.protected()==6837

def test_frozen_prompt_conditions_and_scene_specificity():
 import scripts.prepare_vera_stage5 as prepare
 prompts=json.loads((run.OUT/'prompts.json').read_text())
 assert set(prompts)==set(run.SCENES)
 for scene in run.SCENES:
  assert set(prompts[scene])==set(run.CONDITIONS)
  for c,p in prompts[scene].items():
   assert p.count('<image>')==8 and prepare.NEUTRAL[scene] in p
   assert 'Decision rule for every condition:' in p
   assert 'suspicious person' not in p
   assert ('Partial description from normal training observations' in p)==(c!='P1')
  description=(run.ROOT/f'experiments/stage4/{scene}/normal/normal_description.txt').read_text()
  assert description in prompts[scene]['P2'] and description in prompts[scene]['P3']
  common=lambda p:p.split('Decision rule for every condition:')[1].split('\n')[0]
  assert common(prompts[scene]['P1'])==common(prompts[scene]['P2'])==common(prompts[scene]['P3'])
 for c in run.CONDITIONS:assert len({prompts[s][c] for s in run.SCENES})==4

def test_label_timeline_allows_cached_recompute_but_never_early_labels():
 from scripts.audit_vera_stage5 import verify_label_timeline
 gate=lambda t:dict(event='all_evaluation_inference_validated_before_labels',time=t)
 label=lambda t,v:dict(event='evaluation_labels_open',time=t,video_id=v)
 events=[gate(3),label(4,'a'),label(5,'b'),gate(6),label(7,'a'),label(8,'b')]
 assert verify_label_timeline(events,[1,2],['a','b'])==2
 with pytest.raises(AssertionError):verify_label_timeline([label(.5,'a')]+events,[1,2],['a','b'])
 with pytest.raises(AssertionError):verify_label_timeline(events[:-1],[1,2],['a','b'])
 with pytest.raises(AssertionError):verify_label_timeline(events,[9],['a','b'])
