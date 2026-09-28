"""Stage6 reference provenance, paired inputs, failure and label-boundary regression."""
import csv,json
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import scripts.vera_stage6_common as common
import scripts.run_vera_stage6 as run

def training_rows():
 return [dict(id=f'{s}/{split}/{v:02d}',scene=s,original_split=split,video=f'{v:02d}',length=20,relative_path='unused',frames_sha256=[f'{s}-{split}-{v}-{i}' for i in range(20)],frame_ids=list(range(20)),split=role,video_label=label) for s in common.SCENES for split,role,label,values in [('training','train',0,range(1,9)),('testing','train',1,range(20,22)),('testing','evaluation',0,range(22,24))] for v in values]

def test_reference_selection_deterministic_training_only_and_no_query_overlap():
 rows=training_rows();a,pre=common.select_references(rows);b,pre2=common.select_references(list(reversed(rows)))
 assert a==b and pre==pre2
 for scene in common.SCENES:
  refs=a['references'][scene]
  assert [r['video_id'].split('/')[-1] for r in refs]==['02','04','06','08']
  assert [r['frame_id'] for r in refs]==[2,7,11,16]
  assert all(next(r for r in rows if r['id']==f['video_id'])['split']=='train' for f in refs)
  assert all(f['frame_sha256']==next(r for r in rows if r['id']==f['video_id'])['frames_sha256'][f['frame_id']] for f in refs)
  assert pre[common.SCENES.index(scene)]['id'] not in {f['video_id'] for f in refs}

def test_N_X_identical_text_and_only_grounded_object_phrase_changes_C0():
 old=json.loads((common.ROOT/'experiments/stage5/prompts.json').read_text());p=common.build_prompts(old)
 for scene in common.SCENES:
  assert p[scene]['N']==p[scene]['X'] and p[scene]['N'].count('<image>')==12 and p[scene]['C0'].count('<image>')==8
  prefix='Scene elements mentioned in normal training observations: ';original=old[scene]['P1'].split(prefix)[1].split('. This list identifies')[0]
  assert p[scene]['C0'].replace(common.NEUTRAL[scene],original)==old[scene]['P1']
  assert 'not a temporal sequence' in p[scene]['N'] and 'same scene' not in p[scene]['N']

def test_reference_role_order_and_cross_scene_rotation():
 refs,_=common.select_references(training_rows());r=common.sanitized(next(r for r in training_rows() if r['id']=='R01/testing/22'));seg=common.segments(20)[0]
 a=common.input_identity('N',r,seg,refs);b=common.input_identity('X',r,seg,refs);c=common.input_identity('C0',r,seg,refs)
 assert a[4:]==b[4:]==c and all(x['role']=='reference' for x in a[:4])
 assert all(x['video_id'].startswith('R01/') for x in a[:4]) and all(x['video_id'].startswith('R02/') for x in b[:4])
 assert [x['name'] for x in b]==[f'Ref{i}' for i in range(1,5)]+[f'Frame{i}' for i in range(1,9)]

def synthetic(monkeypatch,tmp_path,invalid=False):
 out=tmp_path/'experiments/stage6';out.mkdir(parents=True)
 rows=[dict(id=f'{s}/testing/99',scene=s,original_split='testing',video='99',length=17,relative_path='unused',frames_sha256=['hash']*17,frame_ids=list(range(17))) for s in run.SCENES]
 refs,_=common.select_references(training_rows());prompts={s:{c:f'{s} {c}' for c in run.CONDITIONS} for s in run.SCENES};frozen={'fingerprint':'fixed'}
 run.write(out/'preflight_summary.json',{'status':'complete'})
 monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'OUT',out);monkeypatch.setattr(run,'protected',lambda:10284)
 monkeypatch.setattr(run,'setup',lambda:(rows,{'evaluation':rows},refs,prompts,frozen));monkeypatch.setattr(run,'resolve_images',lambda images:images)
 calls=[];labels=[]
 class Fake:
  def __init__(self,_):self.adapter=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text,**kw:text.split()))
  def call(self,files,prompt):
   calls.append((files,prompt));r='Output: '+str(int(len(calls)%2==0))
   if invalid:r='no valid decision'
   return r,dict(seconds=.1,peak_allocated_gib=1.,peak_reserved_gib=1.)
 monkeypatch.setattr(run,'Engine',Fake)
 def load(row,allrows):
  assert len(calls)==24 and len(list((out/'inference').rglob('*.json')))==24
  labels.append(row['id']);return np.array([0]*16+[1])
 monkeypatch.setattr(run,'load_labels',load)
 inv=[];baseline=[]
 for row in rows:
  path=tmp_path/f"cache/stage3/features/{row['scene']}/testing/99.jsonl"
  for seg in run.segments(17):run.append(path,{**seg,'feature':[1.,0.]})
  inv.append({'path':str(path.relative_to(tmp_path)),'local_only':True,'sha256':run.sha(path)})
  for i in range(17):baseline.append(dict(condition='P1',scene=row['scene'],original_split='testing',video='99',frame=i,label=int(i==16),initial=0,retrieved=0,smoothed=0,final=0))
 run.write(tmp_path/'experiments/stage3/inference_inventory.json',inv)
 path=tmp_path/'experiments/stage5/frame_scores.csv';path.parent.mkdir(parents=True)
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(baseline[0]));w.writeheader();w.writerows(baseline)
 return out,calls,labels,rows,refs,prompts,frozen

def test_full_pipeline_reference_images_never_enter_postprocessing_and_labels_wait(monkeypatch,tmp_path):
 out,calls,labels,*_=synthetic(monkeypatch,tmp_path);run.main('evaluation')
 assert len(calls)==24 and len(labels)==4 and {len(c[0]) for c in calls}=={8,12}
 assert json.loads((out/'status.json').read_text())['status']=='complete'
 scores=list(csv.DictReader((out/'frame_scores.csv').open()));assert len(scores)==4*4*17
 metrics=json.loads((out/'metrics.json').read_text());assert metrics['R01']['N']['initial_binary']['tp']==1
 assert all(not f['reference_images_in_features'] for f in json.loads((out/'reused_features.json').read_text()))
 assert run.readlines(out/'events.jsonl')[0]['event']=='all_evaluation_inference_validated_before_labels'

def test_invalid_retained_without_retry_or_labels(monkeypatch,tmp_path):
 out,calls,labels,*_=synthetic(monkeypatch,tmp_path,True)
 with pytest.raises(RuntimeError,match='Invalid model output'):run.main('evaluation')
 assert len(calls)==1 and not labels and not (out/'frame_scores.csv').exists()
 failed=run.readlines(out/'parsing_failures.jsonl');assert len(failed)==1 and failed[0]['prediction'] is None

def test_reference_or_fingerprint_cache_mismatch_rejected(monkeypatch,tmp_path):
 out,calls,_,rows,refs,prompts,frozen=synthetic(monkeypatch,tmp_path);r=rows[0];s=run.segments(17)[0];p=out/'cache.json';engine=run.Engine(out)
 run.predict(engine,p,'N',prompts['R01']['N'],r,s,refs,'fixed')
 changed=deepcopy(refs);changed['references']['R01'][0]['frame_sha256']='changed'
 with pytest.raises(AssertionError,match='cache identity'):run.predict(engine,p,'N',prompts['R01']['N'],r,s,changed,'fixed')
 with pytest.raises(AssertionError,match='cache identity'):run.predict(engine,p,'N',prompts['R01']['N'],r,s,refs,'other')
 assert len(calls)==1

def test_prior_stage4_stage5_artifacts_protected():
 assert common.protected()==10284
