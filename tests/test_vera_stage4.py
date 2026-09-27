import json
from pathlib import Path
import pytest
from scripts.vera_stage4_common import split_scene,sanitized,parse_json,digest

@pytest.mark.parametrize('scene,counts',[('R01',(22,6)),('R02',(17,5)),('R03',(13,4)),('R04',(14,4))])
def test_scene_normal_support_is_disjoint_and_deterministic(scene,counts):
 records,a,b=split_scene(scene);_,a2,b2=split_scene(scene)
 assert (len(a),len(b))==counts and a==a2 and b==b2
 assert not {r['id'] for r in a}&{r['id'] for r in b}
 assert all(r['scene']==scene and r['video_label']==0 and r['split']=='train' for r in a+b)
 assert not {r['id'] for r in a+b}&{r['id'] for r in records if r['split']!='train'}
 for r in a+b:
  public=sanitized(r)
  assert not {'video_label','label_path','label_sha256','split'}&set(public)
  assert len(public['frame_ids'])==len(public['frames_sha256'])==public['length']

def test_json_format_fences_do_not_change_semantics():
 x={'state':'unobservable','frame_numbers':[],'evidence':'occluded'}
 assert parse_json('```json\n'+json.dumps(x)+'\n```')==x
 assert digest(x)==digest(dict(reversed(list(x.items()))))

@pytest.mark.parametrize('text',['[]','{"state":','preface {"state":0}','{"x":1} trailing'])
def test_invalid_json_is_not_a_normal_prediction(text):
 with pytest.raises((ValueError,json.JSONDecodeError)):parse_json(text)

def test_format_repair_accumulates_prior_correction(tmp_path):
 from scripts.vera_stage4_common import Engine
 engine=object.__new__(Engine);engine.out=tmp_path;seen=[]
 responses=iter(['{"x":0,"y":0}','{"x":1,"y":0}','{"x":1,"y":1}'])
 def text(prompt):
  seen.append(prompt)
  return next(responses),{'seconds':0.,'peak_allocated_gib':0.,'peak_reserved_gib':0.}
 engine.text=text
 def validator(value):
  assert value['x']==1,'x must be 1'
  assert value['y']==1,'y must be 1'
 record=engine.recorded(tmp_path/'output.json','Return JSON with x=1 and y=1',validator=validator)
 assert record['parsed']=={'x':1,'y':1}
 assert 'Current response to repair:\n{"x":1,"y":0}' in seen[2]
 assert len(record['format_repairs'])==2
 assert record['response']=='{"x":0,"y":0}'

def test_invalid_evidence_rule_is_rejected_without_reanchoring():
 from scripts.build_vera_normal_context import evidence_errors
 summaries=[{'video_id':f'R01/training/{i:02d}','facts':[{'center':16,'claim':'visible'}]} for i in [1,2,3]]
 rule={'evidence':[{'video_id':r['video_id'],'center':16} for r in summaries]}
 assert evidence_errors(rule,summaries)==[]
 rule['evidence'][-1]['center']=0
 assert evidence_errors(rule,summaries)==['Uncited summary center: R01/training/03/0']
 assert rule['evidence'][-1]['center']==0
 rule['evidence'][-1]['video_id']='R02/training/03'
 assert 'Outside normal generation support' in evidence_errors(rule,summaries)[0]

def test_visual_check_failure_is_not_normal_or_unobservable():
 from scripts.build_vera_normal_context import parse_visual_checks
 a=parse_visual_checks('N1: supported | clamp is fixed\nN2: unobservable | obscured', ['N1','N2','N3'])['checks']
 assert [r['state'] for r in a]==['supported','unobservable','failed']
 assert parse_visual_checks('N1: supported | x\nN1: contradicted | y',['N1'])['checks'][0]['state']=='failed'
 assert parse_visual_checks('N1: maybe | x',['N1'])['checks'][0]['state']=='failed'

def test_explicit_single_visual_verdict_is_read_without_inventing_rationale():
 from scripts.build_vera_normal_context import parse_visual_checks
 r=parse_visual_checks('supported',['N3'])['checks'][0]
 assert r['state']=='supported' and r['id']=='N3' and not r['rationale_provided']
 assert all(r['state']=='failed' for r in parse_visual_checks('supported',['N1','N2'])['checks'])
 assert parse_visual_checks('supported or contradicted',['N1'])['checks'][0]['state']=='failed'

def test_named_visual_scalar_keeps_identity_and_rejects_ambiguity():
 from scripts.build_vera_normal_context import parse_visual_checks
 x=parse_visual_checks('N1: supported\nN3: unobservable',['N1','N3'])['checks']
 assert [r['state'] for r in x]==['supported','unobservable']
 assert all(not r['rationale_provided'] for r in x)
 assert parse_visual_checks('N1: supported or contradicted',['N1'])['checks'][0]['state']=='failed'
 assert parse_visual_checks('N2: supported',['N1'])['checks'][0]['state']=='failed'
 assert parse_visual_checks('N1: supported\nN1: contradicted',['N1'])['checks'][0]['state']=='failed'
