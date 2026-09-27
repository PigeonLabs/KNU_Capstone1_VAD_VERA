import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.vera_fact_grounding import fact_table,candidate_errors,filter_candidates,candidate_prompt

def summaries():return [{'video_id':f'R04/training/{i:02d}','facts':[{'center':j,'claim':f'visible {j}'} for j in [32,64,128]]} for i in [1,3,7]]
def rule(table,ident='N1'):
 return {'id':ident,'condition':'while visible','normal_expectation':'the object stays on the surface','question':'Is there a visible deviation from the position?','evidence_fact_ids':[i for i in table if i.endswith('F01')]}
def test_fact_ids_deterministic_and_mapping_exact():
 a=fact_table(summaries());assert a==fact_table(copy.deepcopy(summaries()))
 assert a['R04_training_07_F01']=={'fact_id':'R04_training_07_F01','video_id':'R04/training/07','center':32,'claim':'visible 32'}
 assert a['R04_training_07_F03']['center']==128

def test_unknown_fact_rejected_without_replacement():
 t=fact_table(summaries());r=rule(t);r['evidence_fact_ids'][-1]='R04_training_07_F00';before=copy.deepcopy(r)
 kept,invalid,dupes=filter_candidates([r],t);assert not kept and len(invalid)==1 and not dupes and r==before

def test_three_facts_from_same_video_cannot_bypass_distinct_video_check():
 t=fact_table(summaries());r=rule(t);r['evidence_fact_ids']=[i for i in t if i.startswith('R04_training_07_')]
 assert 'At least three distinct generation videos required' in candidate_errors(r,t)

def test_three_distinct_videos_are_required_and_resolved_by_program():
 t=fact_table(summaries());r=rule(t);assert candidate_errors(r,t)==[]
 kept,_,_=filter_candidates([r],t);assert [e['center'] for e in kept[0]['evidence']]==[32,32,32]
 r['evidence_fact_ids']=r['evidence_fact_ids'][:2];assert candidate_errors(r,t)

def test_duplicate_filter_stable_exact_normalization_only():
 t=fact_table(summaries());a=rule(t);b=rule(t,'N2');b['normal_expectation']='  THE  object stays on the surface ';b['question']='Different question?'
 c=rule(t,'N3');c['normal_expectation']='Something else';c['question']='  Is there a visible deviation from the position?'
 d=rule(t,'N4');d['normal_expectation']='The object rests on the surface';d['question']='Is the object displaced?'
 one=filter_candidates([a,b,c,d],t);assert one==filter_candidates([a,b,c,d],t)
 assert [r['id'] for r in one[0]]==['N1','N4'];assert len(one[2])==2

@pytest.mark.parametrize('field,value',[('center',0),('video_id','R04/training/07'),('evidence',[{'video_id':'R04/training/07','center':0}])])
def test_candidate_direct_coordinate_fields_forbidden(field,value):
 t=fact_table(summaries());r=rule(t);r[field]=value;assert candidate_errors(r,t)

def test_old_zero_center_hallucination_has_no_coordinate_channel():
 t=fact_table(summaries());p=candidate_prompt(t);assert '"center": 0' not in p and '"video_id"' not in p
 r=rule(t);kept,_,_=filter_candidates([r],t);assert all(e['center']==32 for e in kept[0]['evidence'])
 r['evidence_fact_ids']=[{'video_id':'R04/training/07','center':0}];assert candidate_errors(r,t)

@pytest.mark.parametrize('support_state,audit_state,success',[('supported','supported',True),('contradicted','supported',False),('supported','contradicted',False),('supported','maybe',False)])
def test_r04_visual_support_and_normal_audit_pipeline(tmp_path,monkeypatch,support_state,audit_state,success):
 import scripts.build_vera_normal_context_r04 as runner
 from scripts.vera_stage4_common import write,append
 out=tmp_path/'experiments/stage4/R04/normal';out.mkdir(parents=True);write(out/'input_cache_manifest.json',{'test':True})
 rows=[{'id':f'R04/training/{i:02d}','scene':'R04','original_split':'training','video':f'{i:02d}','length':16,'relative_path':'fixture','frames_sha256':[],'frame_ids':list(range(16))} for i in [1,3,7,9]]
 monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'split_scene',lambda scene:(rows,rows[:3],rows[3:]));monkeypatch.setattr(runner,'freeze',lambda *args:{'fingerprint':'test'})
 class FakeEngine:
  def __init__(self,out):self.out=out;self.adapter=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a,**k:[1]*50))
  def recorded(self,path,prompt,row=None,seg=None,validator=None):
   if 'observations' in path.parts:response='object on surface'
   elif 'video_summaries' in path.parts:response=json.dumps({'facts':[{'center':0,'claim':'object on surface'}],'uncertainty':'visible only'})
   elif path.name=='candidates.json':response=json.dumps({'rules':[{k:v for k,v in rule(fact_table([{'video_id':r['id'],'facts':[{'center':0,'claim':'object on surface'}]} for r in rows[:3]])).items() if k!='evidence_fact_ids'}]})
   elif 'grounding' in path.parts:response='R04_training_'+path.stem+'_F01'
   elif 'support_checks' in path.parts:response='N1: '+support_state+' | fixture'
   else:response='N1: '+audit_state+' | fixture'
   record={'response':response,'request_hash':'fixture','video_id':row['id'] if row else None,'segment':seg,'seconds':0,'peak_allocated_gib':0,'peak_reserved_gib':0}
   if validator:record['parsed']=json.loads(response);validator(record['parsed'])
   write(path,record);append(self.out/'calls.jsonl',record);return record
 monkeypatch.setattr(runner,'Engine',FakeEngine)
 if not success:
  with pytest.raises(RuntimeError,match='No normal rules'):runner.main('R04')
 else:
  runner.main('R04');assert json.loads((out/'status.json').read_text())['accepted_rules']==1
  assert len(list((out/'support_checks').rglob('*.json')))==3 and len(list((out/'audit').rglob('*.json')))==1
 rules=json.loads((out/'rules.json').read_text());assert rules[0]['accepted']==success

def test_r04_runner_refuses_other_scenes():
 from scripts.build_vera_normal_context_r04 import main
 with pytest.raises(AssertionError,match='only execute R04'):main('R01')

def test_protected_r01_r03_artifacts_unchanged():
 from scripts.vera_stage4_common import ROOT,sha
 p=ROOT/'experiments/stage4/r04_protected_artifacts.json'
 assert p.exists()
 expected=json.loads(p.read_text())['files'];actual={str(f.relative_to(ROOT)):sha(f) for s in ['R01','R02','R03'] for f in (ROOT/'experiments/stage4'/s).rglob('*') if f.is_file()}
 assert actual==expected

def test_r04_evaluation_protocol_uses_same_base_and_frozen_inputs():
 from scripts.evaluate_vera_normal_context import make_prompt,validate_segments,binary
 from scripts.vera_stage4_common import base,segments,CONFIG
 template=(base.REF/'VERA_learner_instruct.txt').read_text();normal='N1. Object on surface';q='1. Is there visible displacement?'
 assert normal in make_prompt(template,normal,q) and q in make_prompt(template,normal,q)
 assert CONFIG.center_stride==16 and CONFIG.sampled_frames==8
 expected=segments(17);validate_segments(expected,expected)
 with pytest.raises(AssertionError):validate_segments(expected[:-1],expected)
 b=binary([0,1,1],[0,0,1]);assert b['tp']==1 and b['fn']==1 and b['fp']==0

def test_candidate_array_wrapper_never_changes_content_or_evidence():
 from scripts.vera_fact_grounding import parse_candidate_response
 r=rule(fact_table(summaries()));value,action=parse_candidate_response('```json\n'+json.dumps([r])+'\n```')
 assert action=='wrapped_top_level_array' and value=={'rules':[r]}
 assert parse_candidate_response(json.dumps({'rules':[r]}))==({'rules':[r]},'none')
 with pytest.raises(ValueError):parse_candidate_response('{"rules": [')

def test_r04_evaluation_end_to_end_fixture_and_labels_after_inference(tmp_path,monkeypatch):
 """Synthetic pipeline regression only; never a real R04 experiment result."""
 import csv,numpy as np
 import scripts.evaluate_vera_normal_context as module
 from scripts.vera_stage4_common import write,append,sha,segments,CONFIG,base
 normal=tmp_path/'experiments/stage4/R04/normal';normal.mkdir(parents=True)
 (normal/'normal_description.txt').write_text('N1. fixture normal');(normal/'questions.txt').write_text('1. Is there a visible deviation?')
 write(normal/'status.json',{'status':'complete'});write(normal/'frozen.json',{'fixture':True});write(normal/'normal_profile.json',{'description_sha256':sha(normal/'normal_description.txt'),'questions_sha256':sha(normal/'questions.txt')})
 prior=tmp_path/'experiments/stage3';write(prior/'frozen.json',{'config':vars(CONFIG),'source_sha256':{},'fingerprint':'fixture'})
 q=tmp_path/'experiments/stage2_3/questions.txt';q.parent.mkdir(parents=True);q.write_text(base.INITIAL)
 allrows=[];old=[];inventory=[];val=[]
 for i,split in enumerate(['training','testing']):
  row={'id':f'R04/{split}/01','scene':'R04','original_split':split,'video':'01','length':16,'relative_path':'fixture','frames_sha256':[],'frame_ids':list(range(16)),'split':'evaluation'}
  if i:
   lp=tmp_path/'IPAD_dataset/R04/test_label/001.npy';lp.parent.mkdir(parents=True);np.save(lp,np.ones(16,dtype=int));row['label_sha256']=sha(lp)
  allrows.append(row)
  v={**row,'id':f'R04/{split}/02','video':'02','split':'validation','video_label':i,'training_frame_ids':[0]*8};allrows.append(v);val.append({'video_id':v['id'],'prediction':0,'response':'Output: 0','target':i})
  fp=tmp_path/'cache/stage3/features/R04'/split/'01.jsonl';vp=prior/'inference/R04'/split/'01.jsonl'
  for seg in segments(16):
   append(fp,{**seg,'fingerprint':'fixture','feature':[1.,0.]});append(vp,{**seg,'fingerprint':'fixture','prediction':0,'response':'Output: 0'})
  inventory.append({'path':str(fp.relative_to(tmp_path)),'sha256':sha(fp),'local_only':True})
  old.extend({'scene':'R04','original_split':split,'video':'01','frame':j,'label':i,'initial':0.,'retrieved':0.,'smoothed':0.,'final':0.} for j in range(16))
 write(prior/'inference_inventory.json',inventory);write(tmp_path/'experiments/stage2_2/validation/0000.json',{'records':val})
 with (prior/'frame_scores.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(old)
 monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'DATA',tmp_path);monkeypatch.setattr(module,'split_scene',lambda scene:(allrows,[],[]));monkeypatch.setattr(base,'guard',lambda:None)
 seen=[]
 def freeze(out,settings,sources):
  write(out/'frozen.json',{**settings,'fingerprint':'fixture'});return {'fingerprint':'fixture'}
 monkeypatch.setattr(module,'freeze',freeze)
 class FakeEngine:
  def __init__(self,out):self.out=out
  def recorded(self,path,prompt,row,seg):
   assert (self.out/'frozen.json').exists();seen.append((row['id'],prompt,seg))
   r={'video_id':row['id'],'segment':seg,'prompt':prompt,'response':'Output: 0','seconds':0.,'peak_allocated_gib':0.,'peak_reserved_gib':0.};write(path,r);append(self.out/'calls.jsonl',r);return r
 monkeypatch.setattr(module,'Engine',FakeEngine);real_load=np.load
 def load(*args,**kwargs):
  assert len(seen)==8,'Frame labels opened before both conditions finished inference'
  return real_load(*args,**kwargs)
 monkeypatch.setattr(np,'load',load);module.main('R04')
 out=tmp_path/'experiments/stage4/R04/evaluation';metrics=json.loads((out/'metrics.json').read_text())
 assert json.loads((out/'status.json').read_text())['status']=='complete'
 assert all(metrics[c]['final']['frames']==32 and metrics[c]['final']['auroc']==50. and metrics[c]['final']['ap']==50. for c in ['A','B','C'])
 assert len(list(csv.DictReader((out/'frame_scores.csv').open())))==96
