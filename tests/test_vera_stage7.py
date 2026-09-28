"""Prospective Stage7 parser and unchanged paired inference/label gates."""
import json
from types import SimpleNamespace
import pytest
import scripts.vera_stage7_common as common
import scripts.run_vera_stage7 as run
from scripts.vera_stage7_engine import parse_response,quality
import tests.test_vera_stage6 as prior

@pytest.mark.parametrize('text,value',[('Output: 0',0),('\nOutput: 1  \nEvidence: Visible difference.',1),('Output: 0\nOutput: 0',0),('Output: 1\nEvidence: cut unfinished',1)])
def test_explicit_decision_and_consistent_duplicates(text,value):assert parse_response(text)==value
@pytest.mark.parametrize('text',['Evidence: normal\nOutput: 0',' Output: 0','**Output: 0**','output: 0','Output: 0\nOutput: 1','Output: 0\nOutput:','Output: 1\nOUTPUT: 1','Output: 0\nEvidence: Output: 0','Output: 0.'])
def test_invalid_not_recovered(text):
 with pytest.raises(ValueError):parse_response(text)
def test_quality_does_not_change_class():
 text='Output: 1\nEvidence: unfinished';assert parse_response(text)==1
 q=quality(text,{'generated_token_count':128,'ended_with_eos':False});assert q['explanation_truncated']
 assert quality('Output: 0\nOutput: 0',{})['duplicate_output']
 assert quality('Output: 0',{})['evidence_missing']
def test_prompt_only_response_contract_changes():
 new=common.build_prompts(json.loads((common.ROOT/'experiments/stage5/prompts.json').read_text()));old=json.loads((common.ROOT/'experiments/stage6/prompts.json').read_text())
 for scene in common.SCENES:
  assert new[scene]['N']==new[scene]['X']
  for c in common.CONDITIONS:
   p=new[scene][c];assert p.count('<image>')==(8 if c=='C0' else 12)
   assert p.startswith(old[scene][c].split('Explain the most relevant')[0]);assert 'The final line must' not in p and 'Observations:' not in p
   assert 'first nonempty line' in p

def setup(monkeypatch,tmp_path,invalid=False):
 # Exercise same actual evaluation code under fake model/labels; no real inference.
 monkeypatch.setattr(prior,'run',run);monkeypatch.setattr(prior,'common',common)
 out,calls,labels,*rest=prior.synthetic(monkeypatch,tmp_path,invalid)
 fake=run.Engine
 class DecisionFake(fake):
  def call(self,files,prompt):
   response,stats=super().call(files,prompt)
   if not invalid:response='Output: 1\nEvidence: visible test'
   return response,stats
 monkeypatch.setattr(run,'Engine',DecisionFake)
 return out,calls,labels,rest

def test_full_pipeline_waits_for_all_calls_before_labels(monkeypatch,tmp_path):
 out,calls,labels,_=setup(monkeypatch,tmp_path);run.main('evaluation');assert len(calls)==24 and len(labels)==4
 assert json.loads((out/'status.json').read_text())['status']=='complete'
 assert all(not x['reference_images_in_features'] for x in json.loads((out/'reused_features.json').read_text()))
def test_invalid_stops_without_labels_retry_or_imputation(monkeypatch,tmp_path):
 out,calls,labels,_=setup(monkeypatch,tmp_path,True)
 with pytest.raises(RuntimeError):run.main('evaluation')
 assert len(calls)==1 and not labels and not (out/'metrics.json').exists()
def test_stage4_stage5_stage6_artifacts_unchanged():assert common.protected()>10284

def test_generation_capture_preserves_tensor_and_exact_generation_arguments(monkeypatch,tmp_path):
 import torch
 from PIL import Image
 import scripts.vera_stage7_engine as e
 image=tmp_path/'frame.jpg';Image.new('RGB',(2,2)).save(image)
 calls=[];returned=torch.tensor([[7,8,2]])
 class Model:
  def generate(self,**kwargs):calls.append(kwargs);return returned
  def chat(self,tokenizer,pixels,prompt,config,num_patches_list):
   assert config=={'num_beams':1,'max_new_tokens':128,'do_sample':False}
   assert num_patches_list==[1] and prompt=='frozen prompt'
   tensor=self.generate(eos_token_id=2,**config);assert tensor is returned
   return 'Output: 0'
 model=Model();original=model.generate
 engine=e.Engine.__new__(e.Engine);engine.adapter=SimpleNamespace(model=model,tokenizer=None,transform=lambda im:None)
 monkeypatch.setattr(e.base,'guard',lambda:None);monkeypatch.setattr(torch,'stack',lambda images:SimpleNamespace(to=lambda **kw:None))
 for name in ['synchronize','reset_peak_memory_stats']:monkeypatch.setattr(torch.cuda,name,lambda:None)
 for name in ['max_memory_allocated','max_memory_reserved']:monkeypatch.setattr(torch.cuda,name,lambda:0)
 response,stats=engine.call([image],'frozen prompt')
 assert response=='Output: 0' and model.generate==original and len(calls)==1
 assert stats['generated_token_ids']==[7,8,2] and stats['generated_token_count']==3 and stats['ended_with_eos'] is True
 assert stats['actual_generation_config']=={'num_beams':1,'max_new_tokens':128,'do_sample':False}
