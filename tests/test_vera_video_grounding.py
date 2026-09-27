import copy,json
import pytest
from scripts.vera_fact_grounding import fact_table
from scripts.vera_video_grounding import *

def fixture():
 summaries=[{'video_id':f'R04/training/{i:02d}','facts':[{'center':j,'claim':f'normal observed pattern {j}'} for j in [0,16,32]],'uncertainty':'visible only'} for i in range(1,15)]
 rule={'id':'N1','condition':'when contact is visible','normal_expectation':'the object stays on the table','question':'Is there a visible deviation from expected contact?'}
 return summaries,fact_table(summaries),rule

def test_candidate_schema_has_no_evidence_fields():
 s,t,r=fixture();assert text_candidates([r])[0]==[r]
 for key in ['evidence','evidence_fact_ids','video_id','center']:
  bad={**r,key:[]};assert text_candidates([bad])[1]
 prompt=rule_prompt(s);assert 'GROUPED NORMAL SUMMARIES' in prompt and 'three fact IDs' not in prompt
 assert '"evidence_fact_ids":' not in prompt

def test_each_grounding_call_has_exactly_one_video_and_rule_is_immutable():
 s,t,r=fixture();before=copy.deepcopy(r)
 for video in s:
  facts=video_facts(t,video['video_id']);prompt=grounding_prompt(r,facts)
  for other in s:
   prefix=other['video_id'].replace('/','_')+'_F'
   assert (prefix in prompt)==(other['video_id']==video['video_id'])
 assert r==before
 with pytest.raises(AssertionError):grounding_prompt(r,list(t.values()))

def test_foreign_id_rejected_and_none_not_failure():
 s,t,r=fixture();facts=video_facts(t,s[0]['video_id'])
 assert parse_grounding('R04_training_02_F01',facts)['state']=='failed'
 assert parse_grounding('NONE',facts)=={'state':'none','fact_id':None,'evidence':None}
 assert parse_grounding('No support',facts)['state']=='failed'
 assert parse_grounding('R04_training_01_F01 because visible',facts)['state']=='failed'
 assert parse_grounding(' R04_training_01_F01\n',facts)['state']=='selected'

def test_python_distinct_count_and_same_video_cannot_bypass():
 s,t,r=fixture();results=[]
 for video in s:
  facts=video_facts(t,video['video_id']);results.append({'video_id':video['video_id'],**parse_grounding(facts[0]['fact_id'],facts)})
 e,n=grounded_evidence(results);assert n==14 and len(e)==14
 with pytest.raises(AssertionError):grounded_evidence(results[:1]*3)
 results[1]={'video_id':s[1]['video_id'],**parse_grounding('NONE',video_facts(t,s[1]['video_id']))}
 assert grounded_evidence(results)[1]==13

def test_dedup_before_grounding_preserves_first_text():
 s,t,r=fixture();second={**r,'id':'N2','normal_expectation':'  THE object   stays on the table '}
 kept,invalid,duplicates=text_candidates([r,second]);assert kept==[r] and not invalid and duplicates[0]['duplicate_of']=='N1'

def test_normal_pipeline_never_loads_evaluation_labels_and_acceptance_unchanged():
 import ast,inspect
 import scripts.build_vera_normal_context_r04 as runner
 source=inspect.getsource(runner)
 assert 'np.load(' not in source and 'test_label' not in source and 'evaluation/metrics' not in source
 assert "support>=3 and not any(r['state'] in ['contradicted','failed'] for r in checks)" in source
 assert "r['accepted']=r['support_eligible'] and not r['audit_contradictions'] and not r['audit_failures']" in source
 assert 'for e in rule[\'evidence\']:' in source
