"""Text-only proposals followed by independently validated per-video grounding."""
import json
from scripts.vera_fact_grounding import normalize,parse_candidate_response

RULE_KEYS={'id','condition','normal_expectation','question'}
def text_candidates(proposed):
 kept=[];invalid=[];duplicates=[];expectations={};questions={}
 for rule in proposed:
  errors=[]
  if not isinstance(rule,dict):invalid.append({'candidate':rule,'reasons':['Rule must be object']});continue
  if set(rule)!=RULE_KEYS:errors.append('Only rule text fields allowed; evidence/coordinates forbidden')
  for key in RULE_KEYS:
   if not isinstance(rule.get(key),str) or not rule[key].strip():errors.append('Invalid '+key)
  if isinstance(rule.get('question'),str) and not rule['question'].endswith('?'):errors.append('Question must end with ?')
  if len(str(rule.get('condition','')))>200 or len(str(rule.get('normal_expectation','')))>300:errors.append('Text length exceeds fixed limit')
  if errors:invalid.append({'candidate':rule,'reasons':errors});continue
  a=normalize(rule['normal_expectation']);b=normalize(rule['question']);prior=expectations.get(a) or questions.get(b)
  if prior:duplicates.append({'candidate':rule,'duplicate_of':prior});continue
  expectations[a]=rule['id'];questions[b]=rule['id'];kept.append(dict(rule))
 return kept,invalid,duplicates

def rule_prompt(summaries):
 groups=[{'normal_training_video':s['video_id'],'observations':[f['claim'] for f in s['facts']],'uncertainty':s['uncertainty']} for s in summaries]
 return '''Propose 1 to 5 concise conditional normal rules from the grouped NORMAL training video summaries below. Your ONLY task is candidate RULE TEXT generation. Propose observable patterns repeatedly seen across multiple normal videos, not details unique to one video. Preserve conditional phase dependence; do not require an action throughout every frame. Do not infer hidden functions, exact timing, absent-object prohibitions or an unsupported global process order. Each anomaly question must ask about VISIBLE DEVIATION from its expected normal pattern, not merely object presence. Avoid duplicate expectations and duplicate questions. Do not select evidence. Do not output evidence_fact_ids, video_id, center or other evidence fields. Keep condition <=200 characters and normal_expectation <=300 characters. Return JSON only: {"rules":[{"id":"N1","condition":"short observable applicability condition","normal_expectation":"short normal pattern","question":"Is there a visible deviation from the expected pattern?"}]}. Use sequential rule IDs N1, N2, ... .\nGROUPED NORMAL SUMMARIES:\n'''+json.dumps(groups)

def video_facts(table,video_id):
 facts=[dict(f) for f in table.values() if f['video_id']==video_id]
 assert facts and all(f['video_id']==video_id for f in facts)
 return facts

def grounding_prompt(rule,facts):
 assert set(rule)==RULE_KEYS and len({f['video_id'] for f in facts})==1
 return 'Candidate rule (frozen; do not modify):\n'+json.dumps(rule)+'\nFacts from ONE normal training video ONLY:\n'+'\n'.join(f["fact_id"]+': '+f['claim'] for f in facts)+'''\nChoose exactly ONE supplied fact ID that most directly supports the candidate's conditional normal statement. If this video has no direct support, output NONE. Do not modify the rule. Do not infer missing evidence. Output the exact fact ID or NONE only, without explanations, quotes or formatting.'''

def parse_grounding(response,facts):
 value=response.strip();allowed={f['fact_id']:f for f in facts};assert len({f['video_id'] for f in facts})==1
 if value=='NONE':return {'state':'none','fact_id':None,'evidence':None}
 if value in allowed:return {'state':'selected','fact_id':value,'evidence':dict(allowed[value])}
 return {'state':'failed','fact_id':None,'evidence':None,'reason':'Output is neither NONE nor an exact fact ID from this video','invalid_output':response}

def grounded_evidence(results):
 videos=[r['video_id'] for r in results]
 assert len(videos)==len(set(videos)),'Only one grounding result per generation video is allowed'
 evidence=[]
 for r in results:
  assert r['state'] in ['selected','none','failed']
  if r['state']=='selected':
   assert r['evidence']['video_id']==r['video_id'] and r['fact_id']==r['evidence']['fact_id'];evidence.append(dict(r['evidence']))
 return evidence,len({e['video_id'] for e in evidence})
