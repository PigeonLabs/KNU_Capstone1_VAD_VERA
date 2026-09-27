"""Deterministic fact IDs and strict candidate validation; no semantic repair."""
import re

def fact_table(summaries):
 table={}
 for summary in summaries:
  video=summary['video_id']
  assert re.fullmatch(r'R\d{2}/(?:training|testing)/\d+',video)
  for i,fact in enumerate(summary['facts'],1):
   ident=video.replace('/','_')+f'_F{i:02d}'
   assert ident not in table and type(fact['center']) is int
   table[ident]={'fact_id':ident,'video_id':video,'center':fact['center'],'claim':fact['claim']}
 return table

def normalize(text):return ' '.join(text.lower().split())

def candidate_errors(rule,table):
 required={'id','condition','normal_expectation','question','evidence_fact_ids'}
 errors=[]
 if not isinstance(rule,dict):return ['Candidate must be an object']
 if set(rule)!=required:errors.append('Only id, condition, normal_expectation, question, evidence_fact_ids are allowed; direct coordinates forbidden')
 for key in ['id','condition','normal_expectation','question']:
  if not isinstance(rule.get(key),str) or not rule[key].strip():errors.append('Invalid text field: '+key)
 if isinstance(rule.get('question'),str) and not rule['question'].endswith('?'):errors.append('Question must end with ?')
 if isinstance(rule.get('condition'),str) and len(rule['condition'])>200:errors.append('Condition too long')
 if isinstance(rule.get('normal_expectation'),str) and len(rule['normal_expectation'])>300:errors.append('Expectation too long')
 ids=rule.get('evidence_fact_ids')
 if not isinstance(ids,list) or not all(isinstance(i,str) for i in ids):return errors+['Evidence must be a list of exact fact-ID strings']
 if len(set(ids))!=len(ids):errors.append('Repeated fact IDs')
 unknown=[i for i in ids if i not in table]
 if unknown:errors.append('Unknown fact IDs: '+repr(unknown))
 if len({table[i]['video_id'] for i in ids if i in table})<3:errors.append('At least three distinct generation videos required')
 return errors

def filter_candidates(proposed,table):
 accepted=[];invalid=[];duplicates=[];seen_expectation={};seen_question={}
 for rule in proposed:
  errors=candidate_errors(rule,table)
  if errors:invalid.append({'candidate':rule,'reasons':errors});continue
  expectation=normalize(rule['normal_expectation']);question=normalize(rule['question'])
  prior=seen_expectation.get(expectation) or seen_question.get(question)
  if prior:
   duplicates.append({'candidate':rule,'duplicate_of':prior,'reason':'normalized expectation or question identical'});continue
  seen_expectation[expectation]=rule['id'];seen_question[question]=rule['id']
  accepted.append({**rule,'evidence':[dict(table[i]) for i in rule['evidence_fact_ids']]})
 return accepted,invalid,duplicates

def candidate_prompt(table):
 import json
 facts=[{'fact_id':r['fact_id'],'claim':r['claim']} for r in table.values()]
 return '''Using ONLY the supplied normal training facts, propose 1 to 5 concise conditional normal rules observed in at least THREE DISTINCT videos. A fact ID prefix before _F identifies its video. Select evidence_fact_ids verbatim from the supplied list; never invent, edit or correct IDs. Three facts from one video do not count as three videos. Return no video_id or center fields. Preserve phase dependence: state when each rule applies, do not demand actions throughout a process. Do not infer hidden object functions, absent-object prohibitions, unsupported process order or exact timing. Each question must check VISIBLE DEVIATION from its normal rule, not mere object presence. Avoid duplicate expectations and duplicate questions. Keep statements short, condition <=200 characters, expectation <=300 characters. Give exactly three fact IDs from three distinct videos per rule. Return JSON only with top-level "rules" array. Each rule has exactly these fields: "id" (N1, N2, ... in order), "condition" (text), "normal_expectation" (text), "question" (text ending ?), "evidence_fact_ids" (array of supplied ID strings). Do not claim expert-certified normality.\nFACTS:\n'''+json.dumps(facts)


def parse_candidate_response(response):
 import json
 text=response.strip()
 if text.startswith('```'):
  text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I);text=re.sub(r'\s*```$','',text)
 value=json.loads(text)
 # A missing outer object is a structural wrapper only; every rule/ID stays exact.
 if isinstance(value,list):return {'rules':value},'wrapped_top_level_array'
 if not isinstance(value,dict):raise ValueError('Expected rules object or array')
 return value,'none'
