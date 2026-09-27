"""Extract scene-specific normal rules from normal training data, with cited visual checks."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import *
SOURCES=['scripts/build_vera_normal_context.py','scripts/vera_stage4_common.py','scripts/train_vera_questions.py','ipad/vera.py','ipad/vera_models.py','ipad/common.py','experiments/vera_ipad/source_reference/VERA/VERA_learner_instruct.txt']

def main(scene):
 out=ROOT/'experiments/stage4'/scene/'normal';records,generate,audit=split_scene(scene)
 settings={'scene':scene,'generate_ids':[r['id'] for r in generate],'audit_ids':[r['id'] for r in audit],
 'purpose':'normal training only; no abnormal training/validation/evaluation input; no outcome selection',
 'normal_split':'per-scene sorted IDs, default_rng(0), ceil(20%) first shuffled IDs for audit',
 'max_rules':5,'approved_upper_bound_rules':12,'max_normal_tokens':1024,'max_questions':5,
 'candidate_support':'at least 3 distinct generation videos; one proposed cited segment per video is visually checked; fewer than 3 visually supported videos excludes rule',
 'audit':'all windows of normal audit videos; any clear contradiction excludes rule from mandatory description',
 'format_repairs':'at most 2 text-only serialization/schema repairs, original evidence unchanged, failures retained',
 'observations':'all stride-16 clipped 10-second windows, 8 frames; <=80 word target descriptions',
 'rule_validation_limit':'same frozen VLM checks candidate evidence; not independent human ground truth; no precise unsupported phase/timing requirements'}
 frozen=freeze(out,settings,SOURCES)
 if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['status']=='complete':return
 write(out/'input_manifest.json',{'generation':[sanitized(r) for r in generate],'audit':[sanitized(r) for r in audit]})
 write(out/'status.json',{'status':'running','phase':'observations','scene':scene})
 engine=Engine(out);summaries=[]
 for row in generate:
  observations=[]
  for seg in segments(row['length']):
   prompt=image_prefix(seg)+'''These frames are from a NORMAL training video of one industrial tabletop process. Describe only visible objects, relative position/contact, and changes across these frames. Distinguish stationary, moving, and unobservable states. Do not invent object functions, names, required timing, hidden causes, or process steps. Do not diagnose anomalies. Include observed variations and uncertainty. Be concrete and concise, at most 80 English words.'''
   r=engine.recorded(out/'observations'/row['original_split']/row['video']/f"{seg['center']:06d}.json",prompt,row,seg)
   observations.append({'center':seg['center'],'description':r['response']})
  centers={s['center'] for s in segments(row['length'])}
  def validate(v):
   assert isinstance(v['facts'],list) and 1<=len(v['facts'])<=max(6,len(centers))
   for f in v['facts']:assert isinstance(f['claim'],str) and f['claim'] and type(f['center']) is int and f['center'] in centers
   assert isinstance(v['uncertainty'],str)
  prompt='Summarize only the supplied observations of ONE normal training video. Preserve conditional phase dependence and uncertainty; no universal always/never claim or invented process order. Return JSON {"facts":[{"claim":"short observable normal state/relation/change", "center":0}],"uncertainty":"short"}. At most 6 facts, each cite an actual center from the observations.\n'+json.dumps(observations)
  r=engine.recorded(out/'video_summaries'/row['original_split']/(row['video']+'.json'),prompt,validator=validate)
  summaries.append({'video_id':row['id'],**r['parsed']});print(json.dumps({'event':'normal_video_observed','scene':scene,'video':row['id'],'segments':len(observations)}),flush=True)
 write(out/'status.json',{'status':'running','phase':'candidate_rules','scene':scene})
 lookup={r['id']:r for r in generate}
 def candidate_schema(v):
  assert isinstance(v['rules'],list) and 1<=len(v['rules'])<=5
  assert [r['id'] for r in v['rules']]==[f'N{i+1}' for i in range(len(v['rules']))]
  for r in v['rules']:
   for key in ['normal_expectation','condition','question']:assert isinstance(r[key],str) and r[key]
   assert r['question'].endswith('?') and len(r['normal_expectation'])<=300 and len(r['condition'])<=200
   assert isinstance(r['evidence'],list) and len({e['video_id'] for e in r['evidence']})>=3
   for e in r['evidence']:
    assert e['video_id'] in lookup
    facts=next(s['facts'] for s in summaries if s['video_id']==e['video_id'])
    assert type(e['center']) is int and e['center'] in {f['center'] for f in facts}
 prompt='''Build a compact scene-specific normal reference using ONLY these normal-video summaries. Propose 1 to 5 observable rules shared by at least THREE different videos. State when each rule applies; avoid requiring an action throughout the whole process. Do not infer absent objects are forbidden, exact timing, hidden functions, or an unsupported step order. Include an anomaly-detection question for each rule, checking visible deviation, not merely object presence. Each rule must cite 3 distinct video IDs and an actual center in that video's summary. Keep normal_expectation <=300 characters and condition <=200 characters. Return JSON only: {"rules":[{"id":"N1","condition":"when applicable","normal_expectation":"observed normal property","question":"Is there a visible deviation from ...?","evidence":[{"video_id":"...","center":0},{"video_id":"...","center":0},{"video_id":"...","center":0}]}]}. Do not claim expert-certified normality.\n'''+json.dumps(summaries)
 candidates=engine.recorded(out/'candidates.json',prompt,validator=candidate_schema)['parsed']['rules']
 def check_schema(v):
  assert v['state'] in ['supported','contradicted','unobservable']
  assert isinstance(v['frame_numbers'],list) and all(type(n) is int and 1<=n<=8 for n in v['frame_numbers'])
  assert isinstance(v['evidence'],str) and v['evidence']
  if v['state']!='unobservable':assert len(v['frame_numbers'])>0
 checked=[]
 for rule in candidates:
  checks=[]
  for e in rule['evidence']:
   row=lookup[e['video_id']];seg=next(s for s in segments(row['length']) if s['center']==e['center'])
   prompt=image_prefix(seg)+'Check this candidate normal statement against these frames. Supported means its applicability condition is visible AND the expected relation/state is visibly supported. Contradicted means the condition is visible but the expectation is clearly false. If condition/evidence cannot be seen, answer unobservable. A stationary phase does not contradict a conditional movement rule. Do not trust the statement merely because it is proposed. Return JSON {"state":"supported|contradicted|unobservable","frame_numbers":[1],"evidence":"short visual justification"}.\n'+json.dumps({k:rule[k] for k in ['id','condition','normal_expectation']})
   r=engine.recorded(out/'support_checks'/rule['id']/row['original_split']/row['video']/f"{seg['center']:06d}.json",prompt,row,seg,check_schema)
   checks.append({'video_id':row['id'],'center':seg['center'],**r['parsed']})
  support=len({r['video_id'] for r in checks if r['state']=='supported'})
  checked.append({**rule,'support_checks':checks,'supported_generation_videos':support,'support_eligible':support>=3 and not any(r['state']=='contradicted' for r in checks)})
 write(out/'support_verification.json',checked)
 eligible=[r for r in checked if r['support_eligible']]
 write(out/'status.json',{'status':'running','phase':'normal_audit','scene':scene,'supported_rules':len(eligible)})
 audit_results=[]
 def audit_schema(v):
  assert isinstance(v['checks'],list) and [r['id'] for r in v['checks']]==[r['id'] for r in eligible]
  for r in v['checks']:check_schema(r)
 for row in audit:
  if not eligible:break
  for seg in segments(row['length']):
   prompt=image_prefix(seg)+'These are held-out NORMAL training frames. Independently test each proposed conditional normal rule. State supported only if its condition and expected property are visible. State contradicted only for a visible violation when the condition applies. State unobservable for unclear condition, occlusion or insufficient frames. Never assume movement is required in a stationary phase. Return JSON {"checks":[{"id":"N1","state":"supported|contradicted|unobservable","frame_numbers":[1],"evidence":"brief observed evidence"}]} with exactly one entry per supplied rule in the same order.\n'+json.dumps([{k:r[k] for k in ['id','condition','normal_expectation']} for r in eligible])
   r=engine.recorded(out/'audit'/row['original_split']/row['video']/f"{seg['center']:06d}.json",prompt,row,seg,audit_schema)
   audit_results.append({'video_id':row['id'],'center':seg['center'],**r['parsed']})
  print(json.dumps({'event':'normal_audit_video_complete','scene':scene,'video':row['id']}),flush=True)
 for r in checked:
  matches=[{'video_id':a['video_id'],'center':a['center'],**c} for a in audit_results for c in a['checks'] if c['id']==r['id']]
  r['audit_contradictions']=[m for m in matches if m['state']=='contradicted'];r['audit_observable_segments']=sum(m['state']!='unobservable' for m in matches)
  r['accepted']=r['support_eligible'] and not r['audit_contradictions']
 accepted=[r for r in checked if r['accepted']]
 if not accepted:
  write(out/'rules.json',checked);raise RuntimeError('No normal rules survived evidence checks; no invented fallback')
 description='Normal reference for '+scene+' (normal training observations, not exhaustive process specifications):\n'+'\n'.join(f"{r['id']}. When {r['condition']}: {r['normal_expectation']}" for r in accepted)+'\nAllowed variation / uncertainty: apply only when the stated condition is visible. Other phases, occlusion and unobserved details are not automatically violations. Exact timing and a mandatory global step order have not been established.\n'
 tokens=len(engine.adapter.tokenizer.encode(description,add_special_tokens=False));assert tokens<=1024,'Normal reference exceeds fixed token budget'
 # Deterministic order, at most five questions; no validation-label-based choice.
 questions='\n'.join(f"{i+1}. {r['question']}" for i,r in enumerate(accepted[:5]))+'\n'
 (out/'normal_description.txt').write_text(description);(out/'questions.txt').write_text(questions);write(out/'rules.json',checked)
 write(out/'normal_profile.json',{'scene':scene,'normal_description':description,'questions':questions,'accepted_rule_ids':[r['id'] for r in accepted],'question_rule_ids':[r['id'] for r in accepted[:5]],'normal_tokens':tokens,'description_sha256':sha(out/'normal_description.txt'),'questions_sha256':sha(out/'questions.txt'),'selection':'first five accepted rules in frozen candidate order; no performance-based selection','limitations':'Support/audit judgments are generated by the same frozen VLM, not independent human annotation. Sparse frames may miss changes.'})
 calls=readlines(out/'calls.jsonl');write(out/'runtime_summary.json',{'calls':len(calls),'sum_model_seconds':sum(r['seconds'] for r in calls),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in calls),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in calls),'timing_scope':'model.chat only; excludes preprocessing/loading/hashing/logging'})
 write(out/'status.json',{'status':'complete','scene':scene,'generation_videos':len(generate),'audit_videos':len(audit),'candidate_rules':len(candidates),'accepted_rules':len(accepted),'normal_tokens':tokens,'fingerprint':frozen['fingerprint'],'finished_at':time.time()});print('NORMAL PROFILE COMPLETE '+scene,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True);a=p.parse_args();out=ROOT/'experiments/stage4'/a.scene/'normal'
 try:main(a.scene)
 except BaseException as exc:fail(out,exc);raise
