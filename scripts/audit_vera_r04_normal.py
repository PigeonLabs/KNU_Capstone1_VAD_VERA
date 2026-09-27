"""Audit the separated R04 pipeline from raw responses and frozen source artifacts."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write,digest,split_scene,segments,sanitized,readlines
from scripts.vera_fact_grounding import fact_table,parse_candidate_response
from scripts.vera_video_grounding import text_candidates,rule_prompt,video_facts,grounding_prompt
from scripts.build_vera_normal_context_v2 import parse_visual_checks

def main():
 out=ROOT/'experiments/stage4/R04/normal';status=json.loads((out/'status.json').read_text());assert status['status'] in ['complete','failed']
 frozen=json.loads((out/'frozen.json').read_text())
 for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h
 cache=json.loads((out/'input_cache_manifest.json').read_text());assert sha(out/'input_cache_manifest.json')==frozen['input_cache_manifest_sha256']
 for p,h in cache['cached_artifacts'].items():assert sha(out/p)==h
 assert json.loads((out/'model_verification.json').read_text())['files']==json.loads((out/'cached_model_inventory.json').read_text())
 _,gen,audit=split_scene('R04');assert frozen['generate_ids']==[r['id'] for r in gen] and frozen['audit_ids']==[r['id'] for r in audit]
 assert json.loads((out/'input_manifest.json').read_text())=={'generation':[sanitized(r) for r in gen],'audit':[sanitized(r) for r in audit]}
 summaries=[{'video_id':r['id'],**json.loads((out/'video_summaries'/r['original_split']/(r['video']+'.json')).read_text())['parsed']} for r in gen]
 table=fact_table(summaries);assert table==json.loads((out/'fact_table.json').read_text())
 proposal=json.loads((out/'candidates.json').read_text());assert proposal['prompt']==rule_prompt(summaries)
 assert parse_candidate_response(proposal['response'])[0]==proposal['parsed']
 texts,schema_invalid,duplicates=text_candidates(proposal['parsed']['rules'])
 rules_frozen=json.loads((out/'candidate_rules_frozen.json').read_text());assert rules_frozen['rules']==texts and rules_frozen['rules_sha256']==digest(texts)
 assert json.loads((out/'duplicate_candidates.json').read_text())==duplicates
 ground_summary=json.loads((out/'grounding_summary.json').read_text());assert len(ground_summary)==len(texts)
 allrules=json.loads((out/'rules.json').read_text());byid={r['id']:r for r in allrules};grounded=[];selected_totals={};ground_fail=0
 for rule,group in zip(texts,ground_summary):
  assert group['rule_id']==rule['id'] and group['candidate_rule_sha256']==digest(rule)
  assert [r['video_id'] for r in group['videos']]==[r['id'] for r in gen]
  selected=[];failed_count=0;none_count=0
  for video,result in zip(gen,group['videos']):
   raw=json.loads((ROOT/result['source']).read_text());facts=video_facts(table,video['id']);allowed={r['fact_id']:r for r in facts}
   assert raw['prompt']==grounding_prompt(rule,facts) and raw['grounding_video_id']==video['id']
   assert raw['candidate_rule_sha256']==digest(rule) and raw['frozen_rules_sha256']==digest(texts)
   value=raw['response'].strip();state='none' if value=='NONE' else 'selected' if value in allowed else 'failed'
   assert result['state']==raw['parsed']['state']==state
   if state=='selected':
    assert raw['parsed']['evidence']==result['evidence']==allowed[value];assert result['fact_id']==value;selected.append(allowed[value])
   elif state=='failed':failed_count+=1;assert result['evidence'] is None and result['fact_id'] is None
   else:none_count+=1
  count=len({r['video_id'] for r in selected});assert len(selected)==count and group['grounded_generation_videos']==count
  assert group['grounding_failures']==failed_count and group['none_videos']==none_count;ground_fail+=failed_count
  saved=byid[rule['id']];assert all(saved[k]==v for k,v in rule.items()) and saved['evidence']==selected
  selected_totals[rule['id']]={'grounded':count,'none':none_count,'failed':failed_count}
  if count>=3:grounded.append(saved)
  else:assert not saved['support_eligible'] and not saved['support_checks'] and not saved['accepted']
 assert json.loads((out/'candidate_filter_summary.json').read_text())['grounded_rules']==len(grounded)
 eligible=[];support_calls=0;per_rule={}
 for rule in grounded:
  checks=[]
  for fact in rule['evidence']:
   row=next(r for r in gen if r['id']==fact['video_id']);p=out/'support_checks'/rule['id']/row['original_split']/row['video']/f"{fact['center']:06d}.json";raw=json.loads(p.read_text())
   assert raw['video_id']==row['id'] and raw['segment']==next(s for s in segments(row['length']) if s['center']==fact['center'])
   assert raw['parsed']==parse_visual_checks(raw['response'],[rule['id']]);checks.append({'video_id':row['id'],'center':fact['center'],**raw['parsed']['checks'][0]});support_calls+=1
  assert checks==rule['support_checks'];counts={state:sum(r['state']==state for r in checks) for state in ['supported','contradicted','unobservable','failed']}
  n=len({r['video_id'] for r in checks if r['state']=='supported'});ok=n>=3 and not counts['contradicted'] and not counts['failed'];assert rule['supported_generation_videos']==n and rule['support_eligible']==bool(ok)
  per_rule[rule['id']]={'visual':counts}
  if ok:eligible.append(rule['id'])
 audit_records=[];windows=0
 if eligible:
  for row in audit:
   for seg in segments(row['length']):
    raw=json.loads((out/'audit'/row['original_split']/row['video']/f"{seg['center']:06d}.json").read_text());assert raw['segment']==seg and raw['video_id']==row['id'];combined=[]
    assert len(raw['source_records'])==len(eligible)
    for ident,path in zip(eligible,raw['source_records']):
     one=json.loads((ROOT/path).read_text());assert one['video_id']==row['id'] and one['segment']==seg and one['parsed']==parse_visual_checks(one['response'],[ident]);combined+=one['parsed']['checks']
    assert combined==raw['parsed']['checks'];audit_records.extend({'video_id':row['id'],'center':seg['center'],**c} for c in combined);windows+=1
 else:assert not list((out/'audit').rglob('*.json'))
 for rule in allrules:
  matches=[r for r in audit_records if r['id']==rule['id']];contr=[r for r in matches if r['state']=='contradicted'];fail=[r for r in matches if r['state']=='failed']
  assert rule['audit_contradictions']==contr and rule['audit_failures']==fail and rule['accepted']==bool(rule['support_eligible'] and not contr and not fail)
  per_rule.setdefault(rule['id'],{})['audit']={state:sum(r['state']==state for r in matches) for state in ['supported','contradicted','unobservable','failed']}
 accepted=[r for r in allrules if r['accepted']]
 if accepted:
  assert status['status']=='complete';profile=json.loads((out/'normal_profile.json').read_text());assert profile['accepted_rule_ids']==[r['id'] for r in accepted]
  assert sha(out/'normal_description.txt')==profile['description_sha256'] and sha(out/'questions.txt')==profile['questions_sha256']
  assert profile['normal_tokens']<=1024 and (out/'questions.txt').read_text()=='\n'.join(f"{i+1}. {r['question']}" for i,r in enumerate(accepted[:5]))+'\n'
 else:assert status['status']=='failed' and not (out/'normal_profile.json').exists() and not (out.parent/'evaluation').exists()
 protection=json.loads((ROOT/'experiments/stage4/r04_protected_artifacts.json').read_text())['files'];actual={str(p.relative_to(ROOT)):sha(p) for s in ['R01','R02','R03'] for p in (ROOT/'experiments/stage4'/s).rglob('*') if p.is_file()};assert actual==protection
 allowed={r['id'] for r in gen+audit};assert all(r.get('video_id') is None or r['video_id'] in allowed for r in readlines(out/'calls.jsonl'))
 result={'status':'passed','experiment_status':status['status'],'candidate_rules':len(proposal['parsed']['rules']),'schema_invalid':len(schema_invalid),'duplicate_rules':len(duplicates),'unique_text_rules':len(texts),'grounded_rules':len(grounded),'accepted_rules':len(accepted),'accepted_rule_ids':[r['id'] for r in accepted],'per_video_grounding_calls':len(texts)*len(gen),'grounding_by_rule':selected_totals,'grounding_failures':ground_fail,'fact_table_rows':len(table),'per_rule_checks':per_rule,'support_calls':support_calls,'audit_windows':windows,'audit_checks':len(audit_records),'candidate_text_immutable':True,'all_grounded_evidence_visually_checked':True,'normal_train_only':True,'source_hashes_verified':True,'cache_files_unchanged':True,'protected_R01_R03_files':len(protection),'R01_R03_unchanged':True}
 write(out/'independent_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
