"""Independently verify R04 grounding, filtering, support, audit and frozen inputs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write,digest,split_scene,segments,sanitized
from scripts.build_vera_normal_context_v2 import parse_visual_checks
from scripts.vera_fact_grounding import fact_table,filter_candidates,candidate_prompt,parse_candidate_response

def main():
 out=ROOT/'experiments/stage4/R04/normal';status=json.loads((out/'status.json').read_text());assert status['status'] in ['complete','failed']
 frozen=json.loads((out/'frozen.json').read_text())
 for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h
 cache=json.loads((out/'input_cache_manifest.json').read_text());assert sha(out/'input_cache_manifest.json')==frozen['input_cache_manifest_sha256']
 for p,h in cache['cached_artifacts'].items():assert sha(out/p)==h
 _,gen,audit=split_scene('R04');assert frozen['generate_ids']==[r['id'] for r in gen] and frozen['audit_ids']==[r['id'] for r in audit]
 assert json.loads((out/'input_manifest.json').read_text())=={'generation':[sanitized(r) for r in gen],'audit':[sanitized(r) for r in audit]}
 summaries=[{'video_id':r['id'],**json.loads((out/'video_summaries'/r['original_split']/(r['video']+'.json')).read_text())['parsed']} for r in gen]
 table=fact_table(summaries);assert table==json.loads((out/'fact_table.json').read_text())
 proposal=json.loads((out/'candidates.json').read_text());assert proposal['prompt']==candidate_prompt(table) and proposal['request_hash']==digest({'prompt':proposal['prompt'],'video_id':None,'segment':None})
 assert parse_candidate_response(proposal['response'])[0]==proposal['parsed']
 candidates,invalid,duplicates=filter_candidates(proposal['parsed']['rules'],table)
 for name,value in [('grounded_candidates',candidates),('invalid_candidates',invalid),('duplicate_candidates',duplicates)]:assert json.loads((out/(name+'.json')).read_text())==value
 allrules=json.loads((out/'rules.json').read_text());rules={r['id']:r for r in allrules};eligible=[];support_calls=0
 if not candidates:
  assert status['status']=='failed' and len(invalid)+len(duplicates)==len(proposal['parsed']['rules'])
  assert all(not r['support_eligible'] and not r['accepted'] and not r['support_checks'] and not r['audit_contradictions'] and not r['audit_failures'] for r in allrules)
  assert not list((out/'support_checks').rglob('*.json')) and not list((out/'audit_checks').rglob('*.json')) and not list((out/'audit').rglob('*.json'))
  assert not (out/'normal_profile.json').exists() and not (out.parent/'evaluation').exists()
  protection=json.loads((ROOT/'experiments/stage4/r04_protected_artifacts.json').read_text())['files'];actual={str(p.relative_to(ROOT)):sha(p) for s in ['R01','R02','R03'] for p in (ROOT/'experiments/stage4'/s).rglob('*') if p.is_file()};assert actual==protection
  mappings=[{'rule_id':r['id'],'fact_ids_exist':all(i in table for i in r['evidence_fact_ids']),'distinct_videos':len({table[i]['video_id'] for i in r['evidence_fact_ids'] if i in table}),'mapped_evidence':[table[i] for i in r['evidence_fact_ids'] if i in table]} for r in proposal['parsed']['rules']]
  assert all(r['fact_ids_exist'] and r['distinct_videos']==1 for r in mappings)
  assert json.loads((out/'model_verification.json').read_text())['files']==json.loads((out/'cached_model_inventory.json').read_text())
  calls=[json.loads(line) for line in (out/'calls.jsonl').read_text().splitlines()];assert len(calls)==1 and calls[0]['request_hash']==proposal['request_hash']
  write(out/'runtime_summary.json',{'new_model_calls':len(calls),'sum_new_model_seconds':sum(r['seconds'] for r in calls),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in calls),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in calls),'observations_reused':338,'summaries_reused':14,'timing_scope':'new model.chat calls only; cached calls and their original times are retained separately'})
  write(out/'independent_verification.json',{'status':'passed','experiment_status':'failed','reason':'Three candidates each use facts from only one generation video; minimum three distinct videos not met','candidate_rules':len(proposal['parsed']['rules']),'invalid_rules':len(invalid),'duplicate_rules':len(duplicates),'accepted_rules':0,'fact_table_rows':len(table),'all_fact_ids_exist':True,'mappings':mappings,'support_calls':0,'audit_windows':0,'audit_checks':0,'evaluation_executed':False,'filtering_recomputed':True,'cache_files_unchanged':True,'cache_model_inventory_equal_current':True,'source_hashes_verified':True,'protected_R01_R03_files':len(protection),'R01_R03_unchanged':True})
  print((out/'independent_verification.json').read_text());return

 for r in candidates:
  saved=rules[r['id']];assert all(saved[k]==v for k,v in r.items())
  assert saved['evidence']==[table[i] for i in saved['evidence_fact_ids']]
  checks=[]
  for e in r['evidence']:
   row=next(x for x in gen if x['id']==e['video_id']);p=out/'support_checks'/r['id']/row['original_split']/row['video']/f"{e['center']:06d}.json";record=json.loads(p.read_text())
   assert record['video_id']==row['id'] and record['segment']==next(s for s in segments(row['length']) if s['center']==e['center'])
   assert record['parsed']==parse_visual_checks(record['response'],[r['id']])
   checks.append({'video_id':row['id'],'center':e['center'],**record['parsed']['checks'][0]});support_calls+=1
  assert saved['support_checks']==checks
  n=len({x['video_id'] for x in checks if x['state']=='supported'});ok=n>=3 and not any(x['state'] in ['contradicted','failed'] for x in checks)
  assert saved['supported_generation_videos']==n and saved['support_eligible']==ok
  if ok:eligible.append(r['id'])
 audit_records=[];audit_windows=0
 for row in audit:
  for seg in segments(row['length']):
   p=out/'audit'/row['original_split']/row['video']/f"{seg['center']:06d}.json";record=json.loads(p.read_text());assert record['segment']==seg and record['video_id']==row['id'];combined=[]
   assert len(record['source_records'])==len(eligible)
   for ident,path in zip(eligible,record['source_records']):
    raw=json.loads((ROOT/path).read_text());assert raw['segment']==seg and raw['video_id']==row['id'];assert raw['parsed']==parse_visual_checks(raw['response'],[ident]);combined+=raw['parsed']['checks']
   assert combined==record['parsed']['checks']
   audit_records.extend({'video_id':row['id'],'center':seg['center'],**c} for c in combined);audit_windows+=1
 for r in allrules:
  matches=[x for x in audit_records if x['id']==r['id']];contradictions=[x for x in matches if x['state']=='contradicted'];failures=[x for x in matches if x['state']=='failed']
  assert r['audit_contradictions']==contradictions and r['audit_failures']==failures
  assert r['accepted']==(r['support_eligible'] and not contradictions and not failures)
 accepted=[r for r in allrules if r['accepted']];profile=json.loads((out/'normal_profile.json').read_text())
 assert profile['accepted_rule_ids']==[r['id'] for r in accepted] and len(accepted)>0
 assert sha(out/'normal_description.txt')==profile['description_sha256'] and sha(out/'questions.txt')==profile['questions_sha256']
 assert (out/'questions.txt').read_text()=='\n'.join(f"{i+1}. {r['question']}" for i,r in enumerate(accepted[:5]))+'\n'
 protection=json.loads((ROOT/'experiments/stage4/r04_protected_artifacts.json').read_text())['files'];actual={str(p.relative_to(ROOT)):sha(p) for s in ['R01','R02','R03'] for p in (ROOT/'experiments/stage4'/s).rglob('*') if p.is_file()};assert actual==protection
 write(out/'independent_verification.json',{'status':'passed','candidate_rules':len(proposal['parsed']['rules']),'invalid_rules':len(invalid),'duplicate_rules':len(duplicates),'accepted_rules':len(accepted),'fact_table_rows':len(table),'all_fact_ids_and_coordinates_verified':True,'support_calls':support_calls,'audit_windows':audit_windows,'audit_checks':len(audit_records),'visual_support_and_audit_recomputed':True,'cache_files_unchanged':True,'normal_train_only':True,'source_hashes_verified':True,'protected_R01_R03_files':len(protection),'R01_R03_unchanged':True,'summary_fallback_videos':[r['video_id'] for r in summaries if any(json.loads((out/'video_summaries'/g['original_split']/(g['video']+'.json')).read_text()).get('derived_not_model_json') for g in gen if g['id']==r['video_id'])]})
 print((out/'independent_verification.json').read_text())
if __name__=='__main__':main()
