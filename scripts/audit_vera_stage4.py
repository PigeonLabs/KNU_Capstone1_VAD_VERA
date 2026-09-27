"""Recompute stage4 evidence acceptance and matched-frame evaluation independently."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import *
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix

def main(scene,kind):
 out=ROOT/'experiments/stage4'/scene/kind;status=json.loads((out/'status.json').read_text());assert status['status']=='complete'
 frozen=json.loads((out/'frozen.json').read_text())
 for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h
 assert frozen['split_sha256']==sha(ROOT/'experiments/stage2_1/split.json')
 allrows,gen,audit=split_scene(scene)
 if kind=='normal':
  import importlib
  generator=importlib.import_module('scripts.build_vera_normal_context_v2' if 'scripts/build_vera_normal_context_v2.py' in frozen['source_sha256'] else 'scripts.build_vera_normal_context')
  evidence_errors=generator.evidence_errors;parse_visual_checks=generator.parse_visual_checks
  assert frozen['generate_ids']==[r['id'] for r in gen] and frozen['audit_ids']==[r['id'] for r in audit]
  manifest=json.loads((out/'input_manifest.json').read_text());assert manifest=={'generation':[sanitized(r) for r in gen],'audit':[sanitized(r) for r in audit]}
  observed=0
  for row in gen:
   for seg in segments(row['length']):
    r=json.loads((out/'observations'/row['original_split']/row['video']/f"{seg['center']:06d}.json").read_text());assert r['video_id']==row['id'] and r['segment']==seg
    assert r['request_hash']==digest({k:r[k] for k in ['prompt','video_id','segment']});observed+=1
  fallback_videos=[]
  for row in gen:
   summary=json.loads((out/'video_summaries'/row['original_split']/(row['video']+'.json')).read_text())
   if summary.get('derived_not_model_json'):
    observations=[{'center':seg['center'],'description':json.loads((out/'observations'/row['original_split']/row['video']/f"{seg['center']:06d}.json").read_text())['response']} for seg in segments(row['length'])]
    indices=np.linspace(0,len(observations)-1,min(6,len(observations)),dtype=int).tolist()
    assert summary['summary_fallback']['selected_indices']==indices and summary['summary_fallback']['observations_sha256']==digest(observations)
    assert summary['parsed']['facts']==[{'claim':observations[i]['description'],'center':observations[i]['center']} for i in indices]
    fallback_videos.append(row['id'])
  rules=json.loads((out/'rules.json').read_text());profile=json.loads((out/'normal_profile.json').read_text());eligible=[r['id'] for r in rules if r['support_eligible']]
  audit_rows=[]
  if eligible:
   for row in audit:
    for seg in segments(row['length']):
     r=json.loads((out/'audit'/row['original_split']/row['video']/f"{seg['center']:06d}.json").read_text());assert r['video_id']==row['id'] and r['segment']==seg
     assert [c['id'] for c in r['parsed']['checks']]==eligible
     combined=[]
     for source_path,ident in zip(r['source_records'],eligible):
      one=json.loads((ROOT/source_path).read_text());assert one['segment']==seg and one['video_id']==row['id']
      assert one['parsed']==parse_visual_checks(one['response'],[ident]);combined.extend(one['parsed']['checks'])
     assert combined==r['parsed']['checks']
     audit_rows.append({'video_id':row['id'],'center':seg['center'],**r['parsed']})
  summaries=[{'video_id':r['id'],**json.loads((out/'video_summaries'/r['original_split']/(r['video']+'.json')).read_text())['parsed']} for r in gen]
  for rule in rules:
   errors=evidence_errors(rule,summaries)
   if errors:
    assert rule['rejection_reason']==errors and not rule['accepted'] and not rule['support_eligible'] and rule['support_checks']==[]
    continue
   support=[]
   for citation in rule['evidence']:
    row=next(r for r in gen if r['id']==citation['video_id']);p=out/'support_checks'/rule['id']/row['original_split']/row['video']/f"{citation['center']:06d}.json"
    r=json.loads(p.read_text());assert r['segment']==next(s for s in segments(row['length']) if s['center']==citation['center'])
    assert r['parsed']==parse_visual_checks(r['response'],[rule['id']])
    support.append({'video_id':row['id'],'center':citation['center'],**r['parsed']['checks'][0]})
   assert support==rule['support_checks']
   n=len({r['video_id'] for r in support if r['state']=='supported'});ok=n>=3 and not any(r['state']=='contradicted' for r in support)
   assert n==rule['supported_generation_videos'] and ok==rule['support_eligible']
   contradictions=[{'video_id':a['video_id'],'center':a['center'],**c} for a in audit_rows for c in a['checks'] if c['id']==rule['id'] and c['state']=='contradicted']
   assert contradictions==rule['audit_contradictions']
   failures=[{'video_id':a['video_id'],'center':a['center'],**c} for a in audit_rows for c in a['checks'] if c['id']==rule['id'] and c['state']=='failed']
   assert failures==rule['audit_failures'];assert rule['accepted']==(ok and not contradictions and not failures)
  accepted=[r for r in rules if r['accepted']];assert profile['accepted_rule_ids']==[r['id'] for r in accepted]
  assert sha(out/'normal_description.txt')==profile['description_sha256'] and sha(out/'questions.txt')==profile['questions_sha256']
  assert len(accepted)<=12 and profile['normal_tokens']<=1024
  assert (out/'questions.txt').read_text()=='\n'.join(f"{i+1}. {r['question']}" for i,r in enumerate(accepted[:5]))+'\n'
  calls=readlines(out/'calls.jsonl');allowed={r['id'] for r in gen+audit}
  assert all(r.get('video_id') is None or r['video_id'] in allowed for r in calls)
  result={'status':'passed','scene':scene,'generation_videos':len(gen),'audit_videos':len(audit),'observed_segments':observed,'audit_segments':len(audit_rows),'accepted_rules':len(accepted),'normal_train_only':True,'disjoint_generation_audit':True,'all_candidate_decisions_recomputed':True,'source_hashes_verified':True,'human_visual_annotation':False,'summary_fallback_videos':fallback_videos,'fallback_exact_verbatim_and_centers_verified':True}
 else:
  from scripts.evaluate_vera_normal_context import make_prompt
  normal=ROOT/'experiments/stage4'/scene/'normal'
  assert sha(normal/'normal_profile.json')==frozen['normal_profile_sha256'] and sha(normal/'frozen.json')==frozen['normal_frozen_sha256']
  prompt=json.loads((out/'prompts.json').read_text())
  template=(base.REF/'VERA_learner_instruct.txt').read_text();description=(normal/'normal_description.txt').read_text()
  assert prompt['B']==make_prompt(template,description,base.INITIAL) and prompt['C']==make_prompt(template,description,(normal/'questions.txt').read_text())
  assert {k:digest(v) for k,v in prompt.items()}==frozen['prompt_sha256']
  rows=list(csv.DictReader((out/'frame_scores.csv').open()));metrics=json.loads((out/'metrics.json').read_text());evaluation=[r for r in allrows if r['scene']==scene and r['split']=='evaluation']
  expected={(c,r['original_split'],r['video'],str(i)) for c in ['A','B','C'] for r in evaluation for i in range(r['length'])}
  keys=[(r['condition'],r['original_split'],r['video'],r['frame']) for r in rows];assert len(keys)==len(set(keys)) and set(keys)==expected
  for c in ['A','B','C']:
   current=[r for r in rows if r['condition']==c];y=[int(r['label']) for r in current]
   for column in ['initial','retrieved','smoothed','final']:
    p=[float(r[column]) for r in current];assert np.isfinite(p).all()
    assert abs(100*roc_auc_score(y,p)-metrics[c][column]['auroc'])<1e-10
    assert abs(100*average_precision_score(y,p)-metrics[c][column]['ap'])<1e-10
   tn,fp,fn,tp=confusion_matrix(y,[int(float(r['initial'])) for r in current],labels=[0,1]).ravel()
   for key,val in [('tn',tn),('fp',fp),('fn',fn),('tp',tp)]:assert metrics[c]['initial_binary'][key]==val
   if c!='A':
    for video in evaluation:
     for seg in segments(video['length']):
      r=json.loads((out/'inference'/c/video['original_split']/video['video']/f"{seg['center']:06d}.json").read_text())
      assert r['video_id']==video['id'] and r['segment']==seg and r['prompt']==prompt[c]
      pred=base.parse_response(r['response'])
      assert all(int(float(x['initial']))==pred for x in current if x['original_split']==video['original_split'] and x['video']==video['video'] and seg['center']<=int(x['frame'])<seg['score_end'])
  for entry in json.loads((out/'reused_sources.json').read_text()):
   assert sha(ROOT/entry['feature_path'])==entry['sha256'] and sha(ROOT/entry['baseline_inference_path'])==entry['baseline_inference_sha256']
  result={'status':'passed','scene':scene,'videos':len(evaluation),'frames_per_condition':len(rows)//3,'three_conditions_exact_support':True,'four_score_stage_metrics_recomputed':True,'binary_confusion_recomputed':True,'prompts_and_sources_verified':True,'cached_baseline_and_features_verified':True}
 write(out/'independent_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--kind',choices=['normal','evaluation'],required=True);a=p.parse_args();main(a.scene,a.kind)
