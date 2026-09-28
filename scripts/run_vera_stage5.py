"""Frozen prompting-only Stage5; all scene inference precedes evaluation-label access."""
import argparse,csv,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import *
from scripts.evaluate_vera_normal_context import metric,binary,validate_segments
from ipad.vera import refine_scores
OUT=ROOT/'experiments/stage5'
CONDITIONS=['P1','P2','P3']
SOURCES=['scripts/run_vera_stage5.py','scripts/prepare_vera_stage5.py','experiments/stage5/references.json','experiments/stage5/consultation/decisions.md','experiments/vera_ipad/model_inventory.json','scripts/vera_stage4_common.py','scripts/train_vera_questions.py','scripts/evaluate_vera_normal_context.py','ipad/vera.py','ipad/vera_models.py','ipad/common.py','experiments/stage5/prompts.json','experiments/stage5/protocol.json']

def protected():
 expected=json.loads((OUT/'protected_stage4.json').read_text())['files']
 actual={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/stage4').rglob('*') if p.is_file()}
 assert actual==expected,'Stage4 artifacts changed'
 return len(expected)

def inputs():
 allrows=json.loads((ROOT/'experiments/stage2_1/split.json').read_text())['records']
 evaluation=[sanitized(r) for r in allrows if r['scene'] in SCENES and r['split']=='evaluation']
 preflight=[]
 for scene in SCENES:
  _,_,audit=split_scene(scene);preflight.append(sanitized(audit[0]))
 assert not {r['id'] for r in evaluation}&{r['id'] for r in preflight}
 return allrows,evaluation,preflight

def setup():
 protected();allrows,evaluation,preflight=inputs()
 prompts=json.loads((OUT/'prompts.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
 assert set(prompts)==set(SCENES)
 for s in SCENES:
  assert set(prompts[s])==set(CONDITIONS)
  for p in prompts[s].values():assert p.count('<image>')==8
 settings={'protocol':protocol,'prompt_sha256':{s:{c:digest(p) for c,p in d.items()} for s,d in prompts.items()},'evaluation_ids':[r['id'] for r in evaluation],'preflight_ids':[r['id'] for r in preflight],'protected_inventory_sha256':sha(OUT/'protected_stage4.json'),'reference_sources':{f'experiments/stage4/{s}/normal/normal_description.txt':sha(ROOT/f'experiments/stage4/{s}/normal/normal_description.txt') for s in SCENES},'label_policy':'all P1/P2/P3 inference in all four scenes finishes before opening evaluation labels; no selection/tuning','parameters_updated':0}
 frozen=freeze(OUT,settings,SOURCES)
 write(OUT/'inference_manifest.json',{'evaluation':evaluation,'preflight_normal_train':preflight})
 return allrows,evaluation,preflight,prompts,frozen

def predict(engine,path,prompt,row,seg,fingerprint):
 cache_key=digest({'fingerprint':fingerprint,'prompt':prompt,'frame_sha256':[row['frames_sha256'][i] for i in seg['frame_ids']],'segment':seg,'video_id':row['id']})
 if path.exists():
  prior=json.loads(path.read_text());assert prior.get('cache_key')==cache_key,'Inference cache identity differs'
 r=engine.recorded(path,prompt,row,seg)
 tokens=len(engine.adapter.tokenizer.encode(r['response'],add_special_tokens=False))
 r.update(fingerprint=fingerprint,cache_key=cache_key,response_retokenized_tokens=tokens,possible_token_limit=tokens>=1024,input_frame_sha256=[row['frames_sha256'][i] for i in seg['frame_ids']])
 try:r.update(prediction=base.parse_response(r['response']),parse_status='valid')
 except ValueError as exc:
  r.update(prediction=None,parse_status='failed',parse_error=str(exc))
  append(OUT/'parsing_failures.jsonl',{'source':str(path.relative_to(ROOT)),**r})
 write(path,r)
 return r

def load_labels(row,allrows):
 source=next(r for r in allrows if r['id']==row['id'])
 if row['original_split']=='training':return np.zeros(row['length'],dtype=int)
 p=DATA/'IPAD_dataset'/row['scene']/'test_label'/f"{int(row['video']):03d}.npy"
 assert sha(p)==source['label_sha256'];y=np.load(p,allow_pickle=False).reshape(-1)
 assert len(y)==row['length'] and np.isin(y,[0,1]).all()
 return y

def postprocess(allrows,evaluation,prompts,frozen):
 # Fail closed on missing/invalid outputs BEFORE opening any evaluation-label file.
 for row in evaluation:
  for c in CONDITIONS:
   for seg in segments(row['length']):
    p=OUT/'inference'/c/row['scene']/row['original_split']/row['video']/f"{seg['center']:06d}.json"
    r=json.loads(p.read_text());assert r['parse_status']=='valid' and r['prediction']==base.parse_response(r['response'])
    assert r['fingerprint']==frozen['fingerprint'] and r['prompt']==prompts[row['scene']][c] and r['segment']==seg
 append(OUT/'events.jsonl',{'event':'all_evaluation_inference_validated_before_labels','time':time.time()})
 inventory={r['path']:r for r in json.loads((ROOT/'experiments/stage3/inference_inventory.json').read_text()) if r['local_only']}
 output=[];windows=[];reuse=[]
 for row in evaluation:
  base.guard();expected=segments(row['length']);feature=ROOT/'cache/stage3/features'/row['scene']/row['original_split']/(row['video']+'.jsonl')
  assert sha(feature)==inventory[str(feature.relative_to(ROOT))]['sha256']
  features=readlines(feature);validate_segments(features,expected);vectors=[r['feature'] for r in features]
  append(OUT/'events.jsonl',{'event':'evaluation_labels_open','video_id':row['id'],'time':time.time()})
  y=load_labels(row,allrows)
  old=list(csv.DictReader((ROOT/'experiments/stage4'/row['scene']/'evaluation/frame_scores.csv').open()))
  lookup={(r['condition'],int(r['frame'])):r for r in old if r['original_split']==row['original_split'] and r['video']==row['video']}
  reuse.append({'video_id':row['id'],'feature_path':str(feature.relative_to(ROOT)),'feature_sha256':sha(feature)})
  for c in ['A','B','C',*CONDITIONS]:
   if c in CONDITIONS:
    records=[json.loads((OUT/'inference'/c/row['scene']/row['original_split']/row['video']/f"{s['center']:06d}.json").read_text()) for s in expected]
    predictions=[r['prediction'] for r in records];scores,detail=refine_scores(predictions,vectors,row['length'])
    write(OUT/'postprocessing'/c/row['scene']/row['original_split']/(row['video']+'.json'),detail)
   else:
    scores={k:np.array([float(lookup[(c,i)][k]) for i in range(row['length'])]) for k in ['initial','retrieved','smoothed','final']}
    assert all(int(lookup[(c,i)]['label'])==int(y[i]) for i in range(row['length']))
    predictions=[int(scores['initial'][s['center']]) for s in expected]
   for s,pred in zip(expected,predictions):
    sample=y[s['frame_ids']];center=y[s['center']:s['score_end']]
    windows.append({'condition':c,'scene':row['scene'],'video_id':row['id'],'center':s['center'],'score_end':s['score_end'],'prediction':pred,'sampled_positive_frames':int(sample.sum()),'scored_positive_frames':int(center.sum()),'scored_frames':len(center),'frame_ids':s['frame_ids']})
   for i in range(row['length']):output.append({'condition':c,'scene':row['scene'],'original_split':row['original_split'],'video':row['video'],'frame':i,'label':int(y[i]),**{k:float(v[i]) for k,v in scores.items()}})
 keys=[(r['condition'],r['scene'],r['original_split'],r['video'],r['frame']) for r in output]
 assert len(keys)==len(set(keys))==6*sum(r['length'] for r in evaluation)
 assert all(np.isfinite(r[k]) and -1e-12<=r[k]<=1+1e-12 for r in output for k in ['initial','retrieved','smoothed','final'])
 with (OUT/'frame_scores.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
 write(OUT/'window_diagnostics.json',windows);write(OUT/'reused_features.json',reuse)
 metrics={}
 for scene in [*SCENES,'pooled']:
  metrics[scene]={}
  for c in ['A','B','C',*CONDITIONS]:
   rows=[r for r in output if r['condition']==c and (scene=='pooled' or r['scene']==scene)]
   metrics[scene][c]={k:metric(rows,k) for k in ['initial','retrieved','smoothed','final']}
   metrics[scene][c]['initial_binary']=binary([r['label'] for r in rows],[int(r['initial']) for r in rows])
 metrics['macro']={c:{stage:{k:float(np.mean([metrics[s][c][stage][k] for s in SCENES])) for k in ['auroc','ap']} for stage in ['initial','retrieved','smoothed','final']} for c in ['A','B','C',*CONDITIONS]}
 write(OUT/'metrics.json',metrics);return metrics

def main(phase):
 allrows,evaluation,preflight,prompts,frozen=setup()
 if phase=='preflight':
  engine=Engine(OUT);records=[]
  for row in preflight:
   available=segments(row['length']);chosen=np.linspace(0,len(available)-1,min(5,len(available)),dtype=int)
   for c in CONDITIONS:
    for index in chosen:
     seg=available[index]
     r=predict(engine,OUT/'preflight'/c/row['scene']/f"{seg['center']:06d}.json",prompts[row['scene']][c],row,seg,frozen['fingerprint']);records.append({'condition':c,'scene':row['scene'],'center':seg['center'],'parse_status':r['parse_status'],'prediction':r['prediction'],'seconds':r['seconds'],'response_retokenized_tokens':r['response_retokenized_tokens']})
  write(OUT/'preflight_summary.json',{'status':'complete' if all(r['parse_status']=='valid' for r in records) else 'failed','records':records,'selection_or_rewrite':False})
  assert all(r['parse_status']=='valid' for r in records),'Preflight parsing failure; no fallback'
  print(json.dumps({'event':'preflight_complete','records':records}),flush=True);return
 assert json.loads((OUT/'preflight_summary.json').read_text())['status']=='complete'
 write(OUT/'status.json',{'status':'running','phase':'evaluation_inference','fingerprint':frozen['fingerprint']})
 engine=Engine(OUT);invalid=0
 for scene in SCENES:
  for c in CONDITIONS:
   for row in [r for r in evaluation if r['scene']==scene]:
    for seg in segments(row['length']):
     r=predict(engine,OUT/'inference'/c/scene/row['original_split']/row['video']/f"{seg['center']:06d}.json",prompts[scene][c],row,seg,frozen['fingerprint']);invalid+=int(r['parse_status']=='failed')
    print(json.dumps({'event':'video_complete','scene':scene,'condition':c,'video':row['id'],'invalid_outputs':invalid}),flush=True)
 del engine;gc.collect();torch.cuda.empty_cache();base.guard()
 if invalid:raise RuntimeError(f'{invalid} invalid model outputs; no imputation or prompt retry; full metrics withheld')
 metrics=postprocess(allrows,evaluation,prompts,frozen)
 calls=readlines(OUT/'calls.jsonl');write(OUT/'runtime_summary.json',{'calls':len(calls),'model_seconds':sum(r['seconds'] for r in calls),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in calls),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in calls)})
 write(OUT/'status.json',{'status':'complete','videos':len(evaluation),'frames_per_condition':sum(r['length'] for r in evaluation),'new_conditions':CONDITIONS,'baseline_conditions':['A','B','C'],'protected_stage4_files':protected(),'fingerprint':frozen['fingerprint'],'finished_at':time.time()})
 print(json.dumps({'event':'evaluation_complete','macro':metrics['macro']},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','evaluation'],required=True);a=p.parse_args()
 try:main(a.phase)
 except BaseException as exc:fail(OUT,exc);raise
