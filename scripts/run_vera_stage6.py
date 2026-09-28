"""Run consensus Stage6; freeze visual references and all inference before labels."""
import argparse,csv,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage6_common import *
from scripts.vera_stage4_common import freeze,Engine,append,readlines,fail
import scripts.train_vera_questions as base
from scripts.evaluate_vera_normal_context import metric,binary,validate_segments
from ipad.vera import refine_scores
SOURCES=['scripts/run_vera_stage6.py','scripts/prepare_vera_stage6.py','scripts/vera_stage6_common.py','scripts/vera_stage4_common.py','scripts/train_vera_questions.py','scripts/evaluate_vera_normal_context.py','ipad/vera.py','ipad/vera_models.py','ipad/common.py','experiments/vera_ipad/model_inventory.json','experiments/stage5/frozen.json','experiments/stage5/prompts.json','experiments/stage5/frame_scores.csv','experiments/stage5/bootstrap_resamples.json','experiments/stage6/protocol.json','experiments/stage6/prompts.json','experiments/stage6/references.json','experiments/stage6/inference_manifest.json','experiments/stage6/protected_previous.json','experiments/stage6/consultation/consensus.json','experiments/stage6/consultation/training_visual_inspection.json']

def setup():
 protected();rows=json.loads((ROOT/'experiments/stage2_1/split.json').read_text())['records'];manifest=json.loads((OUT/'inference_manifest.json').read_text());refs=json.loads((OUT/'references.json').read_text());prompts=json.loads((OUT/'prompts.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
 expected_refs,expected_preflight=select_references(rows);assert refs==expected_refs and manifest['preflight_normal_train']==expected_preflight
 assert prompts==build_prompts(json.loads((ROOT/'experiments/stage5/prompts.json').read_text()))
 assert manifest['evaluation']==[sanitized(r) for r in rows if r['split']=='evaluation' and r['scene'] in SCENES]
 frozen=freeze(OUT,{'protocol':protocol,'evaluation_ids':[r['id'] for r in manifest['evaluation']],'preflight_ids':[r['id'] for r in manifest['preflight_normal_train']],'prompt_sha256':{s:{c:digest(p) for c,p in d.items()} for s,d in prompts.items()},'references_sha256':sha(OUT/'references.json')},SOURCES)
 return rows,manifest,refs,prompts,frozen

def predict(engine,path,c,prompt,row,seg,refs,fingerprint):
 images=input_identity(c,row,seg,refs)
 request={'condition':c,'prompt':prompt,'video_id':row['id'],'segment':seg,'images':images,'num_patches_list':[1]*len(images),'fingerprint':fingerprint}
 key=digest(request)
 if path.exists():
  result=json.loads(path.read_text());assert result['request']==request and result['cache_key']==key,'Inference cache identity differs'
  assert result['parse_status']=='valid' and base.parse_response(result['response'])==result['prediction']
  return result
 files=resolve_images(images);response,stats=engine.call(files,prompt)
 result={'request':request,'cache_key':key,'response':response,**stats,'time':time.time(),'response_retokenized_tokens':len(engine.adapter.tokenizer.encode(response,add_special_tokens=False))}
 try:result.update(prediction=base.parse_response(response),parse_status='valid')
 except ValueError as exc:result.update(prediction=None,parse_status='failed',parse_error=str(exc))
 write(path,result);append(OUT/'calls.jsonl',result)
 if result['parse_status']!='valid':
  append(OUT/'parsing_failures.jsonl',{'source':str(path.relative_to(ROOT)),**result})
  raise RuntimeError('Invalid model output; no imputation, repair, or retry')
 return result

def load_labels(row,rows):
 source=next(r for r in rows if r['id']==row['id'])
 if row['original_split']=='training':return np.zeros(row['length'],dtype=int)
 p=DATA/'IPAD_dataset'/row['scene']/'test_label'/f"{int(row['video']):03d}.npy"
 assert sha(p)==source['label_sha256'];labels=np.load(p,allow_pickle=False).reshape(-1)
 assert len(labels)==row['length'] and np.isin(labels,[0,1]).all();return labels

def inference_path(c,row,seg):return OUT/'inference'/c/row['scene']/row['original_split']/row['video']/f"{seg['center']:06d}.json"

def validate_complete(evaluation,refs,prompts,frozen):
 for row in evaluation:
  for c in CONDITIONS:
   for seg in segments(row['length']):
    r=json.loads(inference_path(c,row,seg).read_text());q=r['request']
    assert q=={'condition':c,'prompt':prompts[row['scene']][c],'video_id':row['id'],'segment':seg,'images':input_identity(c,row,seg,refs),'num_patches_list':[1]*(8 if c=='C0' else 12),'fingerprint':frozen['fingerprint']}
    assert r['cache_key']==digest(q) and r['parse_status']=='valid' and r['prediction']==base.parse_response(r['response'])

def postprocess(rows,evaluation,refs,prompts,frozen):
 validate_complete(evaluation,refs,prompts,frozen)
 append(OUT/'events.jsonl',{'event':'all_evaluation_inference_validated_before_labels','time':time.time()})
 inventory={r['path']:r for r in json.loads((ROOT/'experiments/stage3/inference_inventory.json').read_text()) if r['local_only']}
 baseline=list(csv.DictReader((ROOT/'experiments/stage5/frame_scores.csv').open()));old={(r['scene'],r['original_split'],r['video'],int(r['frame'])):r for r in baseline if r['condition']=='P1'}
 output=[];windows=[];reuse=[]
 for row in evaluation:
  base.guard();segs=segments(row['length']);feature=ROOT/'cache/stage3/features'/row['scene']/row['original_split']/(row['video']+'.jsonl')
  assert sha(feature)==inventory[str(feature.relative_to(ROOT))]['sha256'];features=readlines(feature);validate_segments(features,segs);vectors=[r['feature'] for r in features]
  reuse.append({'video_id':row['id'],'feature_path':str(feature.relative_to(ROOT)),'feature_sha256':sha(feature),'reference_images_in_features':False})
  append(OUT/'events.jsonl',{'event':'evaluation_labels_open','video_id':row['id'],'time':time.time()});y=load_labels(row,rows)
  for c in ALL_CONDITIONS:
   if c=='P1':
    prior=[old[(row['scene'],row['original_split'],row['video'],i)] for i in range(row['length'])];assert [int(r['label']) for r in prior]==y.tolist()
    scores={s:np.array([float(r[s]) for r in prior]) for s in ['initial','retrieved','smoothed','final']};pred=[int(scores['initial'][s['center']]) for s in segs]
   else:
    pred=[json.loads(inference_path(c,row,s).read_text())['prediction'] for s in segs];scores,detail=refine_scores(pred,vectors,row['length'])
    write(OUT/'postprocessing'/c/row['scene']/row['original_split']/(row['video']+'.json'),detail)
   for seg,p in zip(segs,pred):windows.append({'condition':c,'scene':row['scene'],'video_id':row['id'],'center':seg['center'],'score_end':seg['score_end'],'prediction':p,'frame_ids':seg['frame_ids'],'sampled_positive_frames':int(y[seg['frame_ids']].sum()),'scored_positive_frames':int(y[seg['center']:seg['score_end']].sum()),'scored_frames':seg['score_end']-seg['center']})
   for i in range(row['length']):output.append({'condition':c,'scene':row['scene'],'original_split':row['original_split'],'video':row['video'],'frame':i,'label':int(y[i]),**{s:float(v[i]) for s,v in scores.items()}})
 keys=[(r['condition'],r['scene'],r['original_split'],r['video'],r['frame']) for r in output]
 assert len(keys)==len(set(keys))==4*sum(r['length'] for r in evaluation)
 assert all(np.isfinite(r[s]) and -1e-12<=r[s]<=1+1e-12 for r in output for s in ['initial','retrieved','smoothed','final'])
 with (OUT/'frame_scores.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
 write(OUT/'window_diagnostics.json',windows);write(OUT/'reused_features.json',reuse)
 metrics={scene:{c:{s:metric([r for r in output if r['condition']==c and (scene=='pooled' or r['scene']==scene)],s) for s in ['initial','retrieved','smoothed','final']} for c in ALL_CONDITIONS} for scene in [*SCENES,'pooled']}
 for scene in [*SCENES,'pooled']:
  for c in ALL_CONDITIONS:
   part=[r for r in output if r['condition']==c and (scene=='pooled' or r['scene']==scene)];metrics[scene][c]['initial_binary']=binary([r['label'] for r in part],[int(r['initial']) for r in part])
 metrics['macro']={c:{s:{k:float(np.mean([metrics[scene][c][s][k] for scene in SCENES])) for k in ['auroc','ap']} for s in ['initial','retrieved','smoothed','final']} for c in ALL_CONDITIONS}
 write(OUT/'metrics.json',metrics);return metrics

def main(phase):
 rows,manifest,refs,prompts,frozen=setup();evaluation=manifest['evaluation']
 if phase=='preflight':
  engine=Engine(OUT);records=[]
  for row in manifest['preflight_normal_train']:
   available=segments(row['length']);chosen=np.linspace(0,len(available)-1,5,dtype=int);assert len(set(chosen))==5
   for c in CONDITIONS:
    for index in chosen:
     seg=available[index];r=predict(engine,OUT/'preflight'/c/row['scene']/f"{seg['center']:06d}.json",c,prompts[row['scene']][c],row,seg,refs,frozen['fingerprint'])
     records.append({'scene':row['scene'],'condition':c,'video_id':row['id'],'center':seg['center'],'prediction':r['prediction'],'parse_status':r['parse_status'],'seconds':r['seconds'],'peak_allocated_gib':r['peak_allocated_gib']})
   print(json.dumps({'event':'preflight_scene_complete','scene':row['scene'],'calls':len(records)}),flush=True)
  write(OUT/'preflight_summary.json',{'status':'complete','records':records,'selection_or_rewrite':False});print(json.dumps({'event':'preflight_complete','calls':len(records)}),flush=True);return
 assert json.loads((OUT/'preflight_summary.json').read_text())['status']=='complete'
 write(OUT/'status.json',{'status':'running','phase':'evaluation_inference','fingerprint':frozen['fingerprint']})
 engine=Engine(OUT)
 for scene in SCENES:
  for c in CONDITIONS:
   for row in [r for r in evaluation if r['scene']==scene]:
    for seg in segments(row['length']):predict(engine,inference_path(c,row,seg),c,prompts[scene][c],row,seg,refs,frozen['fingerprint'])
    print(json.dumps({'event':'video_complete','scene':scene,'condition':c,'video':row['id']}),flush=True)
 del engine;gc.collect();torch.cuda.empty_cache();base.guard()
 metrics=postprocess(rows,evaluation,refs,prompts,frozen);calls=readlines(OUT/'calls.jsonl')
 write(OUT/'runtime_summary.json',{'calls':len(calls),'model_seconds':sum(r['seconds'] for r in calls),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in calls),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in calls)})
 write(OUT/'status.json',{'status':'complete','videos':len(evaluation),'frames_per_condition':sum(r['length'] for r in evaluation),'new_conditions':CONDITIONS,'baseline':'Stage5 P1 unchanged','protected_previous_files':protected(),'fingerprint':frozen['fingerprint'],'finished_at':time.time()})
 print(json.dumps({'event':'evaluation_complete','macro':metrics['macro']},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','evaluation'],required=True);args=p.parse_args()
 try:main(args.phase)
 except BaseException as exc:fail(OUT,exc);raise
