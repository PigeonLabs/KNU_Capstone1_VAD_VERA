"""Frozen A/B/C evaluation for one scene; no performance-based prompt selection."""
import argparse,csv,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import *
from ipad.vera import refine_scores
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
SOURCES=['scripts/evaluate_vera_normal_context.py','scripts/vera_stage4_common.py','scripts/train_vera_questions.py','ipad/vera.py','ipad/vera_models.py','ipad/common.py','experiments/vera_ipad/source_reference/VERA/VERA_learner_instruct.txt']

def metric(rows,column):
 y=np.array([int(r['label']) for r in rows]);p=np.array([float(r[column]) for r in rows])
 return {'frames':len(rows),'positive_frames':int(y.sum()),'auroc':float(100*roc_auc_score(y,p)),'ap':float(100*average_precision_score(y,p))}
def binary(y,p):
 tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
 return {'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'accuracy':float((tp+tn)/len(y)),'recall':float(tp/(tp+fn)) if tp+fn else None,'fpr':float(fp/(fp+tn)) if fp+tn else None}
def make_prompt(template,description,questions):
 prompt=base.learner_prompt(template,questions)
 marker='** Prompt Questions: **';assert prompt.count(marker)==1
 return prompt.replace(marker,'** Scene-specific normal reference: **\n'+description+'\nUse only visible evidence. When explaining a deviation, name the applicable normal rule and frame. Unobservable evidence alone is not a violation.\n'+marker)
def validate_segments(records,expected):
 assert len(records)==len(expected)
 for r,s in zip(records,expected):assert all(r[k]==s[k] for k in s)
def main(scene):
 out=ROOT/'experiments/stage4'/scene/'evaluation';normal=ROOT/'experiments/stage4'/scene/'normal';prior=ROOT/'experiments/stage3'
 assert json.loads((normal/'status.json').read_text())['status']=='complete'
 profile=json.loads((normal/'normal_profile.json').read_text());description=(normal/'normal_description.txt').read_text();questions=(normal/'questions.txt').read_text()
 assert sha(normal/'normal_description.txt')==profile['description_sha256'] and sha(normal/'questions.txt')==profile['questions_sha256']
 allrows,_,_=split_scene(scene);evaluation=[sanitized(r) for r in allrows if r['scene']==scene and r['split']=='evaluation'];validation=[r for r in allrows if r['scene']==scene and r['split']=='validation']
 prior_config=json.loads((prior/'frozen.json').read_text());assert prior_config['config']==vars(CONFIG)
 assert (ROOT/'experiments/stage2_3/questions.txt').read_text()==base.INITIAL
 for path,h in prior_config['source_sha256'].items():assert sha(ROOT/path)==h
 template=(base.REF/'VERA_learner_instruct.txt').read_text()
 prompts={'B':make_prompt(template,description,base.INITIAL),'C':make_prompt(template,description,questions)}
 settings={'scene':scene,'conditions':{'A':'reuse stage3 Q0 unchanged','B':'scene normal description + Q0','C':'same normal description + scene questions'},
 'normal_profile_sha256':sha(normal/'normal_profile.json'),'normal_frozen_sha256':sha(normal/'frozen.json'),'prompt_sha256':{k:digest(v) for k,v in prompts.items()},
 'A_frozen_sha256':sha(prior/'frozen.json'),'A_scores_sha256':sha(prior/'frame_scores.csv'),'feature_inventory_sha256':sha(prior/'inference_inventory.json'),
 'evaluation_ids':[r['id'] for r in evaluation],'validation_ids':[r['id'] for r in validation],
 'validation_policy':'one fixed prompt per condition, no selection or revision using validation outcomes',
 'feature_reuse':'same input windows and ImageBind FP32 features as stage3; file SHA and segment IDs verified; features independent of questions',
 'labels':'frame labels opened only after both B/C evaluation inference is complete','normal_parameters_updated':False}
 frozen=freeze(out,settings,SOURCES)
 if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['status']=='complete':return
 write(out/'inference_manifest.json',evaluation);write(out/'prompts.json',prompts)
 write(out/'status.json',{'status':'running','phase':'validation','scene':scene})
 engine=Engine(out)
 val_prior=json.loads((ROOT/'experiments/stage2_2/validation/0000.json').read_text())
 validation_results={'A':[r for r in val_prior['records'] if r['video_id'].startswith(scene+'/')]}
 assert {r['video_id'] for r in validation_results['A']}=={r['id'] for r in validation}
 for condition in ['B','C']:
  result=[]
  for row in validation:
   seg={'frame_ids':row['training_frame_ids']};r=engine.recorded(out/'validation'/condition/row['original_split']/(row['video']+'.json'),prompts[condition],sanitized(row),seg)
   prediction=base.parse_response(r['response']);result.append({'video_id':row['id'],'prediction':prediction,'target':row['video_label'],'response':r['response']})
  validation_results[condition]=result
 write(out/'validation_metrics.json',{k:binary([r['target'] for r in a],[r.get('prediction',base.parse_response(r['response'])) for r in a]) for k,a in validation_results.items()})
 write(out/'status.json',{'status':'running','phase':'evaluation_inference','scene':scene})
 for condition in ['B','C']:
  for row in evaluation:
   for seg in segments(row['length']):
    r=engine.recorded(out/'inference'/condition/row['original_split']/row['video']/f"{seg['center']:06d}.json",prompts[condition],row,seg)
    try:base.parse_response(r['response'])
    except ValueError:
     append(out/'parsing_failures.jsonl',r);raise
   print(json.dumps({'event':'condition_video_complete','scene':scene,'condition':condition,'video':row['id']}),flush=True)
 del engine;gc.collect();torch.cuda.empty_cache();base.guard()
 write(out/'status.json',{'status':'running','phase':'postprocessing','scene':scene})
 old=list(csv.DictReader((prior/'frame_scores.csv').open()));old=[r for r in old if r['scene']==scene]
 old_lookup={(r['original_split'],r['video'],int(r['frame'])):r for r in old}
 inventory=json.loads((prior/'inference_inventory.json').read_text());feature_index={r['path']:r for r in inventory if r['local_only']}
 output_rows=[];counts={};reused=[]
 for row in evaluation:
  expected=segments(row['length']);fp=ROOT/'cache/stage3/features'/scene/row['original_split']/(row['video']+'.jsonl')
  vp=prior/'inference'/scene/row['original_split']/(row['video']+'.jsonl')
  assert sha(fp)==feature_index[str(fp.relative_to(ROOT))]['sha256']
  features=readlines(fp);a_records=readlines(vp);validate_segments(features,expected);validate_segments(a_records,expected)
  assert all(r['fingerprint']==prior_config['fingerprint'] for r in features+a_records)
  for r in a_records:assert r['prediction']==base.parse_response(r['response'])
  reused.append({'feature_path':str(fp.relative_to(ROOT)),'sha256':sha(fp),'baseline_inference_path':str(vp.relative_to(ROOT)),'baseline_inference_sha256':sha(vp)})
  vectors=[r['feature'] for r in features]
  if row['original_split']=='training':labels=np.zeros(row['length'],dtype=int)
  else:
   lp=DATA/'IPAD_dataset'/scene/'test_label'/f"{int(row['video']):03d}.npy";source=next(r for r in allrows if r['id']==row['id'])
   assert sha(lp)==source['label_sha256'];labels=np.load(lp,allow_pickle=False).reshape(-1)
  assert len(labels)==row['length'] and np.isin(labels,[0,1]).all()
  for condition in ['A','B','C']:
   if condition=='A':predictions=[r['prediction'] for r in a_records]
   else:
    inference=[json.loads((out/'inference'/condition/row['original_split']/row['video']/f"{s['center']:06d}.json").read_text()) for s in expected]
    assert all(r['segment']==s for r,s in zip(inference,expected))
    predictions=[base.parse_response(r['response']) for r in inference]
   scores,detail=refine_scores(predictions,vectors,row['length']);write(out/'postprocessing'/condition/row['original_split']/(row['video']+'.json'),detail)
   counts.setdefault(condition,{})[row['id']]={'segments':len(predictions),'positive_segments':sum(predictions)}
   for frame in range(row['length']):
    r={'condition':condition,'scene':scene,'original_split':row['original_split'],'video':row['video'],'frame':frame,'label':int(labels[frame]),**{k:float(v[frame]) for k,v in scores.items()}}
    if condition=='A':
     previous=old_lookup[(row['original_split'],row['video'],frame)];assert int(previous['label'])==r['label']
     for k in ['initial','retrieved','smoothed','final']:assert abs(float(previous[k])-r[k])<1e-14,'A reconstruction differs from frozen baseline'
    output_rows.append(r)
 keys=[(r['condition'],r['original_split'],r['video'],r['frame']) for r in output_rows]
 assert len(keys)==len(set(keys))==3*sum(r['length'] for r in evaluation)
 assert all(np.isfinite(r[k]) and -1e-12<=r[k]<=1+1e-12 for r in output_rows for k in ['initial','retrieved','smoothed','final'])
 with (out/'frame_scores.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(output_rows[0]));w.writeheader();w.writerows(output_rows)
 metrics={}
 for condition in ['A','B','C']:
  rows=[r for r in output_rows if r['condition']==condition]
  metrics[condition]={k:metric(rows,k) for k in ['initial','retrieved','smoothed','final']}
  metrics[condition]['initial_binary']=binary([r['label'] for r in rows],[int(r['initial']) for r in rows])
 write(out/'metrics.json',metrics);write(out/'raw_prediction_counts.json',counts);write(out/'reused_sources.json',reused)
 calls=readlines(out/'calls.jsonl');write(out/'runtime_summary.json',{'calls':len(calls),'sum_model_seconds':sum(r['seconds'] for r in calls),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in calls),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in calls),'timing_scope':'model.chat only; excludes preprocessing/loading/hashing/logging; A/ImageBind reused'})
 write(out/'verification.json',{'status':'passed','unique_complete_frames_per_condition':len(old),'exact_shared_support':True,'baseline_scores_recomputed_and_matched':True,'normal_prompt_frozen_before_validation':True,'finite_bounded_scores_tolerance':1e-12,'same_ImageBind_features_verified':True})
 write(out/'status.json',{'status':'complete','scene':scene,'videos':len(evaluation),'frames_per_condition':len(old),'conditions':['A','B','C'],'fingerprint':frozen['fingerprint'],'finished_at':time.time()})
 print(json.dumps({'event':'evaluation_complete','scene':scene,'metrics':metrics},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True);a=p.parse_args();out=ROOT/'experiments/stage4'/a.scene/'evaluation'
 try:main(a.scene)
 except BaseException as exc:fail(out,exc);raise
