"""Independently audit the completed visual-reference experiment and original pixels."""
import csv,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,digest,write,readlines,segments,SCENES,DATA
from scripts.audit_vera_stage5 import verify_label_timeline
from scripts.vera_stage7_engine import parse_response,quality
from ipad.vera import refine_scores
OUT=ROOT/'experiments/stage7';CONDITIONS=['C0','N','X'];ALL=['P1',*CONDITIONS];STAGES=['initial','retrieved','smoothed','final']

def main():
 status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete'
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(DATA/'cache/vera/InternVL2-8B',trust_remote_code=True,local_files_only=True,use_fast=False)
 frozen=json.loads((OUT/'frozen.json').read_text());value=dict(frozen);value.pop('fingerprint');assert digest(value)==frozen['fingerprint']
 for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h,p
 protect=json.loads((OUT/'protected_previous.json').read_text())['files'];actual={str(p.relative_to(ROOT)):sha(p) for stage in ['stage4','stage5','stage6'] for p in (ROOT/'experiments'/stage).rglob('*') if p.is_file()};assert actual==protect
 rows=json.loads((ROOT/'experiments/stage2_1/split.json').read_text())['records'];byid={r['id']:r for r in rows};manifest=json.loads((OUT/'inference_manifest.json').read_text());refs=json.loads((OUT/'references.json').read_text());prompts=json.loads((OUT/'prompts.json').read_text())
 assert frozen['split_sha256']==sha(ROOT/'experiments/stage2_1/split.json')
 evaluation=[r for r in rows if r['split']=='evaluation' and r['scene'] in SCENES]
 assert [r['id'] for r in manifest['evaluation']]==[r['id'] for r in evaluation]==frozen['evaluation_ids']
 assert all(set(r)=={'id','scene','original_split','video','length','relative_path','frames_sha256','frame_ids'} for r in manifest['evaluation']+manifest['preflight_normal_train'])
 refchecks=[];image_hashes={}
 def check_image(video_id,frame_id,h):
  row=byid[video_id];assert row['frame_ids'][frame_id]==frame_id and row['frames_sha256'][frame_id]==h
  key=(video_id,frame_id)
  if key not in image_hashes:
   folder=DATA/row['relative_path'];p=folder/f'{frame_id:03d}.jpg'
   if not p.exists():p=next(p for p in folder.glob('*.jpg') if int(p.stem)==frame_id)
   assert sha(p)==h;image_hashes[key]=True
 for scene in SCENES:
  normal=sorted([r for r in rows if r['scene']==scene and r['split']=='train' and r['video_label']==0],key=lambda r:r['id'])
  assert len(refs['references'][scene])==4 and len({r['video_id'] for r in refs['references'][scene]})==4
  for i,ref in enumerate(refs['references'][scene]):
   row=normal[int(np.floor((i+.5)*len(normal)/4))];fi=int(np.floor((row['length']-1)*(i+.5)/4))
   assert ref['video_id']==row['id'] and ref['frame_id']==fi and ref['reference']==f'Ref{i+1}'
   assert row['id'] not in frozen['evaluation_ids'];check_image(row['id'],fi,ref['frame_sha256']);refchecks.append({'video_id':row['id'],'frame_id':fi,'normal_train':True})
  pre=next(r for r in manifest['preflight_normal_train'] if r['scene']==scene)
  assert byid[pre['id']]['split']=='train' and byid[pre['id']]['video_label']==0
  assert pre['id'] not in {r['video_id'] for r in refs['references'][scene]} and pre['id'] not in frozen['evaluation_ids']
  assert prompts[scene]['N']==prompts[scene]['X']
 scores=list(csv.DictReader((OUT/'frame_scores.csv').open()));keys=[(r['condition'],r['scene'],r['original_split'],r['video'],int(r['frame'])) for r in scores]
 expected={(c,r['scene'],r['original_split'],r['video'],i) for c in ALL for r in evaluation for i in range(r['length'])}
 assert len(keys)==len(set(keys)) and set(keys)==expected;index=dict(zip(keys,scores))
 old={(r['scene'],r['original_split'],r['video'],int(r['frame'])):r for r in csv.DictReader((ROOT/'experiments/stage5/frame_scores.csv').open()) if r['condition']=='P1'}
 windows=json.loads((OUT/'window_diagnostics.json').read_text());wi={(r['condition'],r['video_id'],r['center']):r for r in windows};assert len(wi)==len(windows)==4*sum(len(segments(r['length'])) for r in evaluation)
 features={r['video_id']:r for r in json.loads((OUT/'reused_features.json').read_text())};times=[];count=0;format_counts={c:0 for c in CONDITIONS};token_lengths={c:[] for c in CONDITIONS}
 rotation=dict(zip(SCENES,['R02','R03','R04','R01']))
 for row in evaluation:
  scene=row['scene'];part=row['original_split'];vid=row['video'];segs=segments(row['length'])
  if part=='training':y=np.zeros(row['length'],dtype=int)
  else:
   p=DATA/'IPAD_dataset'/scene/'test_label'/f'{int(vid):03d}.npy';assert sha(p)==row['label_sha256'];y=np.load(p,allow_pickle=False).reshape(-1)
  f=features[row['id']];assert f['reference_images_in_features'] is False and sha(ROOT/f['feature_path'])==f['feature_sha256']
  feat=readlines(ROOT/f['feature_path']);assert [{k:r[k] for k in segs[0]} for r in feat]==segs;vectors=[r['feature'] for r in feat]
  for c in ALL:
   pred=[]
   if c=='P1':
    result={s:np.array([float(old[(scene,part,vid,i)][s]) for i in range(row['length'])]) for s in STAGES};pred=[int(result['initial'][s['center']]) for s in segs]
   else:
    for seg in segs:
     p=OUT/'inference'/c/scene/part/vid/f"{seg['center']:06d}.json";r=json.loads(p.read_text());q=r['request'];count+=1
     assert q['condition']==c and q['video_id']==row['id'] and q['segment']==seg and q['prompt']==prompts[scene][c] and q['fingerprint']==frozen['fingerprint'] and r['cache_key']==digest(q)
     refimages=[] if c=='C0' else refs['references'][scene if c=='N' else rotation[scene]]
     assert len(q['images'])==8+len(refimages) and q['num_patches_list']==[1]*len(q['images'])
     for i,ref in enumerate(refimages):
      im=q['images'][i];assert im=={'role':'reference','name':f'Ref{i+1}','video_id':ref['video_id'],'relative_path':ref['relative_path'],'frame_id':ref['frame_id'],'sha256':ref['frame_sha256']};check_image(im['video_id'],im['frame_id'],im['sha256'])
     for i,frame in enumerate(seg['frame_ids']):
      im=q['images'][len(refimages)+i];assert im=={'role':'query','name':f'Frame{i+1}','video_id':row['id'],'relative_path':row['relative_path'],'frame_id':frame,'sha256':row['frames_sha256'][frame]};check_image(row['id'],frame,im['sha256'])
     assert r['parse_status']=='valid' and r['prediction']==parse_response(r['response']);pred.append(r['prediction']);times.append(r['time']);token_lengths[c].append(r['response_retokenized_tokens'])
     assert r['actual_generation_config']==frozen['generation']
     assert len(r['generated_token_ids'])==r['generated_token_count']<=128 and r['ended_with_eos']==(r['generated_token_ids'][-1] in r['eos_token_ids'])
     assert r['explanation_quality']==quality(r['response'],r)
     assert tokenizer.decode(r['generated_token_ids'],skip_special_tokens=True).strip()==r['response']
     format_counts[c]+=int(not r['explanation_quality']['evidence_missing'])
    result,detail=refine_scores(pred,vectors,row['length']);assert detail==json.loads((OUT/'postprocessing'/c/scene/part/(vid+'.json')).read_text())
   for i in range(row['length']):
    r=index[(c,scene,part,vid,i)];assert int(r['label'])==int(y[i])
    for stage in STAGES:assert abs(float(r[stage])-result[stage][i])<1e-14
   for seg,prediction in zip(segs,pred):
    w=wi[(c,row['id'],seg['center'])];assert w['prediction']==prediction and w['frame_ids']==seg['frame_ids']
    assert w['sampled_positive_frames']==int(y[seg['frame_ids']].sum()) and w['scored_positive_frames']==int(y[seg['center']:seg['score_end']].sum())
 passes=verify_label_timeline(readlines(OUT/'events.jsonl'),times,[r['id'] for r in evaluation]);metrics=json.loads((OUT/'metrics.json').read_text())
 for scene in [*SCENES,'pooled']:
  for c in ALL:
   part=[r for r in scores if r['condition']==c and (scene=='pooled' or r['scene']==scene)];y=[int(r['label']) for r in part]
   for stage in STAGES:
    x=[float(r[stage]) for r in part];assert np.isfinite(x).all();m=metrics[scene][c][stage]
    assert abs(m['auroc']-100*roc_auc_score(y,x))<1e-10 and abs(m['ap']-100*average_precision_score(y,x))<1e-10
   tn,fp,fn,tp=confusion_matrix(y,[int(float(r['initial'])) for r in part],labels=[0,1]).ravel();assert all(metrics[scene][c]['initial_binary'][k]==int(v) for k,v in zip(['tn','fp','fn','tp'],[tn,fp,fn,tp]))
 for c in ALL:
  for stage in STAGES:
   for k in ['auroc','ap']:assert abs(metrics['macro'][c][stage][k]-np.mean([metrics[s][c][stage][k] for s in SCENES]))<1e-10
 result={'status':'passed','inference_calls_verified':count,'videos':len(evaluation),'frames_per_condition':len(scores)//4,'reference_images_verified':refchecks,'all_reference_images_normal_current_train':True,'reference_images_not_in_legacy_features':True,'N_X_byte_identical_prompts':True,'protected_previous_files_unchanged':len(protect),'raw_image_hashes_verified':len(image_hashes),'baseline_P1_exact':True,'all_inference_before_label_open':True,'label_passes':passes,'postprocessing_reconstructed':True,'all_metrics_recomputed':True,'saved_generation_tokens_decode_to_raw_responses':True,'requested_fields_present':format_counts,'token_lengths':{c:{'mean':float(np.mean(v)),'max':max(v),'possible_limit':sum(t>=128 for t in v)} for c,v in token_lengths.items()},'audit_source_sha256':sha(Path(__file__))}
 quality_report={}
 for c in CONDITIONS:
  records=[r for r in readlines(OUT/'calls.jsonl') if r['request']['condition']==c and r['request']['video_id'] in frozen['evaluation_ids']]
  quality_report[c]={'calls':len(records),'binary_valid':sum(r['parse_status']=='valid' for r in records),'actual_generated_tokens_mean':float(np.mean([r['generated_token_count'] for r in records])),'EOS_ended':sum(r['ended_with_eos'] for r in records),**{k:sum(r['explanation_quality'][k] for r in records) for k in ['duplicate_output','evidence_missing','evidence_over40words','evidence_repeated_sentence','explanation_truncated']},'semantic_explanation_validation':'not systematically assessed; response text is model claim'}
 write(OUT/'response_quality.json',quality_report)
 write(OUT/'independent_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
