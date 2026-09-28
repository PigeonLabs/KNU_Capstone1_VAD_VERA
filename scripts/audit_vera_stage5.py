"""Independent completeness, frozen-input and score reconstruction audit for Stage5."""
import csv,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,digest,write,readlines,SCENES,segments,DATA
from scripts.train_vera_questions import parse_response
from ipad.vera import refine_scores
OUT=ROOT/'experiments/stage5'

def verify_label_timeline(events,inference_times,video_ids):
 gates=[e for e in events if e['event']=='all_evaluation_inference_validated_before_labels']
 opens=[e for e in events if e['event']=='evaluation_labels_open']
 assert gates and opens
 assert max(inference_times)<=min(e['time'] for e in gates)
 assert min(e['time'] for e in opens)>=gates[0]['time']
 for i,gate in enumerate(gates):
  end=gates[i+1]['time'] if i+1<len(gates) else float('inf')
  batch=[e for e in opens if gate['time']<=e['time']<end]
  assert len(batch)==len(video_ids) and {e['video_id'] for e in batch}==set(video_ids)
 return len(gates)

def main():
 status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete'
 frozen=json.loads((OUT/'frozen.json').read_text());identity=dict(frozen);identity.pop('fingerprint');assert digest(identity)==frozen['fingerprint']
 for path,h in frozen['source_sha256'].items():assert sha(ROOT/path)==h,path
 protection=json.loads((OUT/'protected_stage4.json').read_text())['files']
 actual={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/stage4').rglob('*') if p.is_file()};assert actual==protection
 prompts=json.loads((OUT/'prompts.json').read_text());manifest=json.loads((OUT/'inference_manifest.json').read_text())
 split=json.loads((ROOT/'experiments/stage2_1/split.json').read_text())['records'];assert frozen['split_sha256']==sha(ROOT/'experiments/stage2_1/split.json')
 expected=[r for r in split if r['split']=='evaluation' and r['scene'] in SCENES]
 assert [r['id'] for r in manifest['evaluation']]==[r['id'] for r in expected]==frozen['evaluation_ids']
 assert all(set(r)=={'id','scene','original_split','video','length','relative_path','frames_sha256','frame_ids'} for r in manifest['evaluation']+manifest['preflight_normal_train'])
 assert not set(frozen['evaluation_ids'])&set(frozen['preflight_ids'])
 assert all(next(r for r in split if r['id']==v)['split']=='train' and next(r for r in split if r['id']==v)['video_label']==0 for v in frozen['preflight_ids'])
 frame_rows=list(csv.DictReader((OUT/'frame_scores.csv').open()));keys=[(r['condition'],r['scene'],r['original_split'],r['video'],int(r['frame'])) for r in frame_rows]
 wanted={(c,r['scene'],r['original_split'],r['video'],i) for c in ['A','B','C','P1','P2','P3'] for r in expected for i in range(r['length'])};assert len(keys)==len(set(keys)) and set(keys)==wanted
 frame_index={k:v for k,v in zip(keys,frame_rows)};windows=json.loads((OUT/'window_diagnostics.json').read_text());wi={(r['condition'],r['video_id'],r['center']):r for r in windows}
 reuse={r['video_id']:r for r in json.loads((OUT/'reused_features.json').read_text())}
 inference_times=[];inference_count=0;image_checks={};tokens={c:[] for c in ['P1','P2','P3']};format_counts={c:{'records':0,'all_requested_fields_present':0} for c in ['P1','P2','P3']}
 for video in expected:
  scene=video['scene'];vid=video['video'];original=video['original_split'];segs=segments(video['length'])
  if original=='training':labels=np.zeros(video['length'],dtype=int)
  else:
   p=DATA/'IPAD_dataset'/scene/'test_label'/f'{int(vid):03d}.npy';assert sha(p)==video['label_sha256'];labels=np.load(p,allow_pickle=False).reshape(-1)
  f=reuse[video['id']];assert sha(ROOT/f['feature_path'])==f['feature_sha256'];features=readlines(ROOT/f['feature_path'])
  assert [{k:r[k] for k in segs[0]} for r in features]==segs
  vectors=[r['feature'] for r in features]
  baseline=list(csv.DictReader((ROOT/f'experiments/stage4/{scene}/evaluation/frame_scores.csv').open()))
  bi={(r['condition'],int(r['frame'])):r for r in baseline if r['original_split']==original and r['video']==vid}
  for c in ['A','B','C','P1','P2','P3']:
   pred=[]
   if c.startswith('P'):
    prompt=prompts[scene][c];assert digest(prompt)==frozen['prompt_sha256'][scene][c]
    for seg in segs:
     p=OUT/'inference'/c/scene/original/vid/f"{seg['center']:06d}.json";r=json.loads(p.read_text());inference_count+=1
     assert r['segment']==seg and r['video_id']==video['id'] and r['prompt']==prompt and r['fingerprint']==frozen['fingerprint']
     assert r['request_hash']==digest({k:r[k] for k in ['prompt','video_id','segment']})
     assert r['input_frame_sha256']==[video['frames_sha256'][i] for i in seg['frame_ids']]
     assert r['cache_key']==digest({'fingerprint':frozen['fingerprint'],'prompt':prompt,'frame_sha256':r['input_frame_sha256'],'segment':seg,'video_id':video['id']})
     for i in seg['frame_ids']:
      key=(video['id'],i)
      if key not in image_checks:
       file=DATA/video['relative_path']/f'{i}.jpg'
       if not file.exists():file=next(p for p in (DATA/video['relative_path']).glob('*.jpg') if int(p.stem)==i)
       assert sha(file)==video['frames_sha256'][i];image_checks[key]=True
     fields=['Observed changes:','Evidence consistent:','Evidence conflicting:','Uncertainty:'] if c=='P3' else ['Observations:','Assessment:']
     plain=r['response'].replace('**','').lower();format_counts[c]['records']+=1;format_counts[c]['all_requested_fields_present']+=int(all(field.lower() in plain for field in fields))
     assert r['parse_status']=='valid';value=parse_response(r['response']);assert value==r['prediction'];pred.append(value);inference_times.append(r['time']);tokens[c].append(r['response_retokenized_tokens'])
    scores,detail=refine_scores(pred,vectors,video['length'])
    assert detail==json.loads((OUT/'postprocessing'/c/scene/original/(vid+'.json')).read_text())
   else:
    scores={stage:np.array([float(bi[(c,i)][stage]) for i in range(video['length'])]) for stage in ['initial','retrieved','smoothed','final']};pred=[int(scores['initial'][s['center']]) for s in segs]
   for i in range(video['length']):
    r=frame_index[(c,scene,original,vid,i)];assert int(r['label'])==int(labels[i])
    for stage in scores:assert abs(float(r[stage])-scores[stage][i])<1e-14
   for seg,prediction in zip(segs,pred):
    w=wi[(c,video['id'],seg['center'])];assert w['prediction']==prediction and w['frame_ids']==seg['frame_ids']
    assert w['sampled_positive_frames']==int(labels[seg['frame_ids']].sum()) and w['scored_positive_frames']==int(labels[seg['center']:seg['score_end']].sum())
 label_passes=verify_label_timeline(readlines(OUT/'events.jsonl'),inference_times,[r['id'] for r in expected])
 metrics=json.loads((OUT/'metrics.json').read_text())
 for scene in [*SCENES,'pooled']:
  for c in ['A','B','C','P1','P2','P3']:
   rows=[r for r in frame_rows if r['condition']==c and (scene=='pooled' or r['scene']==scene)];y=[int(r['label']) for r in rows]
   for stage in ['initial','retrieved','smoothed','final']:
    x=[float(r[stage]) for r in rows];assert np.isfinite(x).all();m=metrics[scene][c][stage]
    assert abs(m['auroc']-100*roc_auc_score(y,x))<1e-10 and abs(m['ap']-100*average_precision_score(y,x))<1e-10
   tn,fp,fn,tp=confusion_matrix(y,[int(float(r['initial'])) for r in rows],labels=[0,1]).ravel()
   assert all(metrics[scene][c]['initial_binary'][k]==int(v) for k,v in zip(['tn','fp','fn','tp'],[tn,fp,fn,tp]))
 for c in metrics['macro']:
  for stage in metrics['macro'][c]:
   for key in ['auroc','ap']:assert abs(metrics['macro'][c][stage][key]-np.mean([metrics[s][c][stage][key] for s in SCENES]))<1e-10
 result={'status':'passed','inference_calls_verified':inference_count,'videos':len(expected),'frames_per_condition':len(frame_rows)//6,'raw_image_hashes_verified':len(image_checks),'protected_stage4_files_unchanged':len(protection),'all_inference_before_label_open':True,'postprocessing_passes_verified':label_passes,'cached_baselines_exact':True,'postprocessing_reconstructed':True,'four_score_stage_metrics_recomputed':True,'window_label_alignment_verified':True,'source_model_config_frozen':True,'requested_output_field_diagnostics':format_counts,'token_lengths':{c:{'mean':float(np.mean(v)),'max':max(v),'possible_limit':sum(t>=1024 for t in v)} for c,v in tokens.items()}}
 write(OUT/'independent_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
