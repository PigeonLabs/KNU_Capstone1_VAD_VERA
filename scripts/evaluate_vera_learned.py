"""Evaluate frozen selected questions on held-out IPAD videos only."""
import argparse,csv,gc,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from ipad.common import seed_everything
from ipad.vera import VeraConfig,refine_scores,segments
from ipad.vera_models import ImageBind
import scripts.train_vera_questions as training
parse_response=training.parse_response
from scripts.evaluate_vera import summarize
OUT=ROOT/'experiments/stage3';DATA=Path('/home/jeong/Desktop/IPAD');CACHE=DATA/'cache/vera';FEATURES=ROOT/'cache/stage3/features'
sha=training.sha;write=training.write;append=training.append;readlines=training.readlines

def guard():
 if (ROOT/'runs/disk_pause.json').exists() or (OUT/'disk_pause.json').exists():raise RuntimeError('Latched disk pause requires explicit user resume')
 free=shutil.disk_usage(ROOT).free
 if free<=10*1024**3:
  write(OUT/'disk_pause.json',{'status':'paused_low_disk','time':time.time(),'free_bytes':free,'automatic_resume':False});raise RuntimeError('Disk reserve reached')
def video_path(row,base,suffix):return base/row['scene']/row['original_split']/(row['video']+suffix)
def validate_rows(rows,expected,fingerprint):
 assert len(rows)<=len(expected)
 for record,segment in zip(rows,expected):
  assert record['fingerprint']==fingerprint
  assert all(record[k]==segment[k] for k in segment)

def main():
 OUT.mkdir(parents=True,exist_ok=True);FEATURES.mkdir(parents=True,exist_ok=True);guard();seed_everything(0)
 model_inventory=json.loads((ROOT/'experiments/vera_ipad/model_inventory.json').read_text())
 for item in model_inventory:
  guard();assert sha(DATA/item['path'])==item['sha256'],item['path']
 write(OUT/'model_verification.json',{'status':'passed','checked_at':time.time(),'files':model_inventory})
 upstream_inventory=json.loads((ROOT/'experiments/vera_ipad/source_inventory.json').read_text())
 for item in upstream_inventory['files']:
  path=Path(item['path']);base=DATA if path.parts[0]=='cache' else ROOT
  assert sha(base/path)==item['sha256'],str(path)
 write(OUT/'upstream_source_verification.json',{'status':'passed','checked_at':time.time(),'inventory':upstream_inventory})
 selection_path=ROOT/'experiments/stage2_3/selection.json';selection=json.loads(selection_path.read_text())
 assert json.loads((ROOT/'experiments/stage2_3/status.json').read_text())['status']=='complete'
 question_path=ROOT/'experiments/stage2_3/questions.txt';assert sha(question_path)==selection['questions_sha256'];questions=question_path.read_text()
 split_path=ROOT/'experiments/stage2_1/split.json';assert sha(split_path)==selection['split_sha256'];split=json.loads(split_path.read_text())
 # Prediction receives sanitized metadata without label values or label-file paths.
 allowed=['id','scene','original_split','video','length','relative_path','frames_sha256','frame_ids']
 rows=[{k:r[k] for k in allowed} for r in split['records'] if r['split']=='evaluation'];config=VeraConfig()
 source_paths=['scripts/evaluate_vera_learned.py','scripts/train_vera_questions.py','scripts/evaluate_vera.py','ipad/vera.py','ipad/vera_models.py','experiments/vera_ipad/source_reference/VERA/VERA_learner_instruct.txt','experiments/vera_ipad/source_reference/lavad_data.py']
 frozen={'questions':questions,'questions_sha256':sha(question_path),'selection_sha256':sha(selection_path),'split_sha256':sha(split_path),
  'config':vars(config),'source_sha256':{p:sha(ROOT/p) for p in source_paths},'videos':[r['id'] for r in rows],
  'feature_policy':'recompute ImageBind FP32 for all held-out videos; no reuse of stage1 test feature caches',
  'model_inventory_sha256':sha(ROOT/'experiments/vera_ipad/model_inventory.json'),'upstream_inventory_sha256':sha(ROOT/'experiments/vera_ipad/source_inventory.json'),'labels':'opened only after all VLM and ImageBind inference completes'}
 frozen['fingerprint']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest();fingerprint=frozen['fingerprint']
 if (OUT/'frozen.json').exists():assert json.loads((OUT/'frozen.json').read_text())==frozen
 else:write(OUT/'frozen.json',frozen);write(OUT/'inference_manifest.json',rows)
 write(OUT/'status.json',{'status':'running','phase':'vlm','fingerprint':fingerprint})
 append(OUT/'events.jsonl',{'event':'invocation','time':time.time(),'argv':sys.argv,'executable':sys.executable,'fingerprint':fingerprint})
 training.OUT=OUT  # Model load events/disk latch belong to this run, not completed training.
 engine=training.Engine()
 for row in rows:
  expected=segments(row['length'],config);path=video_path(row,OUT/'inference','.jsonl');previous=readlines(path);validate_rows(previous,expected,fingerprint)
  for record in previous:assert parse_response(record['response'])==record['prediction']
  for seg in expected[len(previous):]:
   guard();files=training.files_for(row,seg['frame_ids']);prompt=training.learner_prompt(engine.template,questions)
   response,stats=engine.call(files,prompt)
   raw={**seg,'fingerprint':fingerprint,'video_id':row['id'],'response':response,**stats}
   try:raw['prediction']=parse_response(response)
   except ValueError:
    append(OUT/'parsing_failures.jsonl',raw);raise
   append(path,raw)
  print(json.dumps({'event':'vlm_video_complete','video':row['id'],'segments':len(expected)}),flush=True)
 del engine;gc.collect();torch.cuda.empty_cache();guard()
 write(OUT/'status.json',{'status':'running','phase':'imagebind','fingerprint':fingerprint})
 feature_engine=ImageBind(CACHE,ROOT/'experiments/vera_ipad/source_reference')
 for row in rows:
  expected=segments(row['length'],config);path=video_path(row,FEATURES,'.jsonl');previous=readlines(path);validate_rows(previous,expected,fingerprint)
  files=training.files_for(row,list(range(row['length'])))
  for seg in expected[len(previous):]:
   guard();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   feature,ids=feature_engine.encode(files,seg,config.imagebind_samples);torch.cuda.synchronize()
   assert np.isfinite(feature).all() and np.linalg.norm(feature)>0
   append(path,{**seg,'fingerprint':fingerprint,'video_id':row['id'],'feature':feature.tolist(),'imagebind_frame_ids':ids,
    'seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3})
  print(json.dumps({'event':'feature_video_complete','video':row['id'],'segments':len(expected)}),flush=True)
 del feature_engine;gc.collect();torch.cuda.empty_cache();guard()
 write(OUT/'status.json',{'status':'running','phase':'metrics','fingerprint':fingerprint})
 aligned=[];inventory=[];counts={};timing={'vlm':[],'imagebind':[]};label_inventory=[]
 for row in rows:
  vlm_path=video_path(row,OUT/'inference','.jsonl');feature_path=video_path(row,FEATURES,'.jsonl')
  vlm=readlines(vlm_path);features=readlines(feature_path);expected=segments(row['length'],config)
  assert len(vlm)==len(features)==len(expected);validate_rows(vlm,expected,fingerprint);validate_rows(features,expected,fingerprint)
  initial=[r['prediction'] for r in vlm];vectors=[r['feature'] for r in features];scores,details=refine_scores(initial,vectors,row['length'],config)
  write(video_path(row,OUT/'postprocessing','.json'),details)
  # Only now open frame labels, independently from prediction inputs.
  if row['original_split']=='training':labels=np.zeros(row['length'],dtype=int)
  else:
   lp=DATA/'IPAD_dataset'/row['scene']/'test_label'/f"{int(row['video']):03d}.npy"
   labels=np.load(lp,allow_pickle=False).reshape(-1);original=next(r for r in split['records'] if r['id']==row['id'])
   assert sha(lp)==original['label_sha256'];label_inventory.append({'path':str(lp.relative_to(DATA)),'sha256':sha(lp)})
  assert len(labels)==row['length'] and np.isin(labels,[0,1]).all()
  for frame in range(row['length']):aligned.append({'scene':row['scene'],'original_split':row['original_split'],'video':row['video'],'frame':frame,'label':int(labels[frame]),**{k:float(v[frame]) for k,v in scores.items()}})
  counts[row['id']]={'segments':len(vlm),'positive_segments':sum(initial),'frames':row['length']}
  inventory.extend([{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size,'local_only':p==feature_path} for p in [vlm_path,feature_path]])
  for name,records in [('vlm',vlm),('imagebind',features)]:timing[name].extend({k:r[k] for k in ['seconds','peak_allocated_gib','peak_reserved_gib']} for r in records)
  write(video_path(row,OUT/'feature_metadata','.json'),[{k:v for k,v in r.items() if k!='feature'} for r in features])
 keys=[(r['scene'],r['original_split'],r['video'],r['frame']) for r in aligned]
 assert len(set(keys))==len(keys)==sum(r['length'] for r in rows)
 assert all(np.isfinite(r[k]) and -1e-12<=r[k]<=1+1e-12 for r in aligned for k in ['initial','retrieved','smoothed','final'])
 with (OUT/'frame_scores.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(aligned[0]));writer.writeheader();writer.writerows(aligned)
 metrics={k:summarize(aligned,k) for k in ['initial','retrieved','smoothed','final']};write(OUT/'metrics.json',metrics)
 write(OUT/'raw_prediction_counts.json',counts);write(OUT/'inference_inventory.json',inventory);write(OUT/'label_inventory.json',label_inventory)
 write(OUT/'runtime_summary.json',{name:{'timing_scope':('model.chat after image preprocessing and tensor transfer' if name=='vlm' else 'ImageBind.encode including its frame preprocessing; excluding whole-video file hash audit'),'segments':len(a),'sum_seconds':sum(r['seconds'] for r in a),'mean_seconds':float(np.mean([r['seconds'] for r in a])),
  'p95_seconds':float(np.percentile([r['seconds'] for r in a],95)),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in a),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in a)} for name,a in timing.items()})
 write(OUT/'verification.json',{'status':'passed','unique_complete_frames':len(aligned),'videos':len(rows),'finite_bounded_scores':True,'range_tolerance':1e-12,'selection_frozen_before_inference':True,
 'evaluation_ids_disjoint_from_train_validation':not({r['id'] for r in rows}&{r['id'] for r in split['records'] if r['split']!='evaluation'})})
 write(OUT/'status.json',{'status':'complete','fingerprint':fingerprint,'videos':len(rows),'frames':len(aligned),'finished_at':time.time()})
 print(json.dumps({'event':'evaluation_complete','metrics':metrics},indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  OUT.mkdir(parents=True,exist_ok=True);append(OUT/'failures.jsonl',{'time':time.time(),'error':repr(exc),'traceback':traceback.format_exc()})
  old=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
  write(OUT/'status.json',{**old,'status':'paused_low_disk' if (OUT/'disk_pause.json').exists() or (ROOT/'runs/disk_pause.json').exists() else 'failed','error':repr(exc)});raise
