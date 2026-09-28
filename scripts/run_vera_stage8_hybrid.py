"""Validation-only hybrid weight selection; reuse fixed T1 evaluation probabilities."""
import sys,json,csv,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.vera_stage8_common import *
from scripts.vera_stage8_engine import Engine
from scripts.vera_stage8_eval import refine,summarize,metric,STAGES
import scripts.train_vera_questions as base
LAMBDA=[0.,.1,.25,.5,1.,2.]

def dino(r):
 rr=list(csv.DictReader((OUT/'dino/scores'/r['scene']/r['original_split']/(r['video']+'.csv')).open()));assert [int(x['frame']) for x in rr]==list(range(r['length']));return np.array([float(x['unconditional_nn']) for x in rr])

def vlm(r,split):
 if split=='evaluation':basepath=OUT/'step1/inference'
 else:basepath=OUT/'step5/validation_inference'
 return [json.loads((basepath/r['scene']/r['original_split']/r['video']/f"{s['center']:06d}.json").read_text())['prob'] for s in segments(r['length'])]

def main():
 base.OUT=OUT/'step5';base.guard();evaluator_guard();protected()
 protocol={'seed':0,'lambda_grid':LAMBDA,'selection':'highest validation macro initial AUROC; exact ties choose smaller lambda','z':'per-scene mean/std from ALL validation frames only; std zero ->1','initial':'native DINO per-frame z + lambda * expanded T1 PROB z','downstream':'mean hybrid score in each stride16 block, then frozen retrieval/smoothing/position; initial remains native frame formula','position_default':'OFF; full final position ON ablation also reported','source_sha256':sha(Path(__file__)),'prompt_sha256':hashlib.sha256(T1.encode()).hexdigest(),'T1_model':'reuse Stage8 Step1 frozen model and prompt','no_eval_label_selection':True,'positive_fraction_threshold':0.0,'exploratory':True}
 p=OUT/'step5/frozen.json'
 if p.exists():assert json.loads(p.read_text())==protocol
 else:write(p,protocol)
 rr=rows();validation=[r for r in rr if r['split']=='validation'];evaluation=[r for r in rr if r['split']=='evaluation'];fp=digest(protocol);engine=Engine(OUT/'step5')
 for r in validation:
  for seg in segments(r['length']):
   path=OUT/'step5/validation_inference'/r['scene']/r['original_split']/r['video']/f"{seg['center']:06d}.json"
   if path.exists():continue
   ii=images(r,seg);value=engine.call(resolve_images(ii),T1);write(path,{'request':{'video_id':r['id'],'segment':seg,'images':ii,'fingerprint':fp},'prompt':T1,**value})
  print(json.dumps({'event':'validation_vlm_video','id':r['id']}),flush=True)
 del engine;gc.collect();torch.cuda.empty_cache()
 normalization={};val={}
 for r in validation:
  d=dino(r);v=np.repeat(vlm(r,'validation'),16)[:r['length']];val[r['id']]=(d,v)
 for scene in SCENES:
  dd=np.concatenate([val[r['id']][0] for r in validation if r['scene']==scene]);vv=np.concatenate([val[r['id']][1] for r in validation if r['scene']==scene]);normalization[scene]={'dino_mean':float(dd.mean()),'dino_std':float(dd.std()) or 1.,'vlm_mean':float(vv.mean()),'vlm_std':float(vv.std()) or 1.}
 append(OUT/'events.jsonl',{'event':'step5_validation_labels_open_after_inference','time':time.time()});vy={r['id']:labels(r) for r in validation};selection=[]
 for lam in LAMBDA:
  values=[];scene_metrics={}
  for scene in SCENES:
   ns=normalization[scene];yy=[];scores=[]
   for r in validation:
    if r['scene']!=scene:continue
    d,v=val[r['id']];scores.extend(((d-ns['dino_mean'])/ns['dino_std']+lam*(v-ns['vlm_mean'])/ns['vlm_std']).tolist());yy.extend(vy[r['id']].tolist())
   scene_metrics[scene]=metric(yy,scores);values.append(scene_metrics[scene]['auroc'])
  assert all(v is not None for v in values),'Validation scene missing class'
  selection.append({'lambda':lam,'macro_auroc':float(np.mean(values)),'scene_metrics':scene_metrics})
 best=max(selection,key=lambda r:(r['macro_auroc'],-r['lambda']));locked={'protocol_fingerprint':fp,'normalization':normalization,'selection':selection,'selected_lambda':best['lambda'],'validation_ids':[r['id'] for r in validation],'evaluation_ids':[r['id'] for r in evaluation],'time':time.time()}
 lock=OUT/'step5/selection_lock.json'
 if lock.exists():
  prior=json.loads(lock.read_text());assert all(prior[k]==locked[k] for k in locked if k!='time');locked=prior
 else:write(lock,locked)
 append(OUT/'events.jsonl',{'event':'step5_selection_locked_before_evaluation_labels','selection_sha256':sha(lock),'time':time.time()});output=[];windows=[]
 for r in evaluation:
  d=dino(r);v=np.repeat(vlm(r,'evaluation'),16)[:r['length']];ns=normalization[r['scene']];dz=(d-ns['dino_mean'])/ns['dino_std'];vz=(v-ns['vlm_mean'])/ns['vlm_std'];f=features(r);ss=segments(r['length']);y=labels(r)
  for condition,raw in [('DINO',dz),('HYBRID',dz+locked['selected_lambda']*vz)]:
   xx=[float(raw[s['center']:s['score_end']].mean()) for s in ss];score=refine(xx,f,r['length']);score['initial']=raw
   output.extend({'condition':condition,'scene':r['scene'],'original_split':r['original_split'],'video':r['video'],'frame':i,'label':int(y[i]),**{s:float(w[i]) for s,w in score.items()}} for i in range(r['length']))
   windows.extend({'condition':condition,'video_id':r['id'],'center':s['center'],'raw_score':x} for s,x in zip(ss,xx))
 from scripts.analyze_vera_stage8 import load
 output += [r for r in load(OUT/'step1/frame_scores.csv') if r['condition']=='PROB']
 writecsv(OUT/'step5/frame_scores.csv',output);writecsv(OUT/'step5/window_scores.csv',windows);write(OUT/'step5/metrics.json',summarize(output));write(OUT/'step5/status.json',{'status':'complete','selected_lambda':locked['selected_lambda'],'protected_files':protected(),'test_tuning':False,'explanations':'separate pending call'});print(json.dumps({'selected':best,'macro':{c:d['macro'] for c,d in summarize(output).items()}},indent=2))
if __name__=='__main__':main()
