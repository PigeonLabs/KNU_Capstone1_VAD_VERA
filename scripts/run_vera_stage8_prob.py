import argparse,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.vera_stage8_common import *
from scripts.vera_stage8_eval import refine,summarize,STAGES,metric
from scripts.vera_stage8_engine import Engine
from scripts.vera_stage4_common import seed_everything,fail
import scripts.train_vera_questions as base

def setup():
 protected();evaluator_guard();base.OUT=OUT;base.guard();seed_everything(0)
 sources=['scripts/run_vera_stage8_prob.py','scripts/vera_stage8_engine.py','scripts/vera_stage8_common.py','scripts/vera_stage8_eval.py','ipad/vera_models.py','scripts/train_vera_questions.py','experiments/stage8/protocol.json','experiments/stage8/evaluator_lock.json','experiments/stage2_1/split.json']
 frozen={'source_sha256':{s:sha(ROOT/s) for s in sources},'prompt':T1,'prompt_sha256':hashlib.sha256(T1.encode()).hexdigest(),'generation':{'max_new_tokens':1,'num_beams':1,'do_sample':False},'config':vars(CONFIG),'seed':0}
 frozen['fingerprint']=digest(frozen);p=OUT/'step1/frozen.json'
 if p.exists():assert json.loads(p.read_text())==frozen
 else:write(p,frozen)
 inv=json.loads((ROOT/'experiments/vera_ipad/model_inventory.json').read_text())
 for r in inv:base.guard();assert sha(DATA/r['path'])==r['sha256']
 write(OUT/'step1/model_verification.json',{'status':'passed','files':inv})
 write(OUT/'step1/environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'parameter_updates':0})
 return frozen

def predict(engine,r,seg,frozen,phase):
 ii=images(r,seg);request={'video_id':r['id'],'segment':seg,'images':ii,'prompt_sha256':frozen['prompt_sha256'],'fingerprint':frozen['fingerprint']}
 p=OUT/'step1'/phase/r['scene']/r['original_split']/r['video']/f"{seg['center']:06d}.json"
 if p.exists():
  value=json.loads(p.read_text());assert value['request']==request;return value
 value={'request':request,'condition':'T1_DEC_PROB','prompt':T1,**engine.call(resolve_images(ii),T1),'time':time.time()};write(p,value)
 append(OUT/'step1/calls.jsonl',{'path':str(p.relative_to(ROOT)),'video_id':r['id'],'center':seg['center'],'prob':value['prob'],'dec':value['dec'],'parse_status':value['parse_status'],'seconds':value['seconds']})
 return value

def evaluate(evaluation):
 append(OUT/'events.jsonl',{'event':'step1_all_inference_complete_before_labels','time':time.time()});output=[];windows=[];failures=[]
 for r in evaluation:
  ss=segments(r['length']);rr=[json.loads((OUT/'step1/inference'/r['scene']/r['original_split']/r['video']/f"{s['center']:06d}.json").read_text()) for s in ss]
  y=labels(r);f=features(r)
  for c,key in [('PROB','prob'),('DEC','dec')]:
   if any(v[key] is None for v in rr):
    failures.extend({'video_id':r['id'],'center':s['center'],'response':v['response']} for s,v in zip(ss,rr) if v[key] is None);continue
   x=[v[key] for v in rr];score=refine(x,f,r['length'])
   output.extend({'condition':c,'scene':r['scene'],'original_split':r['original_split'],'video':r['video'],'frame':i,'label':int(y[i]),**{s:float(v[i]) for s,v in score.items()}} for i in range(r['length']))
  for s,v in zip(ss,rr):windows.append({'video_id':r['id'],'scene':r['scene'],'center':s['center'],'prob':v['prob'],'dec':v['dec'],'response':v['response'],'seconds':v['seconds']})
 write(OUT/'step1/dec_failures.json',failures)
 # DEC incomplete support is never silently used in a paired comparison.
 if failures:output=[r for r in output if r['condition']!='DEC']
 writecsv(OUT/'step1/frame_scores.csv',output);writecsv(OUT/'step1/window_scores.csv',windows);m=summarize(output);write(OUT/'step1/metrics.json',m)
 no={(r['video_id'],r['center']) for r in windows if r['dec']==0};subset=[r for r in output if r['condition']=='PROB' and (f"{r['scene']}/{r['original_split']}/{r['video']}",r['frame']//16*16) in no]
 no_metrics={sc:metric([r['label'] for r in subset if sc=='pooled' or r['scene']==sc],[r['initial'] for r in subset if sc=='pooled' or r['scene']==sc]) for sc in [*SCENES,'pooled']}
 write(OUT/'step1/decoded_no_subset.json',{'frames':len(subset),'metrics':no_metrics})
 diag={k:{'n_unique_scores':len({r[k] for r in windows if r[k] is not None}),'positive_fraction':sum(r[k]>=.5 for r in windows if r[k] is not None)/sum(r[k] is not None for r in windows)} for k in ['prob','dec']}
 write(OUT/'step1/score_diagnostics.json',diag);auc=m['PROB']['macro']['initial']['auroc'];gate='reference_and_decomposition' if auc>=55 else 'hybrid'
 write(OUT/'step1/branch.json',{'initial_macro_auroc':auc,'route':gate,'in_user_47_53_band':47<=auc<=53,'causal_interpretation':'A routing gate, not proof of a visual perception limitation','test_driven_branch':'prespecified by user; exploratory'})
 print(json.dumps({'macro':{c:v['macro'] for c,v in m.items()},'diagnostics':diag,'branch':gate},indent=2));write(OUT/'step1/status.json',{'status':'complete','videos':len(evaluation),'frames':sum(r['length'] for r in evaluation),'dec_complete':not failures,'protected_files':protected(),'finished_at':time.time()})

def main(phase):
 frozen=setup();allrows=rows();engine=Engine(OUT/'step1')
 if phase=='preflight':
  for sc in SCENES:
   r=next(r for r in allrows if r['scene']==sc and r['split']=='train' and r['video_label']==0)
   for s in [segments(r['length'])[0],segments(r['length'])[-1]]:
    result=predict(engine,r,s,frozen,'preflight');print(json.dumps({'scene':sc,'center':s['center'],'response':result['response'],'prob':result['prob']}),flush=True)
  write(OUT/'step1/preflight_summary.json',{'status':'complete','calls':8,'labels_opened':False});return
 assert json.loads((OUT/'step1/preflight_summary.json').read_text())['status']=='complete'
 evaluation=[r for r in allrows if r['split']=='evaluation'];write(OUT/'step1/status.json',{'status':'running'})
 for r in evaluation:
  for s in segments(r['length']):predict(engine,r,s,frozen,'inference')
  print(json.dumps({'event':'video_complete','video_id':r['id']}),flush=True)
 del engine;gc.collect();torch.cuda.empty_cache();evaluate(evaluation)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','evaluation'],required=True);a=p.parse_args()
 try:main(a.phase)
 except BaseException as e:fail(OUT/'step1',e);raise
