"""Prespecified 2/4 second temporal window ablation; labels opened after all inference."""
import sys,json,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dataclasses import replace
from scripts.vera_stage8_common import *
from scripts.vera_stage8_engine import Engine
from scripts.vera_stage8_eval import refine,summarize
from ipad.vera_models import ImageBind
import scripts.train_vera_questions as base

def main():
 from ipad.common import seed_everything
 seed_everything(0)
 base.OUT=OUT/'step4';base.guard();evaluator_guard();protected()
 inventory=json.loads((ROOT/'experiments/vera_ipad/model_inventory.json').read_text())
 for item in inventory:base.guard();assert sha(DATA/item['path'])==item['sha256']
 write(OUT/'step4/model_verification.json',{'status':'passed','files':inventory})
 rr=[r for r in rows() if r['split']=='evaluation'];protocol={'windows_seconds':[2,4,10],'fps':30,'actual_fps_unverified':True,'seed':0,'torch_initial_seed':torch.initial_seed(),'prompt':T1,'prompt_sha256':hashlib.sha256(T1.encode()).hexdigest(),'source_sha256':sha(Path(__file__)),'score':'PROB first token','same_centers':True,'position_default':'OFF; ON ablation reported','feature_input':'ImageBind on each actual new window, never mismatched 10s cache','gate':'run irrespective of observed short-window scores; no parameter selection'}
 p=OUT/'step4/frozen.json'
 if p.exists():assert json.loads(p.read_text())==protocol
 else:write(p,protocol)
 fingerprint=digest(protocol);engine=Engine(OUT/'step4');identity_cache={}
 for path in (OUT/'step1/inference').rglob('*.json'):
  value=json.loads(path.read_text());q=value['request'];identity_cache[(q['video_id'],tuple(q['segment']['frame_ids']))]=(path,value)
 for seconds in (2,4):
  config=replace(CONFIG,window_seconds=seconds)
  for r in rr:
   for seg in segments(r['length'],config):
    path=OUT/'step4/inference'/f'{seconds}s'/r['scene']/r['original_split']/r['video']/f"{seg['center']:06d}.json"
    if path.exists():continue
    key=(r['id'],tuple(seg['frame_ids']));ii=images(r,seg);request={'video_id':r['id'],'segment':seg,'images':ii,'fingerprint':fingerprint,'window_seconds':seconds}
    if key in identity_cache:
     oldp,old=identity_cache[key];value={**{k:old[k] for k in ['response','dec','parse_status','generated_token_ids','yes_logit','no_logit','prob','vocabulary_argmax','logits_dtype','logits_source','peak_allocated_gib','peak_reserved_gib']},'seconds':0.,'reused_source':str(oldp.relative_to(ROOT)),'reused_source_sha256':sha(oldp)}
    else:value=engine.call(resolve_images(ii),T1)
    write(path,{'request':request,'prompt':T1,**value});identity_cache[key]=(path,json.loads(path.read_text()))
   print(json.dumps({'event':'window_inference_video','seconds':seconds,'video_id':r['id']}),flush=True)
 del engine;gc.collect();torch.cuda.empty_cache()
 feature_cache=ROOT/'cache/stage8/window_features';feature_cache.mkdir(parents=True,exist_ok=True)
 model=ImageBind(DATA/'cache/vera',ROOT/'experiments/vera_ipad/source_reference')
 for seconds in (2,4):
  config=replace(CONFIG,window_seconds=seconds)
  for r in rr:
   p=feature_cache/f'{seconds}s'/r['scene']/r['original_split']/(r['video']+'.jsonl')
   if p.exists():continue
   files=base.files_for(r,r['frame_ids']);done=[]
   for seg in segments(r['length'],config):
    base.guard();tick=time.perf_counter();f,ids=model.encode(files,seg,samples=CONFIG.imagebind_samples);done.append({**seg,'feature':np.asarray(f).reshape(-1).tolist(),'imagebind_frame_ids':ids,'seconds':time.perf_counter()-tick})
   for d in done:append(p,d)
   print(json.dumps({'event':'window_feature_video','seconds':seconds,'video_id':r['id']}),flush=True)
 del model;gc.collect();torch.cuda.empty_cache();append(OUT/'events.jsonl',{'event':'step4_all_inference_complete_before_labels','time':time.time()});output=[];inventory=[]
 for seconds in (2,4):
  config=replace(CONFIG,window_seconds=seconds)
  for r in rr:
   ss=segments(r['length'],config);fp=feature_cache/f'{seconds}s'/r['scene']/r['original_split']/(r['video']+'.jsonl');ff=readlines(fp);assert len(ss)==len(ff);inventory.append({'path':str(fp.relative_to(ROOT)),'sha256':sha(fp),'local_only':True})
   values=[json.loads((OUT/'step4/inference'/f'{seconds}s'/r['scene']/r['original_split']/r['video']/f"{s['center']:06d}.json").read_text())['prob'] for s in ss]
   score=refine(values,[r['feature'] for r in ff],r['length'],config);y=labels(r)
   output.extend({'condition':f'PROB_{seconds}s','scene':r['scene'],'original_split':r['original_split'],'video':r['video'],'frame':i,'label':int(y[i]),**{s:float(v[i]) for s,v in score.items()}} for i in range(r['length']))
 from scripts.analyze_vera_stage8 import load
 output += [{**r,'condition':'PROB_10s'} for r in load(OUT/'step1/frame_scores.csv') if r['condition']=='PROB']
 writecsv(OUT/'step4/frame_scores.csv',output);write(OUT/'step4/metrics.json',summarize(output));write(OUT/'step4/feature_inventory.json',inventory);write(OUT/'step4/status.json',{'status':'complete','protected_files':protected(),'fps_verified':False});print(json.dumps(summarize(output)))
if __name__=='__main__':main()
