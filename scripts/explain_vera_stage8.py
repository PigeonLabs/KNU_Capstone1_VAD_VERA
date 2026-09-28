"""Separate, bounded explanations for score-selected windows. No labels enter prompts."""
import sys,json,csv,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from scripts.vera_stage8_common import *
import scripts.train_vera_questions as base
PROMPT=''.join(f'Image {i+1}: <image>\n' for i in range(12))+'''REFERENCE (normal): Image 1-4. QUERY: Image 5-12.
In one sentence, describe the most noticeable difference between QUERY and REFERENCE.
If none, say "No visible difference."'''
@torch.inference_mode()
def main():
 out=OUT/'step5/explanations';base.OUT=out;base.guard();evaluator_guard();protected();rr=rows();lookup={r['id']:r for r in rr}
 records=list(csv.DictReader((OUT/'step5/window_scores.csv').open()));records=[r for r in records if r['condition']=='HYBRID'];chosen=[]
 for vid in sorted({r['video_id'] for r in records}):chosen.append(max([r for r in records if r['video_id']==vid],key=lambda r:(float(r['raw_score']),-int(r['center']))))
 selection={'method':'one highest mean hybrid score stride16 block per video, tie earliest center; labels not used','chosen':chosen,'prompt':PROMPT,'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':256,'repetition_penalty':1.1},'score_source_sha256':sha(OUT/'step5/window_scores.csv'),'source_sha256':sha(Path(__file__))}
 write(out/'frozen.json',selection);engine=base.Engine();model=engine.adapter.model
 for item in chosen:
  r=lookup[item['video_id']];seg=next(s for s in segments(r['length']) if s['center']==int(item['center']));ref=next(x for x in rr if x['scene']==r['scene'] and x['split']=='train' and x['video_label']==0);rs=segments(ref['length'])[len(segments(ref['length']))//2];ids=np.linspace(rs['start'],rs['end']-1,4,dtype=int).tolist()
  ii=[{'role':'reference','name':f'Image {i+1}','video_id':ref['id'],'relative_path':ref['relative_path'],'frame_id':f,'sha256':ref['frames_sha256'][f]} for i,f in enumerate(ids)]+images(r,seg)
  path=out/r['scene']/r['original_split']/(r['video']+'.json')
  if path.exists():continue
  pixels=[]
  for p in resolve_images(ii):
   with Image.open(p) as im:pixels.append(engine.adapter.transform(im.convert('RGB').resize((448,448))))
  pixels=torch.stack(pixels).cuda().bfloat16();base.guard();torch.cuda.synchronize();start=time.perf_counter();captured=[];original=model.generate
  def capture(*a,**kw):
   t=original(*a,**kw);captured.append(t[0].cpu().tolist());return t
  model.generate=capture
  try:response=model.chat(engine.adapter.tokenizer,pixels,PROMPT,selection['generation'].copy(),num_patches_list=[1]*12)
  finally:model.generate=original
  torch.cuda.synchronize();write(path,{'video_id':r['id'],'segment':seg,'images':ii,'prompt':PROMPT,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),'response':response,'generated_token_ids':captured[0],'generated_tokens':len(captured[0]),'at_token_limit':len(captured[0])==256,'seconds':time.perf_counter()-start,'model_generated_not_verified_annotation':True});print(json.dumps({'event':'explanation','video':r['id'],'response':response}),flush=True)
 write(out/'status.json',{'status':'complete','calls':len(chosen),'score_changes':False})
if __name__=='__main__':main()
