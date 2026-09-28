"""Read first raw LM-forward logits and decoded token from the SAME greedy call."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from scripts.vera_stage4_common import write
import scripts.train_vera_questions as base
class Engine(base.Engine):
 def __init__(self,out):
  base.OUT=out;super().__init__();self.out=out
  tokenizer=self.adapter.tokenizer
  self.tokens={s:tokenizer.encode(s,add_special_tokens=False) for s in ['Yes','No',' Yes',' No']}
  assert all(self.tokens.values());self.yes=self.tokens['Yes'][0];self.no=self.tokens['No'][0];assert self.yes!=self.no
  write(out/'tokenizer_check.json',{'variants':self.tokens,'selected_yes':self.yes,'selected_no':self.no,'fallback_first_token':any(len(self.tokens[s])!=1 for s in ['Yes','No']),'leading_space_variants_logged_not_summed':True})
 @torch.inference_mode()
 def call(self,files,prompt):
  base.guard();images=[]
  for p in files:
   with Image.open(p) as im:images.append(self.adapter.transform(im.convert('RGB').resize((448,448))))
  pixels=torch.stack(images).to('cuda',dtype=torch.bfloat16);model=self.adapter.model
  captured=[];generated=[];original=model.generate
  def hook(module,args,result):
   logits=result.logits[0,-1].detach().float();v=logits[[self.yes,self.no]].cpu()
   captured.append({'yes_logit':v[0].item(),'no_logit':v[1].item(),'prob':float(torch.softmax(v.double(),dim=0)[0]),'vocabulary_argmax':int(logits.argmax()),'logits_dtype':str(result.logits.dtype),'logits_source':'language_model forward output before generation processors'})
  def capture(*args,**kwargs):
   result=original(*args,**kwargs);generated.append(result[0].detach().cpu().tolist());return result
  handle=model.language_model.register_forward_hook(hook);model.generate=capture
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  try:response=model.chat(self.adapter.tokenizer,pixels,prompt,{'num_beams':1,'max_new_tokens':1,'do_sample':False},num_patches_list=[1]*len(files))
  finally:handle.remove();model.generate=original
  torch.cuda.synchronize();assert len(captured)==len(generated)==1 and len(generated[0])==1
  assert generated[0][0]==captured[0]['vocabulary_argmax'],'Raw-logit greedy argmax differs from generated token'
  stripped=response.strip();dec=1 if stripped=='Yes' else 0 if stripped=='No' else None
  return {'response':response,'dec':dec,'parse_status':'valid' if dec is not None else 'failed','generated_token_ids':generated[0],**captured[0],'seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
