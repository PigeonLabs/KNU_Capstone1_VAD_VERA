"""Full seed-0 replay of validation and explanations after initializer omission.
Original artifacts remain immutable; exact equality is required, no outcome tuning.
"""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from ipad.common import seed_everything
from scripts.vera_stage8_common import *
from scripts.vera_stage8_engine import Engine
import scripts.train_vera_questions as base
@torch.inference_mode()
def main():
 seed_everything(0);out=OUT/'seed0_verification';base.OUT=out;base.guard();initial=torch.initial_seed();engine=Engine(out);count=0
 for path in sorted((OUT/'step5/validation_inference').rglob('*.json')):
  old=json.loads(path.read_text());request=old['request'];files=resolve_images(request['images']);before=torch.get_rng_state();before_cuda=torch.cuda.get_rng_state();new=engine.call(files,old['prompt'])
  matched=all(new[k]==old[k] for k in ['response','dec','generated_token_ids','yes_logit','no_logit','prob','vocabulary_argmax'])
  rng_unchanged=torch.equal(before,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
  append(out/'validation_replay.jsonl',{'source':str(path.relative_to(ROOT)),'source_sha256':sha(path),'initial_seed':initial,'request':request,'prompt':old['prompt'],**new,'exact_match':matched,'inference_rng_unchanged':rng_unchanged});assert matched and rng_unchanged,'Seed0 replay mismatch; preserve and stop';count+=1
  if count%100==0:print('Seed0 validation matched',count,flush=True)
 explanations=0;model=engine.adapter.model
 for path in sorted((OUT/'step5/explanations').glob('R*/*/*.json')):
  old=json.loads(path.read_text());pixels=[]
  for p in resolve_images(old['images']):
   with Image.open(p) as im:pixels.append(engine.adapter.transform(im.convert('RGB').resize((448,448))))
  pixels=torch.stack(pixels).cuda().bfloat16();before=torch.get_rng_state();before_cuda=torch.cuda.get_rng_state();captured=[];original=model.generate
  def capture(*a,**kw):
   ids=original(*a,**kw);captured.append(ids[0].cpu().tolist());return ids
  model.generate=capture;start=time.perf_counter()
  try:response=model.chat(engine.adapter.tokenizer,pixels,old['prompt'],{'do_sample':False,'num_beams':1,'max_new_tokens':256,'repetition_penalty':1.1},num_patches_list=[1]*12)
  finally:model.generate=original
  match=response==old['response'] and captured[0]==old['generated_token_ids'];rng_unchanged=torch.equal(before,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
  append(out/'explanation_replay.jsonl',{'source':str(path.relative_to(ROOT)),'source_sha256':sha(path),'seed':initial,'response':response,'generated_token_ids':captured[0],'seconds':time.perf_counter()-start,'exact_match':match,'inference_rng_unchanged':rng_unchanged});assert match and rng_unchanged;explanations+=1
 write(out/'status.json',{'status':'complete','seed':initial,'validation_calls_exactly_reproduced':count,'explanations_exactly_reproduced':explanations,'all_rng_states_unchanged_in_inference':True,'original_scores_selection_and_metrics_unchanged':True,'source_sha256':sha(Path(__file__))});print('Seed0 full replay complete',count,explanations,flush=True)
if __name__=='__main__':main()
