"""Frozen decision-first parser and128-token engine with read-only generation capture."""
import json,platform,re,subprocess,sys,time
import importlib.metadata as metadata
from pathlib import Path
import torch
from scripts.vera_stage4_common import ROOT,DATA,CONFIG,digest,sha,write,append,seed_everything
import scripts.train_vera_questions as base

def parse_response(text):
 lines=[line.rstrip() for line in text.splitlines() if line.strip()]
 if not lines or not re.fullmatch(r'Output: [01]',lines[0]):raise ValueError('Invalid first nonempty Output line')
 pred=int(lines[0][-1]);fields=[line for line in lines if re.search(r'\boutput\b',line,re.I)]
 if any(line!=f'Output: {pred}' for line in fields):raise ValueError('Conflicting, incomplete, or malformed Output field')
 return pred

def quality(text,stats):
 lines=[line.rstrip() for line in text.splitlines() if line.strip()];fields=[l for l in lines if re.search(r'\boutput\b',l,re.I)]
 evidence=[l[len('Evidence:'):].strip() for l in lines if l.startswith('Evidence:')]
 words=sum(len(x.split()) for x in evidence)
 sentences=[x.strip().lower() for x in re.split(r'[.!?]+',' '.join(evidence)) if x.strip()]
 return {'duplicate_output':len(fields)>1,'evidence_missing':not evidence or not any(evidence),'evidence_over40words':words>40,'evidence_words':words,'evidence_repeated_sentence':len(sentences)!=len(set(sentences)),'explanation_truncated':stats.get('generated_token_count',0)>=128 and stats.get('ended_with_eos') is False,'semantic_consistency_checked':False}

def freeze(out,settings,sources):
 out.mkdir(parents=True,exist_ok=True);base.OUT=out;base.guard();seed_everything(0)
 value={**settings,'seed':0,'model_config':{**vars(CONFIG),'max_new_tokens':128},'split_sha256':sha(ROOT/'experiments/stage2_1/split.json'),'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':128},'source_sha256':{s:sha(ROOT/s) for s in sources}}
 value['fingerprint']=digest(value);p=out/'frozen.json'
 if p.exists():assert json.loads(p.read_text())==value,'Frozen sources/settings changed'
 else:write(p,value)
 append(out/'events.jsonl',{'event':'invocation','time':time.time(),'argv':sys.argv,'fingerprint':value['fingerprint']})
 if not (out/'environment.json').exists():
  write(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'parameter_updates':0})
  (out/'packages.txt').write_text('\n'.join(sorted({d.metadata['Name']+'=='+d.version for d in metadata.distributions() if d.metadata['Name']}))+'\n')
  (out/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
 inventory=json.loads((ROOT/'experiments/vera_ipad/model_inventory.json').read_text())
 for r in inventory:base.guard();assert sha(DATA/r['path'])==r['sha256']
 write(out/'model_verification.json',{'status':'passed','files':inventory,'checked_at':time.time()});return value

class Engine(base.Engine):
 def __init__(self,out):self.out=out;base.OUT=out;super().__init__()
 @torch.inference_mode()
 def call(self,files,prompt):
  from PIL import Image
  base.guard();images=[]
  for file in files:
   with Image.open(file) as im:images.append(self.adapter.transform(im.convert('RGB').resize((448,448))))
  pixels=torch.stack(images).to(device='cuda',dtype=torch.bfloat16)
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  model=self.adapter.model;original=model.generate;captured=[]
  def capture(*args,**kwargs):
   result=original(*args,**kwargs)
   ids=result[0].detach().cpu().tolist();eos=kwargs.get('eos_token_id');eos_ids=eos if isinstance(eos,list) else [eos]
   captured.append({'generated_token_ids':ids,'generated_token_count':len(ids),'eos_token_ids':eos_ids,'ended_with_eos':bool(ids and ids[-1] in eos_ids),'generation_capture':'unmodified model.generate return tensor; decoder-only inputs_embeds','actual_generation_config':{k:kwargs[k] for k in ['num_beams','max_new_tokens','do_sample']}})
   return result
  model.generate=capture
  try:response=model.chat(self.adapter.tokenizer,pixels,prompt,dict(num_beams=1,max_new_tokens=128,do_sample=False),num_patches_list=[1]*len(files))
  finally:model.generate=original
  torch.cuda.synchronize();assert len(captured)==1
  return response,{'seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,**captured[0]}
