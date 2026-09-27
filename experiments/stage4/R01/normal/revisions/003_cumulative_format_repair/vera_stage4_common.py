"""Auditable, scene-isolated normal-context VERA experiment utilities."""
import hashlib,json,math,os,platform,re,subprocess,sys,time,traceback
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.train_vera_questions as base
from ipad.vera import VeraConfig,segments
from ipad.common import seed_everything
sha=base.sha;write=base.write;append=base.append;readlines=base.readlines
DATA=base.DATA;CONFIG=VeraConfig();SCENES=['R01','R02','R03','R04']

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def split_scene(scene):
 records=json.loads((ROOT/'experiments/stage2_1/split.json').read_text())['records']
 normals=sorted([r for r in records if r['scene']==scene and r['split']=='train' and r['video_label']==0],key=lambda r:r['id'])
 order=np.random.default_rng(0).permutation(len(normals));n=math.ceil(.2*len(normals))
 audit=sorted([normals[i] for i in order[:n]],key=lambda r:r['id']);generate=sorted([normals[i] for i in order[n:]],key=lambda r:r['id'])
 return records,generate,audit

def sanitized(row):
 return {k:row[k] for k in ['id','scene','original_split','video','length','relative_path','frames_sha256','frame_ids']}
def freeze(out,settings,sources):
 out.mkdir(parents=True,exist_ok=True);base.OUT=out;base.guard();seed_everything(0)
 value={**settings,'seed':0,'model_config':vars(CONFIG),'split_sha256':sha(ROOT/'experiments/stage2_1/split.json'),
  'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':1024},'source_sha256':{s:sha(ROOT/s) for s in sources}}
 value['fingerprint']=digest(value)
 p=out/'frozen.json'
 if p.exists():assert json.loads(p.read_text())==value,'Frozen sources/settings changed'
 else:write(p,value)
 append(out/'events.jsonl',{'event':'invocation','time':time.time(),'argv':sys.argv,'fingerprint':value['fingerprint']})
 if not (out/'environment.json').exists():
  write(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'parameter_updates':0})
  (out/'packages.txt').write_text('\n'.join(sorted({d.metadata['Name']+'=='+d.version for d in metadata.distributions() if d.metadata['Name']}))+'\n')
  (out/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
 inventory=json.loads((ROOT/'experiments/vera_ipad/model_inventory.json').read_text())
 for r in inventory:base.guard();assert sha(DATA/r['path'])==r['sha256']
 write(out/'model_verification.json',{'status':'passed','files':inventory,'checked_at':time.time()})
 return value

def parse_json(text):
 text=text.strip()
 if text.startswith('```'):
  text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I);text=re.sub(r'\s*```$','',text)
 result=json.loads(text)
 if not isinstance(result,dict):raise ValueError('Expected JSON object')
 return result

class Engine(base.Engine):
 def __init__(self,out):self.out=out;base.OUT=out;super().__init__()
 @torch.inference_mode()
 def text(self,prompt):
  base.guard();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  response=self.adapter.model.chat(self.adapter.tokenizer,None,prompt,dict(num_beams=1,max_new_tokens=1024,do_sample=False),num_patches_list=[])
  torch.cuda.synchronize()
  return response,{'seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
 def recorded(self,path,prompt,row=None,seg=None,validator=None):
  request={'prompt':prompt,'video_id':row['id'] if row else None,'segment':seg}
  request_hash=digest(request)
  if path.exists():
   r=json.loads(path.read_text());assert r['request_hash']==request_hash
   if validator:validator(r['parsed'])
   return r
  files=base.files_for(row,seg['frame_ids']) if row else None
  response,stats=self.call(files,prompt) if row else self.text(prompt)
  raw={**request,'request_hash':request_hash,'response':response,**stats,'time':time.time()}
  append(self.out/'calls.jsonl',raw)
  if validator:
   for attempt in range(3):
    try:
     result=parse_json(response);validator(result);raw['parsed']=result;break
    except (ValueError,KeyError,TypeError,AssertionError) as exc:
     append(self.out/'format_failures.jsonl',{'request_hash':request_hash,'attempt':attempt,'error':str(exc),'response':response})
     if attempt==2:raise
     repair='Repair JSON serialization/schema only. Do not invent evidence, change judgments, or add content. Return only the requested JSON.\nOriginal request:\n'+prompt+'\nOriginal response:\n'+raw['response']+'\nParser error: '+str(exc)
     response,st=self.text(repair);append(self.out/'calls.jsonl',{'role':'format_repair','request_hash':request_hash,'attempt':attempt+1,'prompt':repair,'response':response,**st,'time':time.time()})
     raw.setdefault('format_repairs',[]).append({'response':response,**st})
  write(path,raw);return raw

def image_prefix(seg):return ''.join(f'Frame{i+1} (original frame ID {fid}): <image>\n' for i,fid in enumerate(seg['frame_ids']))
def fail(out,exc):
 append(out/'failures.jsonl',{'time':time.time(),'error':repr(exc),'traceback':traceback.format_exc()})
 write(out/'status.json',{'status':'paused_low_disk' if (out/'disk_pause.json').exists() or (ROOT/'runs/disk_pause.json').exists() else 'failed','error':repr(exc)})
