"""Frozen VERA learner/optimizer question training on a fixed IPAD video split."""
import argparse,hashlib,json,os,platform,re,shutil,subprocess,sys,time,traceback
import importlib.metadata as metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from ipad.common import seed_everything
from ipad.vera import VeraConfig,parse_response
from ipad.vera_models import InternVL
DATA=Path('/home/jeong/Desktop/IPAD');CACHE=DATA/'cache/vera'
OUT=ROOT/'experiments/stage2_2';REF=ROOT/'experiments/vera_ipad/source_reference/VERA'
INITIAL='1. Is there any suspicious person or object that looks unusual in this scene?\n2. Is there any behavior that looks unusual in this scene?\n'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while block:=f.read(8*1024**2):h.update(block)
 return h.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def append(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def readlines(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def guard():
 if (ROOT/'runs/disk_pause.json').exists() or (OUT/'disk_pause.json').exists():raise RuntimeError('Latched disk pause: explicit user resume required')
 free=shutil.disk_usage(ROOT).free
 if free<=10*1024**3:
  write(OUT/'disk_pause.json',{'status':'paused_low_disk','time':time.time(),'free_bytes':free,'automatic_resume':False})
  raise RuntimeError('Disk free space <=10 GiB; no automatic resume')
def learner_prompt(template,questions):
 assert template.count('$Data')==1 and template.count('Based on the analysis above')==1
 return template.replace('$Data',''.join(f'Frame{i+1}: <image>\n' for i in range(8))).replace('Based on the analysis above',questions+'Based on the analysis above')
def optimizer_prompt(template,questions,predictions,targets,batch_size):
 marker='** Model Descriptions: **';assert template.count(marker)==1
 prefix='['+''.join('['+''.join(f'Frame{i+1}: <image>\n' for i in range(8))+'] ' for _ in range(batch_size))+']\n'
 result=template.replace(marker,prefix+marker).replace('Based on the analysis above',questions+'Based on the analysis above')
 for kind,values in [('Prediction',predictions),('GroundTruth',targets)]:
  needle='['+' '.join('[$'+kind+']' for _ in range(10))+']';assert needle in result
  result=result.replace(needle,str(values))
 assert result.count('<image>')==8*batch_size
 return result

def parse_questions(response):
 matches=list(re.finditer(r'New Prompt Questions\s*:?\s*(?:\*\*)?',response,re.I))
 if len(matches)!=1:raise ValueError('Missing or repeated New Prompt Questions marker')
 tail=response[matches[0].end():].strip().strip('`').strip()
 # Accept complete numbered questions, including a wrapped continuation line.
 numbered=list(re.finditer(r'^\s*(\d+)[.)]\s+',tail,re.M))
 if not 1<=len(numbered)<=5:raise ValueError('Optimizer did not return 1..5 numbered questions')
 if [int(m.group(1)) for m in numbered]!=list(range(1,len(numbered)+1)):raise ValueError('Nonsequential questions')
 values=[]
 for i,m in enumerate(numbered):
  item=tail[m.end():numbered[i+1].start() if i+1<len(numbered) else len(tail)].strip().rstrip('`').strip()
  if not item.endswith('?') or len(item)>1500:raise ValueError('Incomplete/trailing content in question')
  values.append(f'{i+1}. {item}')
 return '\n'.join(values)+'\n'

def files_for(row,ids=None):
 folder=DATA/row['relative_path'];mapping={int(p.stem):p for p in folder.glob('*.jpg')}
 selected=row['training_frame_ids'] if ids is None else ids
 files=[mapping[i] for i in selected]
 for i,p in zip(selected,files):
  if sha(p)!=row['frames_sha256'][i]:raise RuntimeError(f'Input changed: {p}')
 return files

class Engine:
 def __init__(self):
  guard();start=time.time();self.adapter=InternVL(CACHE/'InternVL2-8B',(REF/'VERA_learner_instruct.txt').read_text(),VeraConfig())
  self.template=(REF/'VERA_learner_instruct.txt').read_text();self.optimizer_template=(REF/'VERA_optimizer_instruct.txt').read_text()
  append(OUT/'events.jsonl',{'event':'model_loaded','seconds':time.time()-start,'time':time.time()})
 @torch.inference_mode()
 def call(self,files,prompt):
  from PIL import Image
  guard();images=[]
  for file in files:
   with Image.open(file) as im:images.append(self.adapter.transform(im.convert('RGB').resize((448,448))))
  pixels=torch.stack(images).to(device='cuda',dtype=torch.bfloat16)
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  response=self.adapter.model.chat(self.adapter.tokenizer,pixels,prompt,dict(num_beams=1,max_new_tokens=1024,do_sample=False),num_patches_list=[1]*len(files))
  torch.cuda.synchronize()
  return response,{'seconds':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
 def predict(self,row,questions):
  prompt=learner_prompt(self.template,questions);response,stats=self.call(files_for(row),prompt)
  return {'video_id':row['id'],'frame_ids':row['training_frame_ids'],'prompt':prompt,'response':response,**stats}

def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=10);a=p.parse_args()
 if a.epochs!=10:raise ValueError('First approved run fixes 10 epochs; no outcome-based budget changes')
 OUT.mkdir(parents=True,exist_ok=True);guard();seed_everything(0)
 split_path=ROOT/'experiments/stage2_1/split.json';split=json.loads(split_path.read_text())
 train=[r for r in split['records'] if r['split']=='train'];val=[r for r in split['records'] if r['split']=='validation']
 order_rng=np.random.default_rng(0);schedule=[]
 for epoch in range(a.epochs):
  order=order_rng.permutation(len(train)).tolist()
  for start in range(0,len(order),2):schedule.append({'epoch':epoch+1,'batch_ids':[train[i]['id'] for i in order[start:start+2]]})
 sources=['scripts/train_vera_questions.py','ipad/vera.py','ipad/vera_models.py','ipad/common.py','experiments/vera_ipad/source_reference/VERA/VERA_learner_instruct.txt','experiments/vera_ipad/source_reference/VERA/VERA_optimizer_instruct.txt']
 config={'seed':0,'epochs':10,'batch_size':2,'frames_per_video':8,'frame_sampling':'i*floor(F/8), i=0..7','max_questions':5,'initial_questions':INITIAL,
 'validation_period_iterations':100,'validation_steps':[0,*range(100,len(schedule)+1,100)],'selection':'maximum validation video accuracy; strict improvement, earliest candidate wins ties',
 'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':1024},'model':vars(VeraConfig()),'train_count':len(train),'validation_count':len(val),'iterations':len(schedule),
 'split_sha256':sha(split_path),'source_sha256':{s:sha(ROOT/s) for s in sources},'schedule':schedule,
 'deviations':['IPAD video-level stratified re-split instead of original benchmark','Correct per-image token expansion: num_patches_list=[1]*16 optimizer and [1]*8 learner','Strict parsing; malformed optimizer retains prior Q with failure recorded, malformed learner stops','BF16 eager attention, frozen weights; no numerical optimizer or gradient updates']}
 encoded=json.dumps(config,sort_keys=True).encode();fingerprint=hashlib.sha256(encoded).hexdigest();config['fingerprint']=fingerprint
 frozen=OUT/'frozen.json'
 if frozen.exists():
  if json.loads(frozen.read_text())!=config:raise RuntimeError('Frozen training configuration/source differs')
 else:
  write(frozen,config);write(OUT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'model_inventory_sha256':sha(ROOT/'experiments/vera_ipad/model_inventory.json')})
  names=sorted({d.metadata['Name'] for d in metadata.distributions() if d.metadata['Name']})
  (OUT/'installed_packages.txt').write_text('\n'.join(name+'=='+metadata.version(name) for name in names)+'\n')
  (OUT/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
 history=readlines(OUT/'iterations.jsonl');assert [r['iteration'] for r in history]==list(range(1,len(history)+1))
 questions=history[-1]['questions_after'] if history else INITIAL
 append(OUT/'events.jsonl',{'event':'invocation','time':time.time(),'argv':sys.argv,'executable':sys.executable,'resume_iterations':len(history),'fingerprint':fingerprint})
 write(OUT/'status.json',{'status':'running','completed_iterations':len(history),'total_iterations':len(schedule),'fingerprint':fingerprint})
 engine=Engine();lookup={r['id']:r for r in train}
 def validate(step,q):
  path=OUT/'validation'/f'{step:04d}.json'
  if path.exists():
   assert json.loads(path.read_text())['questions']==q;return
  records=[]
  for row in val:
   record=engine.predict(row,q);record['target']=row['video_label'];record['step']=step
   append(OUT/'validation_responses.jsonl',record)
   record['prediction']=parse_response(record['response']);records.append(record)
  correct=sum(r['prediction']==r['target'] for r in records)
  write(path,{'step':step,'questions':q,'correct':correct,'total':len(records),'accuracy':correct/len(records),'records':records})
  print(json.dumps({'event':'validation','step':step,'correct':correct,'total':len(records),'accuracy':correct/len(records)}),flush=True)
 validate(0,INITIAL)
 if history and len(history)%100==0:validate(len(history),questions)
 for zero,entry in enumerate(schedule):
  if zero<len(history):continue
  iteration=zero+1;batch=[lookup[i] for i in entry['batch_ids']];predictions=[];learner_records=[]
  for row in batch:
   record=engine.predict(row,questions);record.update(iteration=iteration,target=row['video_label'])
   append(OUT/'learner_responses.jsonl',record)
   predictions.append(parse_response(record['response']));learner_records.append(record)
  prompt=optimizer_prompt(engine.optimizer_template,questions,predictions,[r['video_label'] for r in batch],len(batch))
  response,stats=engine.call([p for row in batch for p in files_for(row)],prompt)
  raw={'iteration':iteration,'batch_ids':entry['batch_ids'],'prompt':prompt,'response':response,**stats}
  append(OUT/'optimizer_responses.jsonl',raw)
  error=None
  try:new_questions=parse_questions(response)
  except ValueError as exc:new_questions=questions;error=str(exc)
  record={'iteration':iteration,**entry,'questions_before':questions,'questions_after':new_questions,'predictions':predictions,'targets':[r['video_label'] for r in batch],
   'train_correct':sum(p==r['video_label'] for p,r in zip(predictions,batch)),'optimizer_parse_error':error,'time':time.time(),'optimizer_seconds':stats['seconds'],'learner_seconds':sum(r['seconds'] for r in learner_records)}
  append(OUT/'iterations.jsonl',record);questions=new_questions
  write(OUT/'status.json',{'status':'running','completed_iterations':iteration,'total_iterations':len(schedule),'fingerprint':fingerprint})
  print(json.dumps({'event':'iteration','iteration':iteration,'epoch':entry['epoch'],'correct':record['train_correct'],'optimizer_parse_error':error,'seconds':record['learner_seconds']+record['optimizer_seconds']}),flush=True)
  if iteration%100==0:validate(iteration,questions)
 write(OUT/'status.json',{'status':'complete','completed_iterations':len(schedule),'epochs':10,'validation_steps':config['validation_steps'],'fingerprint':fingerprint,'finished_at':time.time()})
 print('TRAINING COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  OUT.mkdir(parents=True,exist_ok=True);append(OUT/'failures.jsonl',{'time':time.time(),'error':repr(exc),'traceback':traceback.format_exc()})
  old=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
  write(OUT/'status.json',{**old,'status':'paused_low_disk' if (OUT/'disk_pause.json').exists() or (ROOT/'runs/disk_pause.json').exists() else 'failed','error':repr(exc)});raise
