import csv,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import DATA,sha,digest,write,append,sanitized,CONFIG,segments,readlines,SCENES
from scripts.vera_stage7_common import resolve_images
from ipad.vera import QUESTIONS
OUT=ROOT/'experiments/stage8';REPORT=ROOT/'reports/stage8'
T1=''.join(f'Frame{i+1}: <image>\n' for i in range(8))+'''You are shown 8 frames from a short industrial video segment.
Consider these questions:
'''+QUESTIONS+'''Based only on visible evidence, does this segment contain an anomaly?
Answer with a single word: Yes or No.
Answer:'''

def rows():return json.loads((ROOT/'experiments/stage2_1/split.json').read_text())['records']
def writecsv(path,records):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def labels(r):
 import numpy as np
 if r['original_split']=='training':return np.zeros(r['length'],int)
 p=DATA/'IPAD_dataset'/r['scene']/'test_label'/f"{int(r['video']):03d}.npy";assert sha(p)==r['label_sha256']
 y=np.load(p,allow_pickle=False).reshape(-1);assert len(y)==r['length'] and np.isin(y,[0,1]).all();return y

def features(r):
 inv={r['path']:r for r in json.loads((ROOT/'experiments/stage3/inference_inventory.json').read_text()) if r['local_only']}
 p=ROOT/'cache/stage3/features'/r['scene']/r['original_split']/(r['video']+'.jsonl');assert sha(p)==inv[str(p.relative_to(ROOT))]['sha256']
 rr=readlines(p);ss=segments(r['length']);assert len(rr)==len(ss)
 for a,b in zip(rr,ss):assert all(a[k]==v for k,v in b.items())
 return [x['feature'] for x in rr]

def images(r,seg):return [{'role':'query','name':f'Frame{i+1}','video_id':r['id'],'relative_path':r['relative_path'],'frame_id':fid,'sha256':r['frames_sha256'][fid]} for i,fid in enumerate(seg['frame_ids'])]

def protected(init=False):
 p=OUT/'protected_previous.json';paths=[x for d in (ROOT/'experiments').iterdir() if d.is_dir() and d.name!='stage8' for x in d.rglob('*') if x.is_file()]
 paths += [ROOT/x for x in ['ipad/vera.py','ipad/vera_models.py','scripts/vera_stage4_common.py','scripts/evaluate_vera_normal_context.py','scripts/run_vera_stage7.py']]
 current={str(x.relative_to(ROOT)):sha(x) for x in paths}
 if init and not p.exists():write(p,{'files':current})
 else:assert current==json.loads(p.read_text())['files'],'Prior artifacts changed'
 return len(current)

def evaluator_guard(init=False):
 p=OUT/'evaluator_lock.json';value={'files':{n:sha(ROOT/n) for n in ['scripts/vera_stage8_eval.py','ipad/vera.py']},'auroc':'sklearn.metrics.roc_auc_score','seed':0,'bootstrap':2000,'frozen_at':time.time()}
 if init and not p.exists():write(p,value)
 else:assert value['files']==json.loads(p.read_text())['files'],'Frozen evaluator changed'
