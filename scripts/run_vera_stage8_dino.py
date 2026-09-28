"""Leakage-free original DINOv2 unconditional spatial prototype, streaming queries.
Only sampled normal TRAIN features are cached. No full patch-feature cache.
"""
import argparse,sys,json,time,gc,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scripts.vera_stage8_common import *
import scripts.train_vera_questions as base
COMMIT='7764ea0f912e53c92e82eb78a2a1631e92725fc8';CACHE=ROOT/'cache/stage8/dino';DEST=OUT/'dino'

def kmeans(points,k,seed=0):
 points=F.normalize(points.float(),dim=-1);p,n,c=points.shape;k=min(k,n)
 gen=torch.Generator(device=points.device).manual_seed(seed);ids=torch.randperm(n,generator=gen,device=points.device)[:k];centers=points[:,ids].clone()
 for _ in range(20):
  assigned=(points@centers.transpose(1,2)).argmax(-1);sums=torch.zeros_like(centers);sums.scatter_add_(1,assigned[:,:,None].expand(p,n,c),points)
  counts=torch.zeros(p,k,device=points.device);counts.scatter_add_(1,assigned,torch.ones_like(assigned,dtype=torch.float32));centers=torch.where((counts>0)[:,:,None],F.normalize(sums,dim=-1),centers)
 return centers

class Extractor:
 def __init__(self):
  inv=json.loads((OUT/'dino_download.json').read_text());assert sha(CACHE/'dinov2_vitb14_pretrain.pth')==inv['weights_sha256']
  for p,h in inv['source_files'].items():assert sha(CACHE/COMMIT/p)==h
  self.model=torch.hub.load(str(CACHE/COMMIT),'dinov2_vitb14',source='local',pretrained=False).eval().requires_grad_(False).cuda()
  self.model.load_state_dict(torch.load(CACHE/'dinov2_vitb14_pretrain.pth',map_location='cpu',weights_only=True))
  self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 @torch.inference_mode()
 def __call__(self,items):
  arrays=[]
  for r,i in items:
   p=base.files_for(r,[i])[0];arrays.append(cv2.resize(cv2.imread(str(p)),(256,256)))
  x=torch.from_numpy(np.stack(arrays)).cuda().permute(0,3,1,2).float().div_(127.5).sub_(1)
  rgb=(x[:,[2,1,0]]+1)/2;rgb=F.interpolate(rgb,size=(252,252),mode='bilinear',align_corners=False);rgb=(rgb-self.mean)/self.std
  value=self.model.get_intermediate_layers(rgb,n=[11],return_class_token=True,norm=True)[0]
  return F.normalize(value[0].float(),dim=-1),F.normalize(value[1].float(),dim=-1)

def main():
 from ipad.common import seed_everything
 base.OUT=DEST;base.guard();seed_everything(0);torch.set_num_threads(8);evaluator_guard();allrows=rows();start=time.time()
 config={'seed':0,'normal_train_only':True,'phase_bins':20,'per_phase_cap':512,'prototypes_per_phase_budget':10,'kmeans_iterations':20,'query_all_frames':True,'source_sha256':sha(Path(__file__)),'split_sha256':sha(ROOT/'experiments/stage2_1/split.json'),'weights_sha256':json.loads((OUT/'dino_download.json').read_text())['weights_sha256'],'feature_precision':'FP32 extraction -> FP16 sampled storage -> FP32 match','postprocessing':'per-frame raw spatial mean cosine distance; temporal grouping for VERA only in downstream hybrid','frame_normalization':'same RGB252 bilinear and original BGR256 cv2 preprocessing','validation_and_evaluation_not_in_memory':True}
 fp=DEST/'frozen.json'
 if fp.exists():assert json.loads(fp.read_text())==config
 else:write(fp,config)
 model=Extractor();inventory=[]
 for scene in SCENES:
  base.guard();normal=[r for r in allrows if r['scene']==scene and r['split']=='train' and r['video_label']==0];selected=[];counts=[];rng=np.random.default_rng(0)
  for phase in range(20):
   pool=[(r,i+8) for r in normal for i in range(r['length']-15) if min(19,i*20//r['length'])==phase]
   chosen=rng.choice(len(pool),min(len(pool),512),replace=False);selected.extend(pool[i] for i in chosen);counts.append(len(chosen))
  k=sum(min(10,n) for n in counts);meta={'scene':scene,'normal_train_ids':[r['id'] for r in normal],'selected':[{'video_id':r['id'],'frame':i,'sha256':r['frames_sha256'][i]} for r,i in selected],'phase_counts':counts,'prototype_count':k};write(DEST/scene/'memory_inputs.json',meta)
  cache=CACHE/scene;cache.mkdir(exist_ok=True);bankfile=cache/'unconditional.pt'
  if not bankfile.exists():
   pf=cache/'sampled_patch.npy';cf=cache/'sampled_cls.npy'
   if not pf.exists():
    patches=np.lib.format.open_memmap(pf.with_suffix('.tmp.npy'),mode='w+',dtype=np.float16,shape=(len(selected),324,768));cls=np.lib.format.open_memmap(cf.with_suffix('.tmp.npy'),mode='w+',dtype=np.float16,shape=(len(selected),768))
    for i in range(0,len(selected),32):
     base.guard();pp,cc=model(selected[i:i+32]);patches[i:i+len(pp)]=pp.cpu().numpy();cls[i:i+len(pp)]=cc.cpu().numpy()
    patches.flush();cls.flush();del patches,cls;pf.with_suffix('.tmp.npy').replace(pf);cf.with_suffix('.tmp.npy').replace(cf)
   patches=np.load(pf,mmap_mode='r');parts=[]
   for pos in range(0,324,9):
    base.guard();points=torch.from_numpy(np.array(patches[:,pos:pos+9],copy=True)).cuda().transpose(0,1);parts.append(kmeans(points,k).cpu().half());del points
   bank=torch.cat(parts,dim=0);torch.save(bank,bankfile);del patches,parts;print(json.dumps({'event':'memory_fit','scene':scene,'samples':len(selected),'prototypes':k}),flush=True)
  bank=F.normalize(torch.load(bankfile,weights_only=True).cuda().float(),dim=-1)
  for r in [r for r in allrows if r['scene']==scene and r['split'] in ['validation','evaluation']]:
   dest=DEST/'scores'/r['scene']/r['original_split']/(r['video']+'.csv')
   if dest.exists():continue
   output=[];tick=time.perf_counter()
   for i in range(0,r['length'],32):
    base.guard();pp,_=model([(r,f) for f in range(i,min(i+32,r['length']))]);pp=pp.half().float();pp=F.normalize(pp,dim=-1)
    with torch.inference_mode():score=(1-torch.einsum('bpc,pkc->bpk',pp,bank).max(-1).values).clamp_min(0).mean(-1).cpu().tolist()
    output.extend({'video_id':r['id'],'frame':i+j,'unconditional_nn':v} for j,v in enumerate(score))
   writecsv(dest,output);append(DEST/'events.jsonl',{'event':'score_video','video_id':r['id'],'split':r['split'],'seconds':time.perf_counter()-tick});print(json.dumps({'event':'score_video','video_id':r['id']}),flush=True)
  inventory.extend({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size,'local_only':True} for p in cache.iterdir() if p.is_file())
  del bank;gc.collect();torch.cuda.empty_cache()
 write(DEST/'local_inventory.json',inventory);write(DEST/'status.json',{'status':'complete','seconds':time.time()-start,'labels_opened':False,'protected_files':protected()})
if __name__=='__main__':main()
