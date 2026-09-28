"""Frozen Stage8 evaluator: continuous VERA scores and sklearn-only AUROC.
No model/prompt selection here. Labels enter only caller-approved evaluation phases.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from ipad.vera import VeraConfig, segments
STAGES=('initial','retrieved','smoothed','final')
SCENES=('R01','R02','R03','R04')

def refine(initial,features,length,config=VeraConfig()):
 x=np.asarray(initial,dtype=np.float64);f=np.asarray(features,dtype=np.float64)
 assert x.shape==(len(segments(length,config)),) and f.ndim==2 and len(f)==len(x)
 assert np.isfinite(x).all() and np.isfinite(f).all()
 n=np.linalg.norm(f,axis=1,keepdims=True);assert (n>0).all();f=f/n
 sim=f@f.T;k=max(1,int(config.retrieval_fraction*len(x)))
 neighbors=np.argsort(-sim,axis=1,kind='stable')[:,:k]
 logits=np.take_along_axis(sim,neighbors,axis=1)/config.temperature
 weights=np.exp(logits-logits.max(axis=1,keepdims=True));weights/=weights.sum(axis=1,keepdims=True)
 retrieved=(x[neighbors]*weights).sum(axis=1)
 half=config.kernel_size//2;kernel=np.exp(-np.arange(-half,half+1)**2/(2*config.sigma1**2));kernel/=kernel.sum()
 smooth=np.convolve(np.pad(retrieved,(half,half)),kernel,mode='valid')
 out={s:np.repeat(v,config.center_stride)[:length] for s,v in zip(STAGES,(x,retrieved,smooth))}
 position=np.exp(-(np.arange(1,length+1)-length//2)**2/(2*max(1,length//2)**2))
 out['final']=out['smoothed']*position
 return out

def metric(y,p,weight=None):
 y=np.asarray(y);p=np.asarray(p);mask=np.ones(len(y),bool) if weight is None else np.asarray(weight)>0
 assert len(y)==len(p) and np.isfinite(p).all()
 if len(np.unique(y[mask]))!=2:return {'auroc':None,'ap':None}
 return {'auroc':float(100*roc_auc_score(y,p,sample_weight=weight)), 'ap':float(100*average_precision_score(y,p,sample_weight=weight))}

def summarize(rows):
 out={}
 for condition in sorted({r['condition'] for r in rows}):
  part=[r for r in rows if r['condition']==condition];out[condition]={}
  for scene in (*SCENES,'pooled'):
   rr=[r for r in part if scene=='pooled' or r['scene']==scene]
   out[condition][scene]={s:metric([r['label'] for r in rr],[r[s] for r in rr]) for s in STAGES}
  out[condition]['macro']={s:{m:float(np.mean([out[condition][sc][s][m] for sc in SCENES])) if all(out[condition][sc][s][m] is not None for sc in SCENES) else None for m in ('auroc','ap')} for s in STAGES}
 return out

def paired_bootstrap(rows,left,right,n=2000,seed=0):
 a=[r for r in rows if r['condition']==left];b=[r for r in rows if r['condition']==right]
 key=lambda r:(r['scene'],r['original_split'],r['video'],int(r['frame']))
 a=sorted(a,key=key);b=sorted(b,key=key);assert [key(r) for r in a]==[key(r) for r in b]
 assert [r['label'] for r in a]==[r['label'] for r in b]
 y=np.array([r['label'] for r in a]);sc=np.array([r['scene'] for r in a]);vid=[(r['scene'],r['original_split'],r['video']) for r in a]
 videos=sorted(set(vid));index={v:i for i,v in enumerate(videos)};vi=np.array([index[v] for v in vid]);rng=np.random.default_rng(seed)
 values={s:{m:[] for m in ('auroc','ap')} for s in STAGES};invalid=0;draws=[]
 masks=[sc==s for s in SCENES]
 for _ in range(n):
  count=np.zeros(len(videos),int)
  for scene in SCENES:
   ids=[i for i,v in enumerate(videos) if v[0]==scene];selected=rng.choice(ids,len(ids),replace=True);np.add.at(count,selected,1)
  draws.append(count.tolist());w=count[vi]
  if any(len(np.unique(y[mask & (w>0)]))<2 for mask in masks):invalid+=1;continue
  for s in STAGES:
   pa=np.array([r[s] for r in a]);pb=np.array([r[s] for r in b])
   if np.array_equal(pa,pb):delta={'auroc':0.,'ap':0.}
   else:
    ma=[metric(y[m],pa[m],w[m]) for m in masks];mb=[metric(y[m],pb[m],w[m]) for m in masks]
    delta={k:float(np.mean([x[k]-z[k] for x,z in zip(ma,mb)])) for k in ('auroc','ap')}
   for k in delta:values[s][k].append(delta[k])
 return {'left':left,'right':right,'unit':'whole video; resample within scene','n':n,'seed':seed,'undefined_single_class':invalid,'undefined_rate':invalid/n,'valid':n-invalid,'ci95':{s:{m:np.percentile(v,[2.5,97.5]).tolist() if v else None for m,v in d.items()} for s,d in values.items()},'videos':videos,'resample_counts':draws}
