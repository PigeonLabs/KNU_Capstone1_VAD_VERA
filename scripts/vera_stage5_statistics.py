"""Exact tied-score weighted rank metrics for paired video bootstrap."""
import numpy as np

def rank_buckets(labels,scores,video_indices,nvideos):
 y=np.asarray(labels,dtype=int);x=np.asarray(scores,dtype=float);v=np.asarray(video_indices,dtype=int)
 assert len(y)==len(x)==len(v) and np.isin(y,[0,1]).all() and np.isfinite(x).all()
 values,inverse=np.unique(-x,return_inverse=True)
 positive=np.zeros((nvideos,len(values)),dtype=np.float64);negative=np.zeros_like(positive)
 np.add.at(positive,(v,inverse),y);np.add.at(negative,(v,inverse),1-y)
 return positive,negative

def weighted_rank_metrics(buckets,multiplicity,batch=100):
 positive,negative=buckets;counts=np.asarray(multiplicity,dtype=float);auroc=[];ap=[]
 for begin in range(0,len(counts),batch):
  p=counts[begin:begin+batch]@positive;n=counts[begin:begin+batch]@negative
  tp=np.cumsum(p,axis=1);fp=np.cumsum(n,axis=1);pt=tp[:,-1];nt=fp[:,-1]
  valid=(pt>0)&(nt>0)
  auc=np.full(len(p),np.nan);avg=np.full(len(p),np.nan)
  auc[valid]=100*np.sum((tp[valid]-.5*p[valid])*n[valid],axis=1)/(pt[valid]*nt[valid])
  precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
  avg[valid]=100*np.sum(precision[valid]*p[valid],axis=1)/pt[valid]
  auroc.extend(auc);ap.extend(avg)
 return {'auroc':np.asarray(auroc),'ap':np.asarray(ap)}

def draw_video_counts(nvideos,repetitions,rng):
 draws=rng.integers(0,nvideos,size=(repetitions,nvideos));counts=np.zeros((repetitions,nvideos),dtype=int)
 for row in range(repetitions):counts[row]=np.bincount(draws[row],minlength=nvideos)
 return counts

def interval(values):
 a=np.asarray(values);valid=a[np.isfinite(a)]
 return {'low':float(np.percentile(valid,2.5)) if len(valid) else None,'high':float(np.percentile(valid,97.5)) if len(valid) else None,'valid_resamples':len(valid),'undefined_resamples':len(a)-len(valid)}
