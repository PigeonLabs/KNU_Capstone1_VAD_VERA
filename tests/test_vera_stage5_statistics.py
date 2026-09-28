import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from scripts.vera_stage5_statistics import rank_buckets,weighted_rank_metrics,draw_video_counts,interval

def test_ties_and_video_multiplicity_equal_full_resampling():
 y=np.array([0,1,1,0,1,0]);x=np.array([.2,.2,.9,.6,.6,.1]);v=np.array([0,0,1,1,2,2])
 counts=np.array([[1,1,1],[2,1,0],[0,2,1],[3,0,0]])
 metrics=weighted_rank_metrics(rank_buckets(y,x,v,3),counts,batch=2)
 for i,c in enumerate(counts):
  indices=np.repeat(np.arange(len(y)),c[v]);assert abs(metrics['auroc'][i]-100*roc_auc_score(y[indices],x[indices]))<1e-12
  assert abs(metrics['ap'][i]-100*average_precision_score(y[indices],x[indices]))<1e-12

def test_degenerate_resamples_are_undefined_not_replaced():
 b=rank_buckets([0,0,1,1],[0,0,0,0],[0,0,1,1],2)
 m=weighted_rank_metrics(b,np.array([[2,0],[0,2],[1,1]]))
 assert np.isnan(m['auroc'][:2]).all() and m['auroc'][2]==50 and m['ap'][2]==50
 assert interval(m['auroc'])==dict(low=50.,high=50.,valid_resamples=1,undefined_resamples=2)

def test_resampling_deterministic_and_whole_video():
 a=draw_video_counts(9,2000,np.random.default_rng(0));b=draw_video_counts(9,2000,np.random.default_rng(0))
 assert np.array_equal(a,b) and (a.sum(axis=1)==9).all() and (a>=0).all()
