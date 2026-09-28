"""Prespecified Stage6 secondary resampling, separate from the unchanged primary."""
import numpy as np

def stratified_video_counts(labels,repetitions,rng):
 labels=np.asarray(labels,dtype=int);assert np.isin(labels,[0,1]).all()
 counts=np.zeros((repetitions,len(labels)),dtype=int)
 for label in [0,1]:
  group=np.flatnonzero(labels==label)
  if not len(group):continue
  draws=rng.choice(group,size=(repetitions,len(group)),replace=True)
  for i,row in enumerate(draws):counts[i]+=np.bincount(row,minlength=len(labels))
 return counts

def permutation_indices(length,repetitions,rng):
 assert length>0
 return np.stack([rng.permutation(length) for _ in range(repetitions)])
