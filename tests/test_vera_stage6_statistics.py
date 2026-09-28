import numpy as np
from scripts.vera_stage6_statistics import stratified_video_counts,permutation_indices

def test_paired_stratification_preserves_each_video_class_sample_size():
 labels=np.array([0,0,0,1,1]);a=stratified_video_counts(labels,2000,np.random.default_rng(0));b=stratified_video_counts(labels,2000,np.random.default_rng(0))
 assert np.array_equal(a,b) and np.all(a[:,:3].sum(axis=1)==3) and np.all(a[:,3:].sum(axis=1)==2)
 assert np.any(a>1)

def test_within_video_permutation_preserves_prediction_counts():
 order=permutation_indices(5,200,np.random.default_rng(1));pred=np.array([0,1,0,1,0])
 assert np.all(np.sort(order,axis=1)==np.arange(5)) and np.all(pred[order].sum(axis=1)==2)
 assert np.array_equal(order,permutation_indices(5,200,np.random.default_rng(1)))
 assert np.array_equal(permutation_indices(1,200,np.random.default_rng(1)),np.zeros((200,1)))
