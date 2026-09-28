"""Consensus-fixed bootstrap, alignment and within-video permutation diagnostics."""
import csv,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write,readlines,SCENES,segments
from scripts.vera_stage5_statistics import rank_buckets,weighted_rank_metrics,interval
from scripts.vera_stage6_statistics import stratified_video_counts,permutation_indices
from scripts.evaluate_vera_normal_context import binary
from ipad.vera import refine_scores
OUT=ROOT/'experiments/stage6';STAGES=['initial','retrieved','smoothed','final'];METHODS=['P1','C0','N','X'];CONDITIONS=METHODS+['ZERO','ONE'];PAIRS=[('N','C0'),('N','X'),('C0','P1')]

def bootstrap(rows,manifest,metrics,controls):
 primary=json.loads((ROOT/'experiments/stage5/bootstrap_resamples.json').read_text());videos={s:[r['id'] for r in manifest if r['scene']==s] for s in SCENES};assert videos==primary['videos']
 counts={'primary':{s:np.array(primary['counts'][s],dtype=int) for s in SCENES},'stratified_secondary':{}}
 labels={};rng=np.random.default_rng(0)
 for scene in SCENES:
  labels[scene]=[int(any(int(r['label']) for r in rows if r['condition']=='P1' and f"{r['scene']}/{r['original_split']}/{r['video']}"==v)) for v in videos[scene]]
  counts['stratified_secondary'][scene]=stratified_video_counts(labels[scene],2000,rng)
 for scope in counts:counts[scope]['pooled']=np.concatenate([counts[scope][s] for s in SCENES],axis=1)
 videos['pooled']=sum([videos[s] for s in SCENES],[])
 write(OUT/'bootstrap_resamples.json',{'primary_source':'experiments/stage5/bootstrap_resamples.json','primary_source_sha256':sha(ROOT/'experiments/stage5/bootstrap_resamples.json'),'videos':{s:videos[s] for s in SCENES},'primary_counts':primary['counts'],'stratified_secondary_counts':{s:counts['stratified_secondary'][s].tolist() for s in SCENES},'stratified_secondary_video_labels':labels,'seed':0,'repetitions':2000})
 samples={name:{} for name in counts}
 for scene in [*SCENES,'pooled']:
  index={v:i for i,v in enumerate(videos[scene])}
  for scope in samples:samples[scope][scene]={}
  for c in CONDITIONS:
   part=[r for r in rows if r['condition']==c and (scene=='pooled' or r['scene']==scene)];y=np.array([int(r['label']) for r in part]);v=np.array([index[f"{r['scene']}/{r['original_split']}/{r['video']}"] for r in part])
   for scope in samples:samples[scope][scene][c]={}
   for stage in STAGES:
    buckets=rank_buckets(y,[float(r[stage]) for r in part],v,len(index));point=weighted_rank_metrics(buckets,np.ones((1,len(index)),dtype=int));target=(controls if c in ['ZERO','ONE'] else metrics)[scene][c][stage]
    for k in ['auroc','ap']:assert abs(point[k][0]-target[k])<1e-9
    for scope in samples:samples[scope][scene][c][stage]=weighted_rank_metrics(buckets,counts[scope][scene])
  print(json.dumps({'event':'bootstrap_scene_complete','scene':scene}),flush=True)
 result={}
 for scope,sample in samples.items():
  sample['macro']={c:{stage:{k:np.mean(np.stack([sample[s][c][stage][k] for s in SCENES]),axis=0) for k in ['auroc','ap']} for stage in STAGES} for c in CONDITIONS}
  intervals={scene:{c:{stage:{k:interval(sample[scene][c][stage][k]) for k in ['auroc','ap']} for stage in STAGES} for c in CONDITIONS} for scene in [*SCENES,'macro','pooled']}
  deltas={scene:{new+'-'+old:{stage:{k:{'point':metrics[scene][new][stage][k]-metrics[scene][old][stage][k],**interval(sample[scene][new][stage][k]-sample[scene][old][stage][k])} for k in ['auroc','ap']} for stage in STAGES} for new,old in PAIRS} for scene in [*SCENES,'macro','pooled']}
  result[scope]={'intervals':intervals,'deltas_percentage_points':deltas}
 write(OUT/'bootstrap.json',{'method':'primary exactly reuses Stage5 scene-paired whole-video draws; secondary additionally stratifies by video any-anomaly label; percentile95%, exploratory only','resamples_sha256':sha(OUT/'bootstrap_resamples.json'),**result})
 return result

def permutations(rows,manifest,windows,metrics):
 rng=np.random.default_rng(1);prepared=[];perm={};y={s:[] for s in [*SCENES,'pooled']};wi={(r['condition'],r['video_id'],r['center']):r['prediction'] for r in windows}
 for video in manifest:
  part=[r for r in rows if r['condition']=='P1' and r['scene']==video['scene'] and r['original_split']==video['original_split'] and r['video']==video['video']];labels=np.array([int(r['label']) for r in part]);segs=segments(video['length'])
  vectors=[r['feature'] for r in readlines(ROOT/'cache/stage3/features'/video['scene']/video['original_split']/(video['video']+'.jsonl'))];orders=permutation_indices(len(segs),200,rng);perm[video['id']]=orders.tolist()
  pred={c:np.array([wi[(c,video['id'],seg['center'])] for seg in segs]) for c in METHODS};prepared.append((video,labels,vectors,orders,pred));y[video['scene']].append(labels);y['pooled'].append(labels)
 y={s:np.concatenate(v) for s,v in y.items()};samples={s:{c:{t:{k:[] for k in ['auroc','ap']} for t in STAGES} for c in METHODS} for s in [*SCENES,'pooled']}
 for draw in range(200):
  collected={s:{c:{t:[] for t in STAGES} for c in METHODS} for s in [*SCENES,'pooled']}
  for video,labels,vectors,orders,pred in prepared:
   for c in METHODS:
    scores,_=refine_scores(pred[c][orders[draw]],vectors,video['length'])
    for scope in [video['scene'],'pooled']:
     for t in STAGES:collected[scope][c][t].append(scores[t])
  for scope in collected:
   for c in METHODS:
    for t in STAGES:
     x=np.concatenate(collected[scope][c][t]);samples[scope][c][t]['auroc'].append(float(100*roc_auc_score(y[scope],x)));samples[scope][c][t]['ap'].append(float(100*average_precision_score(y[scope],x)))
  if (draw+1)%50==0:print(json.dumps({'event':'permutation_progress','draws':draw+1}),flush=True)
 samples['macro']={c:{t:{k:np.mean(np.array([samples[s][c][t][k] for s in SCENES]),axis=0).tolist() for k in ['auroc','ap']} for t in STAGES} for c in METHODS}
 summary={s:{c:{t:{k:{'observed':metrics[s][c][t][k],'median':float(np.median(v)),**interval(v)} for k,v in ks.items()} for t,ks in ts.items()} for c,ts in cs.items()} for s,cs in samples.items()}
 write(OUT/'permutation_indices.json',{'seed':1,'repetitions':200,'orders_by_video':perm,'same_orders_for_all_conditions':True,'unit':'initial segment predictions, not frames; each video retains its number of positive predictions'})
 write(OUT/'permutation_diagnostics.json',{'interpretation':'Computation-only within-video shuffle diagnostic; not confirmatory p-values; frame expansion can change positive frame totals in a partial final segment','indices_sha256':sha(OUT/'permutation_indices.json'),'summary':summary,'samples':samples})

def main():
 assert json.loads((OUT/'independent_verification.json').read_text())['status']=='passed'
 rows=list(csv.DictReader((OUT/'frame_scores.csv').open()));manifest=json.loads((OUT/'inference_manifest.json').read_text())['evaluation'];metrics=json.loads((OUT/'metrics.json').read_text());windows=json.loads((OUT/'window_diagnostics.json').read_text())
 control_rows=list(csv.DictReader((ROOT/'experiments/stage5/constant_control_scores.csv').open()));controls=json.loads((ROOT/'experiments/stage5/constant_control_metrics.json').read_text())
 support={(r['scene'],r['original_split'],r['video'],r['frame']):r['label'] for r in rows if r['condition']=='P1'}
 for c in ['ZERO','ONE']:assert {(r['scene'],r['original_split'],r['video'],r['frame']):r['label'] for r in control_rows if r['condition']==c}==support
 # Reconstruct both constant controls with the unchanged query-only postprocessing.
 for video in manifest:
  vectors=[r['feature'] for r in readlines(ROOT/'cache/stage3/features'/video['scene']/video['original_split']/(video['video']+'.jsonl'))]
  for c,value in [('ZERO',0),('ONE',1)]:
   calculated,_=refine_scores([value]*len(segments(video['length'])),vectors,video['length']);part=[r for r in control_rows if r['condition']==c and r['scene']==video['scene'] and r['original_split']==video['original_split'] and r['video']==video['video']]
   for stage in STAGES:assert np.allclose(calculated[stage],[float(r[stage]) for r in part],rtol=0,atol=1e-14)
 write(OUT/'constant_control_metrics.json',controls);write(OUT/'reused_control_sources.json',{'scores':'experiments/stage5/constant_control_scores.csv','scores_sha256':sha(ROOT/'experiments/stage5/constant_control_scores.csv'),'metrics_sha256':sha(ROOT/'experiments/stage5/constant_control_metrics.json'),'independently_reconstructed':True})
 result=bootstrap(rows+control_rows,manifest,metrics,controls)
 alignment={}
 for scene in [*SCENES,'pooled']:
  alignment[scene]={}
  for c in METHODS:
   part=[r for r in windows if r['condition']==c and (scene=='pooled' or r['scene']==scene)];pred=[r['prediction'] for r in part];yi=[int(r['sampled_positive_frames']>0) for r in part];yt=[int(r['scored_positive_frames']>0) for r in part]
   strata={}
   for name,lo,hi in [('0',0,0),('1-4',1,4),('5-8',5,8)]:
    subset=[r for r in part if lo<=r['sampled_positive_frames']<=hi];pos=[r for r in subset if r['scored_positive_frames']>0];neg=[r for r in subset if r['scored_positive_frames']==0]
    strata[name]={'windows':len(subset),'predicted_positive':sum(r['prediction'] for r in subset),'target_positive_windows':len(pos),'recall':sum(r['prediction'] for r in pos)/len(pos) if pos else None,'fpr':sum(r['prediction'] for r in neg)/len(neg) if neg else None}
   alignment[scene][c]={'windows':len(part),'predicted_positive':sum(pred),'positive_response_rate':sum(pred)/len(pred),'input_any_target_any':{f'input{i}_target{j}':sum(a==i and b==j for a,b in zip(yi,yt)) for i in [0,1] for j in [0,1]},'input_any_confusion':binary(yi,pred),'target_any_confusion':binary(yt,pred),'sampled_positive_strata':strata}
 write(OUT/'alignment_diagnostics.json',alignment);permutations(rows,manifest,windows,metrics)
 sources=['scripts/analyze_vera_stage6.py','scripts/vera_stage6_statistics.py','scripts/vera_stage5_statistics.py','experiments/stage6/frame_scores.csv','experiments/stage6/window_diagnostics.json','experiments/stage5/bootstrap_resamples.json']
 write(OUT/'analysis_verification.json',{'status':'passed','bootstrap_point_estimates_match_sklearn':True,'primary_resamples_exact_Stage5':True,'secondary_stratified_separate':True,'constant_controls_reconstructed':True,'permutations_same_across_methods':True,'no_model_calls':True,'source_sha256':{p:sha(ROOT/p) for p in sources}})
 print(json.dumps({'event':'analysis_complete','primary_macro_initial':result['primary']['deltas_percentage_points']['macro']['N-C0']['initial']},indent=2))
if __name__=='__main__':main()
