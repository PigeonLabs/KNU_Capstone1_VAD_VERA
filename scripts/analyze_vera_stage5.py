"""Preregistered paired video uncertainty and non-inference score diagnostics."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write,digest,readlines,SCENES
from scripts.vera_stage5_statistics import rank_buckets,weighted_rank_metrics,draw_video_counts,interval
from scripts.evaluate_vera_normal_context import metric,binary
from ipad.vera import refine_scores,segments
OUT=ROOT/'experiments/stage5';STAGES=['initial','retrieved','smoothed','final'];CONDITIONS=['A','B','C','P1','P2','P3','ZERO','ONE'];PAIRS=[('P3','P2'),('P2','P1'),('P3','A')]

def main():
 assert json.loads((OUT/'independent_verification.json').read_text())['status']=='passed'
 rows=list(csv.DictReader((OUT/'frame_scores.csv').open()));manifest=json.loads((OUT/'inference_manifest.json').read_text())['evaluation'];metrics=json.loads((OUT/'metrics.json').read_text());controls=[]
 for video in manifest:
  scene=video['scene'];part=[r for r in rows if r['condition']=='A' and r['scene']==scene and r['original_split']==video['original_split'] and r['video']==video['video']]
  assert len(part)==video['length']
  vectors=[r['feature'] for r in readlines(ROOT/'cache/stage3/features'/scene/video['original_split']/(video['video']+'.jsonl'))]
  for name,value in [('ZERO',0),('ONE',1)]:
   scores,_=refine_scores([value]*len(segments(video['length'])),vectors,video['length'])
   for i,base in enumerate(part):controls.append({**base,'condition':name,**{k:float(scores[k][i]) for k in STAGES}})
 with (OUT/'constant_control_scores.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(controls[0]));w.writeheader();w.writerows(controls)
 controls_metrics={s:{c:{stage:metric([r for r in controls if r['condition']==c and (s=='pooled' or r['scene']==s)],stage) for stage in STAGES} for c in ['ZERO','ONE']} for s in [*SCENES,'pooled']}
 controls_metrics['macro']={c:{stage:{k:float(np.mean([controls_metrics[s][c][stage][k] for s in SCENES])) for k in ['auroc','ap']} for stage in STAGES} for c in ['ZERO','ONE']}
 write(OUT/'constant_control_metrics.json',controls_metrics)
 rows+=controls;repetitions=2000;rng=np.random.default_rng(0);counts={};videos={};samples={};source_hash=sha(OUT/'frame_scores.csv')
 for scene in SCENES:
  videos[scene]=[r['id'] for r in manifest if r['scene']==scene];counts[scene]=draw_video_counts(len(videos[scene]),repetitions,rng)
 counts['pooled']=np.concatenate([counts[s] for s in SCENES],axis=1);videos['pooled']=sum([videos[s] for s in SCENES],[])
 write(OUT/'bootstrap_resamples.json',{'seed':0,'repetitions':repetitions,'method':'with replacement within scene, paired across all conditions, entire video frames retained','videos':{s:videos[s] for s in SCENES},'counts':{s:counts[s].tolist() for s in SCENES}})
 for scene in [*SCENES,'pooled']:
  samples[scene]={};index={v:i for i,v in enumerate(videos[scene])}
  for c in CONDITIONS:
   part=[r for r in rows if r['condition']==c and (scene=='pooled' or r['scene']==scene)];y=np.array([int(r['label']) for r in part]);v=np.array([index[f"{r['scene']}/{r['original_split']}/{r['video']}"] for r in part]);samples[scene][c]={}
   for stage in STAGES:
    x=np.array([float(r[stage]) for r in part]);buckets=rank_buckets(y,x,v,len(index));point=weighted_rank_metrics(buckets,np.ones((1,len(index)),dtype=int))
    target=(controls_metrics if c in ['ZERO','ONE'] else metrics)[scene][c][stage]
    for k in ['auroc','ap']:assert abs(point[k][0]-target[k])<1e-9
    samples[scene][c][stage]=weighted_rank_metrics(buckets,counts[scene])
  print(json.dumps({'event':'bootstrap_scene_complete','scene':scene}),flush=True)
 samples['macro']={c:{stage:{k:np.mean(np.stack([samples[s][c][stage][k] for s in SCENES]),axis=0) for k in ['auroc','ap']} for stage in STAGES} for c in CONDITIONS}
 intervals={};deltas={}
 for scene in [*SCENES,'macro','pooled']:
  intervals[scene]={c:{stage:{k:interval(samples[scene][c][stage][k]) for k in ['auroc','ap']} for stage in STAGES} for c in CONDITIONS}
  deltas[scene]={}
  for new,old in PAIRS:
   deltas[scene][new+'-'+old]={stage:{k:{'point':metrics[scene][new][stage][k]-metrics[scene][old][stage][k],**interval(samples[scene][new][stage][k]-samples[scene][old][stage][k])} for k in ['auroc','ap']} for stage in STAGES}
 write(OUT/'bootstrap.json',{'method':'scene-stratified paired video bootstrap; percentile95%; 2000 draws seed0; undefined resamples retained as undefined; exploratory only','frame_scores_sha256':source_hash,'resamples_sha256':sha(OUT/'bootstrap_resamples.json'),'intervals':intervals,'deltas_percentage_points':deltas})
 windows=json.loads((OUT/'window_diagnostics.json').read_text());alignment={}
 for scene in [*SCENES,'pooled']:
  alignment[scene]={}
  for c in CONDITIONS[:6]:
   part=[r for r in windows if r['condition']==c and (scene=='pooled' or r['scene']==scene)];pred=[r['prediction'] for r in part];yi=[int(r['sampled_positive_frames']>0) for r in part];yt=[int(r['scored_positive_frames']>0) for r in part]
   table={f'input{i}_target{j}':sum(a==i and b==j for a,b in zip(yi,yt)) for i in [0,1] for j in [0,1]}
   strata={}
   for name,lo,hi in [('0',0,0),('1-4',1,4),('5-8',5,8)]:
    p=[r for r in part if lo<=r['sampled_positive_frames']<=hi];target=[r for r in p if r['scored_positive_frames']>0];negative=[r for r in p if not r['scored_positive_frames']]
    strata[name]={'windows':len(p),'predicted_positive':sum(r['prediction'] for r in p),'target_positive_windows':len(target),'recall_target_any':sum(r['prediction'] for r in target)/len(target) if target else None,'fpr_target_allnormal':sum(r['prediction'] for r in negative)/len(negative) if negative else None}
   alignment[scene][c]={'windows':len(part),'predicted_positive':sum(pred),'positive_response_rate':sum(pred)/len(pred),'input_any_target_any':table,'input_any_confusion':binary(yi,pred),'target_any_confusion':binary(yt,pred),'sampled_positive_strata':strata}
 write(OUT/'alignment_diagnostics.json',alignment)
 write(OUT/'analysis_verification.json',{'status':'passed','all_bootstrap_point_estimates_match_sklearn':True,'scene_stratified_paired_video_sampling':True,'no_frame_bootstrap':True,'constant_controls_are_computation_not_model_inference':True,'constant_initial_auroc':{c:controls_metrics['pooled'][c]['initial']['auroc'] for c in ['ZERO','ONE']},'source_sha256':{p:sha(ROOT/p) for p in ['scripts/analyze_vera_stage5.py','scripts/vera_stage5_statistics.py','experiments/stage5/frame_scores.csv','experiments/stage5/window_diagnostics.json']}})
 print(json.dumps({'event':'analysis_complete','primary_macro_initial_delta':deltas['macro']['P3-P2']['initial'],'all_one_macro_final':controls_metrics['macro']['ONE']['final']},indent=2))
if __name__=='__main__':main()
