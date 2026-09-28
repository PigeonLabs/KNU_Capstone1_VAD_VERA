"""Executed sanity diagnostics, including intentionally label-informed oracle control."""
import sys,json,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.vera_stage8_common import *
from scripts.vera_stage8_eval import refine,summarize,metric,STAGES

def main():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import scripts.train_vera_questions as base
 base.OUT=OUT;base.guard();protected(init=True);evaluator_guard(init=True)
 evaluation=[r for r in rows() if r['split']=='evaluation'];rng=np.random.default_rng(0);output=[];windows=[]
 write(OUT/'protocol.json',{'stage':8,'exploratory':True,'holdout':'No unseen holdout available; user approved exploratory only; confirmatory claim deferred','seed':0,'position_default':'off (smoothed); final column always reports original position ON ablation','step1_gate':'macro initial PROB >=55: reference/decomposition; all values <55: hybrid route, 47..53 triggers user-prescribed branch, otherwise inconclusive extension of conservative gate','oracle':'max ground-truth in scored center:score_end block; diagnostic leakage control, not mathematical upper bound','fps':30,'fps_verified':False,'score_threshold_for_positive_rate':0.5,'no_result_based_tuning':True})
 for r in evaluation:
  f=features(r);y=labels(r);ss=segments(r['length'])
  for c,x in [('RANDOM',rng.uniform(size=len(ss))),('ZERO',np.zeros(len(ss))),('ONE',np.ones(len(ss))),('ORACLE',np.array([max(y[s['center']:s['score_end']]) for s in ss]))]:
   score=refine(x,f,r['length'])
   windows.extend({'condition':c,'video_id':r['id'],'center':s['center'],'raw_score':float(v)} for s,v in zip(ss,x))
   output.extend({'condition':c,'scene':r['scene'],'original_split':r['original_split'],'video':r['video'],'frame':i,'label':int(y[i]),**{s:float(v[i]) for s,v in score.items()}} for i in range(r['length']))
 writecsv(OUT/'step0/frame_scores.csv',output);writecsv(OUT/'step0/window_scores.csv',windows);metrics=summarize(output);write(OUT/'step0/metrics.json',metrics)
 oldpaths=[ROOT/'experiments/vera_ipad/frame_scores.csv',ROOT/'experiments/stage3/frame_scores.csv',ROOT/'experiments/stage5/frame_scores.csv']+sorted((ROOT/'experiments/stage4').glob('*/evaluation/frame_scores.csv'))
 inventory=[];hist=[]
 for p in oldpaths:
  if not p.exists():continue
  rr=list(csv.DictReader(p.open()));inventory.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
  for c in sorted({r.get('condition','default') for r in rr}):
   part=[r for r in rr if r.get('condition','default')==c];v=np.array([float(r['initial']) for r in part]);m=metric([int(r['label']) for r in part],v)
   counts={str(float(k)):int(n) for k,n in zip(*np.unique(v,return_counts=True))}
   hist.append({'source':str(p.relative_to(ROOT)),'condition':c,'frames':len(v),'n_unique_scores':len(counts),'positive_fraction':float((v>=.5).mean()),'initial_pooled':m,'constant_rank_collapse':len(counts)==1,'user_diagnostic_flag':len(counts)<=2 and m['auroc']==50.,'histogram':counts})
 write(OUT/'step0/historical_score_distributions.json',hist);write(OUT/'step0/source_inventory.json',inventory)
 fig,axes=plt.subplots(1,2,figsize=(12,4));names=[h['source'].split('/')[1]+'/'+h['condition'] for h in hist]
 axes[0].bar(range(len(hist)),[h['n_unique_scores'] for h in hist]);axes[0].set_title('Initial distinct frame scores');axes[0].set_xticks(range(len(hist)),names,rotation=90,fontsize=6)
 axes[1].bar(range(len(hist)),[h['positive_fraction'] for h in hist]);axes[1].set_title('Initial positive fraction');axes[1].set_xticks(range(len(hist)),names,rotation=90,fontsize=6)
 fig.tight_layout();fig.savefig(REPORT/'step0_score_distribution.svg');plt.close(fig)
 # Cached old DINO has only original testing and excludes temporal boundary frames.
 lookup={(r['scene'],r['video'],r['frame']):r for r in output if r['condition']=='ONE' and r['original_split']=='testing'};pairs=[];inv=[]
 for sc in SCENES:
  p=DATA/'experiments/stage2_dinov2'/sc/'prototype/scores.csv';inv.append({'path':str(p.relative_to(DATA)),'sha256':sha(p)})
  for old in csv.DictReader(p.open()):
   k=(sc,old['video'],int(old['frame']))
   if k not in lookup:continue
   new=lookup[k];assert int(old['label'])==new['label'];pairs.append({**new,'dino':float(old['unconditional_nn'])})
 dino={sc:{c:metric([r['label'] for r in pairs if sc=='pooled' or r['scene']==sc],[r[c] for r in pairs if sc=='pooled' or r['scene']==sc]) for c in ['dino','initial','final']} for sc in [*SCENES,'pooled']}
 dino['macro']={c:{m:float(np.mean([dino[sc][c][m] for sc in SCENES])) for m in ['auroc','ap']} for c in ['dino','initial','final']}
 write(OUT/'step0/dino_comparison.json',{'frames':len(pairs),'evaluation_frames':sum(r['length'] for r in evaluation),'support':'strict original-testing frame intersection; old DINO trained original normal-training, unavailable on resplit-normal evaluation; NOT full 37-video baseline','metrics':dino,'sources':inv});writecsv(OUT/'step0/dino_matched_scores.csv',pairs)
 videos=[str(p.relative_to(DATA)) for p in (DATA/'IPAD_dataset').rglob('*') if p.is_file() and p.suffix.lower() in ['.mp4','.avi','.mov','.mkv','.webm']]
 write(OUT/'step0/fps_inventory.json',{'original_video_files':videos,'observed_fps':None,'assumption':30,'reason':'Only extracted frame files found; ffprobe cannot recover original recording FPS from JPG sequences' if not videos else 'Original containers discovered; probing required'})
 lines=['# 8단계 Step 0 — 점수·후처리 진단','','기존 37영상·16,862프레임의 탐색적 진단입니다. 미관측 hold-out은 없으며 확증 평가는 보류합니다.','', '| 조건 | 초기 AUROC / AP (%) | 최종 AUROC / AP (%) | 최종 Δ vs ONE (pp) |','|---|---:|---:|---:|']
 for c,m in metrics.items():
  a=m['macro']['initial'];b=m['macro']['final'];one=metrics['ONE']['macro']['final'];lines.append(f"| {c} | {a['auroc']:.4f} / {a['ap']:.4f} | {b['auroc']:.4f} / {b['ap']:.4f} | {b['auroc']-one['auroc']:.4f} / {b['ap']-one['ap']:.4f} |")
 lines+=['','![점수 분포](step0_score_distribution.svg)','','ORACLE은 각 점수 블록의 정답 최대값을 주입한 진단용 대조군이며 모델 결과나 이론적 상한이 아닙니다.',f"ORACLE 최종 macro AUROC={metrics['ORACLE']['macro']['final']['auroc']:.4f}%; 90 미만이면 시간 창/후처리 진단을 우선 수행합니다.",f"기존 DINOv2 무조건부 NN: 동일 프레임 교집합 {len(pairs)}개, macro AUROC {dino['macro']['dino']['auroc']:.4f}%, AP {dino['macro']['dino']['ap']:.4f}%. 37영상 전체 기준선으로 표현하지 않습니다.",'DINOv2 원 메모리는 원래 정상 training 전체를 사용했습니다. 재분할 validation/evaluation에 쓸 하이브리드는 해당 영상들을 제외한 메모리를 새로 학습해야 합니다.','초기 점수 1종이면 순위 소실이 확정됩니다. 2종이고 AUROC=50이라는 사실만으로 원인이 확정되지는 않습니다.','ONE은 위치 prior 대조군입니다. 최초 점수의 우연 AUROC 기준은 여전히 50%입니다.','원본 컨테이너가 없어 실제 FPS는 검증할 수 없습니다. 30 FPS 가정을 유지하며 JPEG를 영상으로 만들어 측정값처럼 사용하지 않습니다.','학습·추론 없음. 원시 점수, 지표, 입력 SHA는 experiments/stage8/step0에 보존합니다.']
 (REPORT/'step0_sanity.md').write_text('\n'.join(lines)+'\n');write(OUT/'step0/status.json',{'status':'diagnostic','complete':True,'protected_files':protected(),'advance_step4':metrics['ORACLE']['macro']['final']['auroc']<90,'finished_at':time.time()})
 print(json.dumps({c:m['macro'] for c,m in metrics.items()},indent=2))
if __name__=='__main__':main()
