"""Report actually executed window/hybrid stages using the locked evaluator."""
import argparse,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.vera_stage8_common import *
from scripts.vera_stage8_eval import paired_bootstrap,STAGES
from scripts.analyze_vera_stage8 import load

def main(step):
 evaluator_guard();protected();rr=load(OUT/step/'frame_scores.csv');rr += [r for r in load(OUT/'step0/frame_scores.csv') if r['condition']=='ONE'];conditions=sorted({r['condition'] for r in rr if r['condition']!='ONE'})
 pairs=[(c,'ONE') for c in conditions]
 pairs += [('HYBRID','DINO'),('HYBRID','PROB')] if step=='step5' else [('PROB_2s','PROB_10s'),('PROB_4s','PROB_10s')]
 for a,b in pairs:
  p=OUT/step/'bootstrap'/f'{a}_vs_{b}.json'
  if p.exists():continue
  # Reuse PROB vs ONE and 10s vs ONE from byte-identical cached Step1 scores.
  if a in ['PROB','PROB_10s'] and b=='ONE':
   old=json.loads((OUT/'step1/bootstrap/PROB_vs_ONE.json').read_text());old.update(left=a,reused_source='experiments/stage8/step1/bootstrap/PROB_vs_ONE.json');write(p,old);continue
  value=paired_bootstrap(rr,a,b);write(p,value);print(json.dumps({'comparison':f'{a}-{b}','ci95':value['ci95']}),flush=True)
 m=json.loads((OUT/step/'metrics.json').read_text());one=json.loads((OUT/'step0/metrics.json').read_text())['ONE'];title='검증 선택 하이브리드' if step=='step5' else '시간 창·후처리 분해'
 lines=[f'# 8단계 {step} — {title}','','37영상 탐색적 평가. 설정은 평가 전에 고정하고 결과에 따라 재선택하지 않았습니다. 위치 가중치는 기본 OFF(smoothed)이며 원형 최종 점수(final, ON)를 병기합니다.','', '| 조건 | 초기 AUROC/AP (%) | 검색 | smoothing (OFF) | 최종 (ON) | 최종 Δ vs ONE (pp) |','|---|---:|---:|---:|---:|---:|']
 for c,d in m.items():
  v=d['macro'];delta={k:v['final'][k]-one['macro']['final'][k] for k in ['auroc','ap']};lines.append('| '+c+' | '+' | '.join(f"{v[s]['auroc']:.4f} / {v[s]['ap']:.4f}" for s in STAGES)+f" | {delta['auroc']:.4f} / {delta['ap']:.4f} |")
 lines+=['','영상 단위 paired bootstrap 2,000회(장면 내 재표집):']
 for p in sorted((OUT/step/'bootstrap').glob('*.json')):
  b=json.loads(p.read_text());lines.append(f"- {b['left']}−{b['right']}: 초기 AUROC CI {b['ci95']['initial']['auroc']}, 최종 {b['ci95']['final']['auroc']}; 단일 클래스 무효 {b['undefined_single_class']}/2000. 0을 포함하면 검출된 차이 없음.")
 diag={}
 for c in conditions:
  a=[r for r in rr if r['condition']==c];threshold=0 if c in ['DINO','HYBRID'] else .5;x=np.array([r['initial'] for r in a]);diag[c]={'initial_frame_unique':len(np.unique(x)),'score_threshold':threshold,'score_at_or_above_threshold_fraction':float((x>=threshold).mean()),'threshold_is_diagnostic_not_calibrated_alarm':True}
 write(OUT/step/'score_diagnostics.json',diag)
 if step=='step5':
  lock=json.loads((OUT/step/'selection_lock.json').read_text());lines+=['',f"검증에서 선택한 λ={lock['selected_lambda']}. 후보는 0, 0.1, 0.25, 0.5, 1, 2. 동일 점수면 작은 λ를 선택합니다.",'정규화는 장면별 validation 전체 프레임 평균·표준편차만 사용합니다. DINO 메모리는 재분할 normal train만 사용하며 검증·평가 영상과 교집합이 없습니다.','초기 하이브리드는 원래 프레임별 DINO z + λ·VLM z입니다. 검색부터는 각 16프레임 블록 평균에 VERA 후처리를 적용하므로 초기와 후처리 사이에 집계도 포함됩니다. signed z 점수에 위치 가중치를 적용한 결과임을 유의해야 합니다.','[검증 선택 이력](../../experiments/stage8/step5/selection_lock.json) · [장면별 지표](../../experiments/stage8/step5/metrics.json)','상위 점수 구간의 설명은 별도 호출이며 정답 주석이나 점수 변경 근거로 사용하지 않습니다.']
  filename='step5_hybrid.md'
 else:
  lines+=['','2/4/10초는 실제 FPS가 검증되지 않은 30 FPS 가정에 해당합니다. 원본 컨테이너가 없어 ffprobe 실측과 25/30 FPS 교정 비교는 수행할 수 없었습니다.','2초·4초의 ImageBind 특징은 해당 창에서 계산했습니다. 10초 VLM 점수와 특징은 해시 검증 후 재사용했습니다.','창 길이별 높은 테스트 점수를 선택하여 기본값으로 변경하지 않았습니다.'];filename='step4_temporal.md'
 (REPORT/filename).write_text('\n'.join(lines)+'\n')
 print('Report complete',step)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',choices=['step4','step5'],required=True);main(p.parse_args().step)
