import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import *
from scripts.vera_stage8_eval import paired_bootstrap,STAGES

def load(p):
 rr=list(csv.DictReader(p.open()))
 for r in rr:
  r['frame']=int(r['frame']);r['label']=int(r['label'])
  for s in STAGES:r[s]=float(r[s])
 return rr

def main(step):
 evaluator_guard();rr=load(OUT/'step0/frame_scores.csv');pairs=[('RANDOM','ONE'),('ZERO','ONE'),('ORACLE','ONE')]
 if step=='step1':rr=[r for r in rr if r['condition']=='ONE']+load(OUT/'step1/frame_scores.csv');pairs=[('PROB','ONE'),('DEC','ONE'),('PROB','DEC')]
 for a,b in pairs:
  if not {a,b}<={r['condition'] for r in rr}:continue
  p=OUT/step/'bootstrap'/f'{a}_vs_{b}.json'
  if p.exists():continue
  result=paired_bootstrap(rr,a,b);write(p,result);print(json.dumps({'comparison':f'{a}-{b}','ci95':result['ci95'],'undefined':result['undefined_single_class']}),flush=True)
 if step=='step1':
  m=json.loads((OUT/step/'metrics.json').read_text());one=json.loads((OUT/'step0/metrics.json').read_text())['ONE']['macro'];diag=json.loads((OUT/step/'score_diagnostics.json').read_text());branch=json.loads((OUT/step/'branch.json').read_text())
  lines=['# 8단계 Step 1 — 같은 forward의 DEC / PROB 비교','','InternVL2-8B BF16, greedy 1토큰, 공개 UCF 질문 고정. 추가 학습 없음. 37영상 탐색적 평가.','', '| 조건 | 초기 AUROC/AP (%) | 최종 AUROC/AP (%) | 최종 Δ vs ONE (pp) |','|---|---:|---:|---:|']
  for c,d in m.items():
   a=d['macro']['initial'];b=d['macro']['final'];lines.append(f"| {c} | {a['auroc']:.4f} / {a['ap']:.4f} | {b['auroc']:.4f} / {b['ap']:.4f} | {b['auroc']-one['final']['auroc']:.4f} / {b['ap']-one['final']['ap']:.4f} |")
  lines+=['','| 조건 | 초기 | 검색 | smoothing (위치 OFF) | 위치 ON |','|---|---:|---:|---:|---:|']
  for c,d in m.items():lines.append('| '+c+' | '+' | '.join(f"{d['macro'][s]['auroc']:.4f} / {d['macro'][s]['ap']:.4f}" for s in STAGES)+' |')
  lines+=['','점수 종류·양성 비율: `'+json.dumps(diag)+'`','', '영상 단위 장면 내 paired bootstrap 2,000회. AUROC/AP 차이의 95% CI가 0을 포함하면 검출된 차이 없음으로 해석하며 동등성은 주장하지 않습니다.']
  for p in sorted((OUT/step/'bootstrap').glob('*.json')):
   b=json.loads(p.read_text());lines.append(f"- {b['left']}−{b['right']}: 초기 AUROC CI {b['ci95']['initial']['auroc']}, 최종 {b['ci95']['final']['auroc']}; 단일 클래스 무효 {b['undefined_single_class']}/2000 ({b['undefined_rate']:.2%}).")
  lines+=['',f"사전 분기 적용: {branch['route']}. 초기 macro AUROC {branch['initial_macro_auroc']:.4f}%. 이 분기는 시각 인식 한계를 입증하는 검정이 아닙니다.",'No 부분집합 지표: [JSON](../../experiments/stage8/step1/decoded_no_subset.json).','토큰 변형과 실제 모델 logits, 생성 토큰, 프롬프트 SHA, 원본 이미지 SHA, 시간·VRAM은 experiments/stage8/step1에 보존합니다.','[점수화 연구](https://arxiv.org/abs/2608.21244)는 실험 동기이며 그 논문의 성능 향상을 본 데이터의 예상 결과로 취급하지 않습니다.']
  (REPORT/'step1_prob_vs_dec.md').write_text('\n'.join(lines)+'\n')
 else:
  p=REPORT/'step0_sanity.md';text=p.read_text()+'\n영상 단위 paired bootstrap 2,000회 결과:\n\n'
  for path in sorted((OUT/step/'bootstrap').glob('*.json')):
   b=json.loads(path.read_text());text+=f"- {b['left']}−ONE: 초기 AUROC CI {b['ci95']['initial']['auroc']}, 최종 {b['ci95']['final']['auroc']}; 무효 {b['undefined_single_class']}/2000.\n"
  p.write_text(text)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',choices=['step0','step1'],required=True);main(p.parse_args().step)
