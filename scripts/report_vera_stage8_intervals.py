import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import OUT,REPORT

def main():
 lines=['# 8단계 전체 비교 신뢰구간','','장면 내 영상 단위 paired bootstrap, seed=0, 2,000회. 단일 클래스가 생긴 macro resample은 정의 불가로 별도 집계합니다. AUROC와 AP는 sklearn로 계산하며 아래 단위는 %p입니다. 95% CI가 0을 포함하면 차이를 검출하지 못한 것으로 해석하고 동등성은 주장하지 않습니다. 이 CI는 고정된 조건의 평가 영상 불확실성이며 학습·질문·가중치 선택을 bootstrap마다 다시 수행하지 않았습니다.','', '| 단계·비교 | 초기 AUROC CI | 초기 AP CI | 최종 AUROC CI | 최종 AP CI | 무효/전체 |','|---|---:|---:|---:|---:|---:|']
 for step in ['step0','step1','step4','step5']:
  for p in sorted((OUT/step/'bootstrap').glob('*.json')):
   v=json.loads(p.read_text());ci=v['ci95'];f=lambda s,m:'['+', '.join(f'{x:.3f}' for x in ci[s][m])+']'
   lines.append(f"| {step} {v['left']}−{v['right']} | {f('initial','auroc')} | {f('initial','ap')} | {f('final','auroc')} | {f('final','ap')} | {v['undefined_single_class']}/{v['n']} |")
 (REPORT/'confidence_intervals.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
