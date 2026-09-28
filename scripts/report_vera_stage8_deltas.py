import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import OUT,REPORT,write
from scripts.vera_stage8_eval import STAGES

def main():
 one=json.loads((OUT/'step0/metrics.json').read_text())['ONE'];out={};lines=['# 8단계 ONE 대비 차이','','동일 37영상·16,862프레임, macro AUROC/AP %p. ONE의 smoothing은 zero padding 경계 효과를 포함하며 최종은 위치 가중치도 포함합니다.','', '| 단계·조건 | 초기 Δ AUROC/AP | 검색 Δ | smoothing Δ | 최종 Δ |','|---|---:|---:|---:|---:|']
 for step in ['step0','step1','step4','step5']:
  if not (OUT/step/'metrics.json').exists():continue
  out[step]={}
  for c,m in json.loads((OUT/step/'metrics.json').read_text()).items():
   delta={s:{k:m['macro'][s][k]-one['macro'][s][k] for k in ['auroc','ap']} for s in STAGES};out[step][c]=delta
   lines.append('| '+step+' '+c+' | '+' | '.join(f"{delta[s]['auroc']:.4f} / {delta[s]['ap']:.4f}" for s in STAGES)+' |')
 write(OUT/'delta_vs_ONE.json',out);(REPORT/'delta_vs_ONE.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
