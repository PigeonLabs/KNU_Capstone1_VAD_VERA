"""Observed numerical/temporal effects of the fixed all-one control, no method edits."""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.vera_stage8_common import OUT,REPORT,write
from scripts.vera_stage8_eval import STAGES
rows=[r for r in csv.DictReader((OUT/'step0/frame_scores.csv').open()) if r['condition']=='ONE'];value={}
for s in STAGES:
 x=np.array([float(r[s]) for r in rows]);value[s]={'minimum':float(x.min()),'maximum':float(x.max()),'n_unique_float64_scores':len(np.unique(x)),'max_deviation_from_one':float(np.max(np.abs(x-1)))}
write(OUT/'step0/one_numerical_diagnostics.json',value)
p=REPORT/'step0_sanity.md';s=p.read_text();s+='\nONE은 검색 이후 부동소수점 합산의 미세 차이가 생기며(최대 1과의 차이는 위 진단 JSON 참조), 이를 인위적으로 반올림하지 않았습니다. smoothing만 적용해도 macro AUROC가 56.1072%이고 위치 가중치 적용 후 58.9537%입니다. 따라서 58.95% 전체를 위치 가중치 하나의 효과로 귀속하지 않습니다. 원형 zero-padding smoothing도 상수 입력의 영상 경계 점수를 바꿉니다. [수치 진단](../../experiments/stage8/step0/one_numerical_diagnostics.json).\n';p.write_text(s)
print(json.dumps(value,indent=2))
