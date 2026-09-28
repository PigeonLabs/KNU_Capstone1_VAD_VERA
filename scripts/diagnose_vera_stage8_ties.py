import sys,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import OUT,REPORT,write
p=OUT/'step1';rows=[json.loads(f.read_text()) for f in (p/'inference').rglob('*.json')];gaps=np.array([r['yes_logit']-r['no_logit'] for r in rows]);v={'calls':len(rows),'unique_logit_gaps':len(np.unique(gaps)),'logit_gap_values':np.unique(gaps).tolist(),'unique_probabilities':len({r['prob'] for r in rows}),'extra_rounding_applied':False,'model_precision':'BF16','scoring_precision':'raw LM logits float32; two-token softmax float64','observation':'Probability scores retain ties; not every segment has a distinct score'}
write(p/'probability_ties.json',v)
d=json.loads((p/'decoded_no_subset.json').read_text());macro={k:float(np.mean([d['metrics'][s][k] for s in ['R01','R02','R03','R04']])) for k in ['auroc','ap']};write(p/'decoded_no_macro.json',macro)
r=REPORT/'step1_prob_vs_dec.md';s=r.read_text();s+=f'\nNo로 디코딩된 부분집합은 {d["frames"]}프레임이며 PROB macro AUROC/AP={macro["auroc"]:.4f}/{macro["ap"]:.4f}%입니다. pooled와 macro는 서로 다른 값이므로 혼용하지 않습니다. BF16 logits에 추가 반올림을 하지 않았지만 확률은 28종으로 여전히 동점이 남아 있습니다. [logit 차이](../../experiments/stage8/step1/probability_ties.json).\n';r.write_text(s)
print(json.dumps({'no_subset_macro':macro,'logit_gaps':v['logit_gap_values']}))
