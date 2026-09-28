"""Historical Stage2 predictions have video-level supervision, not frame-score AUROC."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import OUT,write,sha
from collections import Counter
from scripts.train_vera_questions import parse_response

def main():
 output=[]
 for name in ['learner_responses.jsonl','validation_responses.jsonl']:
  p=ROOT/'experiments/stage2_2'/name;rr=[json.loads(s) for s in p.read_text().splitlines() if s.strip()];counts=Counter();missing=0
  for r in rr:
   try:value=parse_response(r['response']);counts[str(value)]+=1
   except ValueError:missing+=1
  output.append({'source':str(p.relative_to(ROOT)),'sha256':sha(p),'calls':len(rr),'prediction_counts':dict(counts),'parsing_failures':missing,'parser':'unchanged scripts.train_vera_questions.parse_response applied to saved raw responses','n_unique_scores':len(counts),'unit':'historical video-level learner/validation calls, not frame-aligned evaluation','frame_auroc':'not available; never pool with frame evaluation','keys_first_record':list(rr[0])})
 write(OUT/'step0/stage2_score_availability.json',output);print(json.dumps(output,indent=2))
if __name__=='__main__':main()
