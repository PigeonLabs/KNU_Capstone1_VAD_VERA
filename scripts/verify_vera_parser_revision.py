"""Verify the formatting-only learner parser revision against all recorded responses."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_vera_questions import parse_response,readlines,write,sha
from ipad.vera import parse_response as previous_parser

def main():
 out=ROOT/'experiments/stage2_2';history=readlines(out/'iterations.jsonl');learners=readlines(out/'learner_responses.jsonl')
 assert len(history)==185 and len(learners)==371
 checked=0
 for i,r in enumerate(history):
  for raw,prediction in zip(learners[2*i:2*i+2],r['predictions']):
   assert previous_parser(raw['response'])==parse_response(raw['response'])==prediction;checked+=1
 val=readlines(out/'validation_responses.jsonl')
 for r in val:assert previous_parser(r['response'])==parse_response(r['response']);checked+=1
 failed=learners[-1];assert failed['iteration']==186
 try:previous_parser(failed['response'])
 except ValueError:pass
 else:raise AssertionError('Expected previous formatting-only failure')
 assert parse_response(failed['response'])==0
 report={'status':'passed','revision':'accept inline terminal explicit Output: 0/1; reject missing, ambiguous or nonbinary output','previous_successful_responses_unchanged':checked,
  'previous_completed_iterations':185,'failed_iteration':186,'failed_response_revised_prediction':0,'source_sha256':sha(ROOT/'scripts/train_vera_questions.py'),'time':time.time(),
  'model_questions_split_generation_budget_changed':False,'resume_policy':'reexecute incomplete iteration 186, retain original failed raw response as an uncommitted attempt'}
 write(out/'revisions/001_inline_output_fix/verification.json',report);print(json.dumps(report,indent=2))
 # Preserve the prior frozen configuration in revisions; main() creates a new
 # source fingerprint for the resumed code, with an identical data/config schedule.
 (out/'frozen.json').rename(out/'revisions/001_inline_output_fix/previous_active_frozen.json')
if __name__=='__main__':main()
