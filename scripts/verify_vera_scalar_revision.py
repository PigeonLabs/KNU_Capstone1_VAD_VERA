"""Migration audit for an explicit scalar Output parser; preserve prior frozen run."""
import importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_vera_questions import parse_response,readlines,write,sha

def main():
 out=ROOT/'experiments/stage2_2';archive=out/'revisions/002_scalar_output_field'
 spec=importlib.util.spec_from_file_location('previous_training',archive/'train_vera_questions.py');previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
 history=readlines(out/'iterations.jsonl');learners=readlines(out/'learner_responses.jsonl');val=readlines(out/'validation_responses.jsonl')
 assert len(history)==217
 latest={(r['iteration'],r['video_id']):r for r in learners}
 checked=0
 for step in history:
  for video,pred in zip(step['batch_ids'],step['predictions']):
   r=latest[(step['iteration'],video)];assert previous.parse_response(r['response'])==parse_response(r['response'])==pred;checked+=1
 for r in val:assert previous.parse_response(r['response'])==parse_response(r['response']);checked+=1
 failed=learners[-1];assert failed['iteration']==218
 try:previous.parse_response(failed['response'])
 except ValueError:pass
 else:raise AssertionError('Expected explanation-format failure')
 assert parse_response(failed['response'])==0
 report={'status':'passed','revision':'read unique explicit binary scalar Output field; allow explanation; reject missing/nonbinary/ambiguous values',
 'previous_successful_predictions_unchanged':checked,'completed_iterations':217,'failed_iteration':218,'new_explicit_prediction':0,'time':time.time(),'source_sha256':sha(ROOT/'scripts/train_vera_questions.py'),
 'model_questions_split_generation_budget_changed':False,'resume_policy':'reexecute incomplete batch; retain every raw failed attempt'}
 write(archive/'verification.json',report);print(json.dumps(report,indent=2))
 (out/'frozen.json').rename(archive/'previous_active_frozen.json')
if __name__=='__main__':main()
