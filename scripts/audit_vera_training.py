"""Replay question updates and audit train/validation separation after execution."""
import collections,hashlib,json,sys,csv
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_vera_questions import parse_questions,parse_response,readlines,sha,write

def main():
 out=ROOT/'experiments/stage2_2';cfg=json.loads((out/'frozen.json').read_text());status=json.loads((out/'status.json').read_text())
 assert status['status']=='complete' and status['completed_iterations']==600
 for name,digest in cfg['source_sha256'].items():assert sha(ROOT/name)==digest,name
 split=json.loads((ROOT/'experiments/stage2_1/split.json').read_text());assert sha(ROOT/'experiments/stage2_1/split.json')==cfg['split_sha256']
 train={r['id']:r for r in split['records'] if r['split']=='train'};val={r['id']:r for r in split['records'] if r['split']=='validation'};test={r['id'] for r in split['records'] if r['split']=='evaluation'}
 iterations=readlines(out/'iterations.jsonl');raw=readlines(out/'optimizer_responses.jsonl');learners=readlines(out/'learner_responses.jsonl')
 assert len(iterations)==len(raw)==600 and len(learners)>=1200
 learner_attempts=learners
 latest={(r['iteration'],r['video_id']):r for r in learner_attempts}
 learners=[latest[(r['iteration'],v)] for r in iterations for v in r['batch_ids']]
 assert len(learners)==1200
 questions=cfg['initial_questions'];changed=0;errors=collections.Counter()
 for i,(r,opt) in enumerate(zip(iterations,raw)):
  assert r['iteration']==opt['iteration']==i+1 and r['questions_before']==questions
  assert r['batch_ids']==cfg['schedule'][i]['batch_ids']==opt['batch_ids']
  assert not(set(r['batch_ids'])&test)
  lr=learners[2*i:2*i+2]
  assert [x['video_id'] for x in lr]==r['batch_ids']
  assert [parse_response(x['response']) for x in lr]==r['predictions']
  assert [train[x['video_id']]['video_label'] for x in lr]==r['targets']
  assert r['train_correct']==sum(pred==target for pred,target in zip(r['predictions'],r['targets']))
  assert all(x['frame_ids']==train[x['video_id']]['training_frame_ids'] for x in lr)
  error=None
  try:q=parse_questions(opt['response'])
  except ValueError as exc:q=questions;error=str(exc);errors[error]+=1
  assert error==r['optimizer_parse_error'] and q==r['questions_after']
  changed+=q!=questions;questions=q
 validation=[]
 for step in cfg['validation_steps']:
  r=json.loads((out/'validation'/f'{step:04d}.json').read_text())
  assert r['questions']==(cfg['initial_questions'] if step==0 else iterations[step-1]['questions_after'])
  assert {x['video_id'] for x in r['records']}==set(val)
  assert all(x['prediction']==parse_response(x['response']) and x['target']==val[x['video_id']]['video_label'] for x in r['records'])
  assert sum(x['prediction']==x['target'] for x in r['records'])==r['correct']
  v={k:r[k] for k in ['step','correct','total','accuracy']}
  v.update({name:sum(x['target']==target and x['prediction']==prediction for x in r['records']) for name,target,prediction in [('tp',1,1),('fp',0,1),('tn',0,0),('fn',1,0)]})
  validation.append(v)
 epochs=[]
 for epoch in range(1,11):
  rs=[r for r in iterations if r['epoch']==epoch]
  seen=[v for r in rs for v in r['batch_ids']];assert len(seen)==len(set(seen))==120 and set(seen)==set(train)
  epochs.append({'epoch':epoch,'iterations':len(rs),'videos':len(seen),'online_correct':sum(r['train_correct'] for r in rs),'online_accuracy':sum(r['train_correct'] for r in rs)/len(seen),
    'invalid_optimizer_outputs':sum(r['optimizer_parse_error'] is not None for r in rs)})
 summary={'status':'passed','iterations':600,'epochs':epochs,'train_videos':120,'validation_videos':17,'learner_predictions':1200,'learner_attempts':len(learner_attempts),'uncommitted_learner_attempts':len(learner_attempts)-1200,'validation_predictions':119,
  'questions_changed':changed,'optimizer_parse_failures':dict(errors),'invalid_optimizer_total':sum(errors.values()),'valid_optimizer_updates':600-sum(errors.values()),
  'validation':validation,'total_model_seconds':sum(r['learner_seconds']+r['optimizer_seconds'] for r in iterations),
  'test_videos_used_in_training_or_selection':0,'parameter_updates':0,'source_hashes_verified':True,
  'note':'Training accuracy describes changing questions online; it is not fixed-model evaluation. Invalid optimizer output retains previous questions.'}
 summary['runtime_scope']='model.chat only, after image decoding/preprocessing/tensor transfer; excludes model loading, hashing and log I/O. Reserved VRAM includes shared-process allocator cache from previous calls.'
 summary['runtime_by_role']={}
 for name,records in [('learner',learner_attempts),('optimizer',raw),('validation',readlines(out/'validation_responses.jsonl'))]:
  summary['runtime_by_role'][name]={'calls':len(records),'sum_seconds':sum(r['seconds'] for r in records),'mean_seconds':float(np.mean([r['seconds'] for r in records])),'p95_seconds':float(np.percentile([r['seconds'] for r in records],95)),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in records),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in records)}
 catalog={};history=[]
 for r in iterations:
  before=hashlib.sha256(r['questions_before'].encode()).hexdigest();after=hashlib.sha256(r['questions_after'].encode()).hexdigest()
  for key,q in [(before,r['questions_before']),(after,r['questions_after'])]:
   if key not in catalog:catalog[key]={'questions':q,'first_observed_iteration':r['iteration']}
  instant=datetime.fromtimestamp(r['time'],timezone.utc)
  history.append({'iteration':r['iteration'],'epoch':r['epoch'],'utc':instant.isoformat(),'kst':instant.astimezone(ZoneInfo('Asia/Seoul')).isoformat(),'train_correct':r['train_correct'],'batch_size':2,'question_before_sha256':before,'question_after_sha256':after,'optimizer_parse_error':r['optimizer_parse_error'] or '', 'learner_seconds':r['learner_seconds'],'optimizer_seconds':r['optimizer_seconds']})
 with (out/'history.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
 write(out/'questions_catalog.json',catalog)
 write(out/'training_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
