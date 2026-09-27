"""Select Q* solely from the fixed validation candidates after training ends."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def choose(candidates):
 return max(sorted(candidates,key=lambda c:c['step']),key=lambda c:c['correct']/c['total'])
def main():
 out=ROOT/'experiments/stage2_3';train=ROOT/'experiments/stage2_2'
 assert json.loads((train/'status.json').read_text())['status']=='complete'
 cfg=json.loads((train/'frozen.json').read_text());split=json.loads((ROOT/'experiments/stage2_1/split.json').read_text())
 assert sha(ROOT/'experiments/stage2_1/split.json')==cfg['split_sha256']
 expected={r['id']:r for r in split['records'] if r['split']=='validation'}
 rows=[];hashes={}
 for step in cfg['validation_steps']:
  p=train/'validation'/f'{step:04d}.json';record=json.loads(p.read_text());hashes[str(p.relative_to(ROOT))]=sha(p)
  assert record['step']==step and {r['video_id'] for r in record['records']}==set(expected)
  assert len(record['records'])==len(expected)
  for r in record['records']:
   assert r['target']==expected[r['video_id']]['video_label']
   assert r['frame_ids']==expected[r['video_id']]['training_frame_ids']
  correct=sum(r['target']==r['prediction'] for r in record['records'])
  assert correct==record['correct'] and record['total']==len(expected)
  assert record['accuracy']==correct/len(expected)
  confusion={name:sum(r['target']==target and r['prediction']==prediction for r in record['records']) for name,target,prediction in [('tp',1,1),('fp',0,1),('tn',0,0),('fn',1,0)]}
  rows.append({'step':step,'questions':record['questions'],'correct':correct,'total':len(expected),'accuracy':record['accuracy'],'confusion':confusion})
 selected=choose(rows);out.mkdir(parents=True,exist_ok=True)
 if (out/'status.json').exists():raise RuntimeError('Preserve existing selection')
 (out/'questions.txt').write_text(selected['questions'])
 selection={'rule':'highest validation video classification accuracy; earliest candidate on ties','selected':selected,'candidates':rows,'validation_source_sha256':hashes,
  'training_frozen_sha256':sha(train/'frozen.json'),'split_sha256':cfg['split_sha256'],'questions_sha256':sha(out/'questions.txt'),'evaluation_data_used':False}
 (out/'selection.json').write_text(json.dumps(selection,indent=2,ensure_ascii=False)+'\n')
 (out/'status.json').write_text(json.dumps({'status':'complete','selected_step':selected['step'],'validation_accuracy':selected['accuracy'],'questions_sha256':sha(out/'questions.txt'),'source_sha256':sha(Path(__file__))},indent=2)+'\n')
 print(json.dumps(selection,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
