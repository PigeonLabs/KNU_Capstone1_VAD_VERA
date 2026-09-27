"""Verify Stage 4 aggregate and illustrative examples from public text artifacts."""
import csv,hashlib,json,re
from pathlib import Path
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'experiments/stage4';s=read(out/'summary.json');assert s['status']=='complete'
 assert s['completed_evaluation_scenes']==['R01','R02','R03','R04'] and read(out/'status.json')['all_four_scenes_complete'] is True
 for file,h in s['sources_sha256'].items():assert sha(ROOT/file)==h
 protection=read(out/'r04_protected_artifacts.json')['files']
 actual={str(p.relative_to(ROOT)):sha(p) for scene in ['R01','R02','R03'] for p in (out/scene).rglob('*') if p.is_file()};assert actual==protection
 assert read(out/'R04/normal/independent_verification.json')['candidate_text_immutable']
 assert not (out/'R04/normal/failure_verification.json').exists()
 for scene in ['R01','R02','R03','R04']:
  for kind in ['normal','evaluation']:
   p=out/scene/kind/'frozen.json'
   if p.exists():
    for file,h in read(p)['source_sha256'].items():assert sha(ROOT/file)==h
 assert read(out/'R04/normal/status.json')['status']=='complete' and read(out/'R04/evaluation/status.json')['status']=='complete'
 rows=[r for scene in s['completed_evaluation_scenes'] for r in csv.DictReader((out/scene/'evaluation/frame_scores.csv').open())]
 for c,metrics in s['conditions'].items():
  part=[r for r in rows if r['condition']==c];y=[int(r['label']) for r in part];pred=[float(r['final']) for r in part]
  assert len(part)==metrics['frames']==16862
  assert len({(r['scene'],r['original_split'],r['video'],r['frame']) for r in part})==len(part)
  for k,v in [('auroc',100*roc_auc_score(y,pred)),('ap',100*average_precision_score(y,pred))]:assert abs(v-metrics['pooled'][k])<1e-10
  for k in ['auroc','ap']:assert abs(sum(s['scenes'][scene][c]['final'][k] for scene in s['completed_evaluation_scenes'])/4-metrics['macro'][k])<1e-10
  counts={k:0 for k in ['tp','fp','tn','fn']}
  for r in part:
   actual=int(r['label']);predicted=int(float(r['initial']));counts['tp' if actual and predicted else 'fn' if actual else 'fp' if predicted else 'tn']+=1
  assert all(metrics['initial_binary_pooled'][k]==v for k,v in counts.items())
 examples=read(out/'explanation_examples.json');total=0
 for scene,data in examples['scenes'].items():
  labels={(r['original_split'],r['video'],int(r['frame'])):int(r['label']) for r in rows if r['condition']=='A' and r['scene']==scene}
  for e in data['examples']:
   assert e['sampled_frame_labels']==[labels[(e['original_split'],e['video'],i)] for i in e['frame_ids']]
   assert e['scored_anomalous_frames']==sum(labels[(e['original_split'],e['video'],i)] for i in range(e['center'],e['score_end']))
   for c,a in e['answers'].items():
    p=ROOT/a['source_path'];assert sha(p)==a['source_sha256']
    original=next(r for r in map(json.loads,p.read_text().splitlines()) if r['center']==e['center']) if c=='A' else read(p)
    assert original['response']==a['response'];assert re.search(r'Output\s*:\s*'+str(a['prediction']),a['response'])
   total+=1
 for doc in [ROOT/'README.md',out/'results.md',out/'explanation_examples.md']:
  for target in re.findall(r'\]\(([^\s)]+)\)',doc.read_text()):
   if '://' not in target and not target.startswith('#'):assert (doc.parent/target.split('#')[0]).exists(),(doc,target)
 print(json.dumps({'status':'passed','matched_scenes':s['completed_evaluation_scenes'],'frames_per_condition':16862,'conditions':3,'examples_verified':total,'frozen_sources_verified':True,'summary_source_hashes_verified':True,'metrics_recomputed':True,'markdown_links_verified':True,'current_R04_terminal_state_verified':True},indent=2))
if __name__=='__main__':main()
