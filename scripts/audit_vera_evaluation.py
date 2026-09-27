"""Independently recalculate held-out metrics and stage1 common-support comparison."""
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from scripts.train_vera_questions import sha,write

def calculate(rows,column):
 def metrics(a):
  y=[int(x['label']) for x in a];p=[float(x[column]) for x in a]
  return {'frames':len(a),'auroc':100*roc_auc_score(y,p),'auprc':100*average_precision_score(y,p)}
 scenes={s:metrics([r for r in rows if r['scene']==s]) for s in ['R01','R02','R03','R04']}
 return {'scenes':scenes,'pooled':metrics(rows),'macro':{k:float(np.mean([r[k] for r in scenes.values()])) for k in ['auroc','auprc']}}

def main():
 out=ROOT/'experiments/stage3';assert json.loads((out/'status.json').read_text())['status']=='complete'
 frozen=json.loads((out/'frozen.json').read_text())
 for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h
 assert sha(ROOT/'experiments/stage2_3/questions.txt')==frozen['questions_sha256']
 manifest=json.loads((out/'inference_manifest.json').read_text());rows=list(csv.DictReader((out/'frame_scores.csv').open()));saved=json.loads((out/'metrics.json').read_text())
 expected={(r['scene'],r['original_split'],r['video'],i) for r in manifest for i in range(r['length'])}
 keys=[(r['scene'],r['original_split'],r['video'],int(r['frame'])) for r in rows]
 assert len(keys)==len(set(keys)) and set(keys)==expected
 assert all(np.isfinite(float(r[k])) for r in rows for k in ['initial','retrieved','smoothed','final'])
 for column in ['initial','retrieved','smoothed','final']:
  current=calculate(rows,column)
  for unit in ['macro','pooled']:
   for metric in ['auroc','auprc']:assert abs(current[unit][metric]-saved[column][unit][metric])<1e-10
  for scene in current['scenes']:
   for metric in ['auroc','auprc']:assert abs(current['scenes'][scene][metric]-saved[column]['scenes'][scene][metric])<1e-10
 oldpath=ROOT/'experiments/vera_ipad/frame_scores.csv';old=list(csv.DictReader(oldpath.open()));lookup={(r['scene'],r['video'],int(r['frame'])):r for r in old}
 common=[]
 for r in rows:
  if r['original_split']!='testing':continue
  k=(r['scene'],r['video'],int(r['frame']));prior=lookup[k];assert int(r['label'])==int(prior['label'])
  common.append({**r,'stage1':prior['final']})
 comparison={'scope':'exact common frames: original testing videos that are held out in the new split; excludes all new training/validation videos and original training videos',
  'frames':len(common),'videos':len({(r['scene'],r['video']) for r in common}),
  'stage1_public_UCF_questions':calculate(common,'stage1'),'stage3_selected_questions':calculate(common,'final'),'stage1_scores_sha256':sha(oldpath)}
 write(out/'stage1_common_support.json',comparison)
 report={'status':'passed','frames':len(rows),'videos':len(manifest),'exact_frame_coverage':True,'four_stage_metrics_independently_recomputed':True,
  'frozen_source_and_question_hashes_verified':True,'common_support_frames':len(common),'common_support_videos':comparison['videos']}
 write(out/'independent_verification.json',report);print(json.dumps({'verification':report,'comparison':comparison},indent=2))
if __name__=='__main__':main()
