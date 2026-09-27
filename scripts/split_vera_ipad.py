"""Create an immutable video-level IPAD train/validation/evaluation manifest."""
import collections,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/jeong/Desktop/IPAD')
OUT=ROOT/'experiments/stage2_1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 if (OUT/'status.json').exists():raise RuntimeError('Preserve completed split; refusing overwrite')
 rng=np.random.default_rng(0);records=[];source={}
 for oldsplit,filename in [('training','normal_manifest'),('testing','test_manifest')]:
  path=ROOT/f'experiments/vera_ipad/{filename}.json';source[str(path.relative_to(ROOT))]=sha(path)
  for row in json.loads(path.read_text()):
   if oldsplit=='testing' and row['scene']=='R02' and int(row['video']) in [12,13,14]:continue
   files=sorted((DATA/row['relative_path']).glob('*.jpg'),key=lambda p:int(p.stem))
   assert len(files)==row['length']
   assert [int(p.stem) for p in files]==list(range(len(files)))
   assert [sha(p) for p in files]==row['frames_sha256'],'Input image hash changed'
   label_path=DATA/'IPAD_dataset'/row['scene']/'test_label'/f"{int(row['video']):03d}.npy"
   labels=np.zeros(len(files),dtype=int) if oldsplit=='training' else np.load(label_path,allow_pickle=False).reshape(-1)
   assert len(labels)==len(files) and np.isin(labels,[0,1]).all()
   records.append(dict(id=f"{row['scene']}/{oldsplit}/{row['video']}",scene=row['scene'],original_split=oldsplit,video=row['video'],
     length=len(files),relative_path=row['relative_path'],video_label=int(labels.max()),
     label_path=str(label_path.relative_to(DATA)) if oldsplit=='testing' else None,
     label_sha256=sha(label_path) if oldsplit=='testing' else None,
     frame_ids=list(range(len(files))),frames_sha256=row['frames_sha256'],
     video_content_sha256=hashlib.sha256(''.join(row['frames_sha256']).encode()).hexdigest(),
     training_frame_ids=[i*(len(files)//8) for i in range(8)]))
 # Complete videos are indivisible. Identical full videos cannot cross subsets.
 assert len({r['video_content_sha256'] for r in records})==len(records),'Duplicate full videos require grouping before splitting'
 for scene in ['R01','R02','R03','R04']:
  for label in [0,1]:
   group=[r for r in records if r['scene']==scene and r['video_label']==label]
   assert len(group)>=3
   order=rng.permutation(len(group));n_test=max(1,math.ceil(.2*len(group)));n_val=max(1,math.ceil(.1*(len(group)-n_test)))
   for j,index in enumerate(order):group[index]['split']='evaluation' if j<n_test else 'validation' if j<n_test+n_val else 'train'
 counts={split:{scene:{str(label):sum(r['split']==split and r['scene']==scene and r['video_label']==label for r in records) for label in [0,1]} for scene in ['R01','R02','R03','R04']} for split in ['train','validation','evaluation']}
 # Audit byte-identical frames across splits without changing split based on outcomes.
 fingerprints={s:set(h for r in records if r['split']==s for h in r['frames_sha256']) for s in counts}
 overlap={f'{a}:{b}':len(fingerprints[a]&fingerprints[b]) for a,b in [('train','validation'),('train','evaluation'),('validation','evaluation')]}
 if any(overlap.values()):raise RuntimeError(f'Cross-split identical frames require grouping: {overlap}')
 manifest={'seed':0,'method':'stratify scene/video_label, ceil(20%) evaluation, ceil(10% remainder) validation, remaining train; numpy default_rng(0)',
  'unit':'whole video','original_data_modified':False,'frame_labels_used_for_split':'only max(label) to obtain weak video label; no temporal annotations supplied to learner/optimizer',
  'excluded':['R02/testing/12','R02/testing/13','R02/testing/14'],'source_sha256':source,'records':records}
 write(OUT/'split.json',manifest)
 write(OUT/'summary.json',{'counts':counts,'total_videos':len(records),'totals':{s:sum(r['split']==s for r in records) for s in counts},'identical_frame_overlap':overlap,
  'heldout_limitation':'Original testing data were previously evaluated in stage1; this is an exploratory re-split evaluation, not an untouched standard IPAD test benchmark.',
  'session_group_limitation':'No recording-session identifiers available; separation and exact-duplicate audit are at video/file-content level.'})
 write(OUT/'status.json',{'status':'complete','split_sha256':sha(OUT/'split.json'),'script_sha256':sha(__file__)})
 print(json.dumps(json.loads((OUT/'summary.json').read_text()),indent=2))
if __name__=='__main__':main()
