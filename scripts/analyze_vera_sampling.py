"""Describe weak video labels vs uniform training frames without changing training."""
import json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 data=Path('/home/jeong/Desktop/IPAD');manifest=json.loads((ROOT/'experiments/stage2_1/split.json').read_text());records=[]
 for row in manifest['records']:
  if row['split'] not in ['train','validation']:continue
  labels=np.load(data/row['label_path'],allow_pickle=False).reshape(-1) if row['label_path'] else np.zeros(row['length'],dtype=int)
  records.append({'video_id':row['id'],'split':row['split'],'video_label':row['video_label'],'sampled_frame_ids':row['training_frame_ids'],'sampled_anomalous_frames':int(labels[row['training_frame_ids']].sum()),'sampled_frames':8})
 summary={}
 for split in ['train','validation']:
  rows=[r for r in records if r['split']==split and r['video_label']==1]
  summary[split]={'abnormal_videos':len(rows),'abnormal_videos_with_no_anomaly_in_sampled_frames':sum(r['sampled_anomalous_frames']==0 for r in rows)}
 (ROOT/'experiments/stage2_2/sampling_label_diagnostic.json').write_text(json.dumps({'purpose':'Descriptive audit only; frame labels are not passed to learner/optimizer and do not alter sampling or selection','checked_at':time.time(),'summary':summary,'records':records},indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
