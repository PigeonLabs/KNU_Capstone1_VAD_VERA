"""Stage7 deterministic normal-reference inputs; no inference-time label access."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,digest,write,sanitized,SCENES,segments,DATA
OUT=ROOT/'experiments/stage7'
CONDITIONS=['C0','N','X']
ALL_CONDITIONS=['P1',*CONDITIONS]
ROTATION=dict(zip(SCENES,['R02','R03','R04','R01']))
NEUTRAL={'R01':'a green belt-like surface, its metal frame, nearby tools, and small objects','R02':'a white platform, crossed supports, and a dial device','R03':'a yellow-and-black forklift-like object, its front support, and white cylindrical objects','R04':'a blade-like piece, brown sheet material, and green and metal components'}
REFERENCE_PREFIX='''REFERENCE EXAMPLES FROM NORMAL TRAINING RECORDINGS:
Ref1: <image>
Ref2: <image>
Ref3: <image>
Ref4: <image>
These four references are independent still examples from separate normal training recordings, not a temporal sequence or an exhaustive operating specification.
Compare only directly observable corresponding object states and relationships. Do not infer required positions, mandatory immobility, motion, order, or timing from these still references. A different phase or appearance alone does not establish an anomaly. If no reference is meaningfully comparable, assess the query evidence alone. Judge only the QUERY images, never the reference images.

QUERY:
'''

def build_prompts(old):
 prompts={}
 for scene in SCENES:
  p=old[scene]['P1'];start='Scene elements mentioned in normal training observations: ';a=p.index(start)+len(start);b=p.index('. This list identifies the scene;',a)
  c0=p[:a]+NEUTRAL[scene]+p[b:]
  n=REFERENCE_PREFIX+c0.replace('Assess only these eight supplied images in their supplied time order.','Assess only the eight QUERY images (Frame1 through Frame8) in their supplied time order.',1).replace('Refer only to supplied images as Frame1 through Frame8.','Refer to query images as Frame1 through Frame8 and to reference examples as Ref1 through Ref4.',1)
  c0=decision_first(c0);n=decision_first(n)
  prompts[scene]={'C0':c0,'N':n,'X':n}
 return prompts

def select_references(rows):
 references={};preflight=[]
 for scene in SCENES:
  normal=sorted([r for r in rows if r['scene']==scene and r['split']=='train' and r['video_label']==0],key=lambda r:r['id']);assert len(normal)>=4
  references[scene]=[]
  for i in range(4):
   vi=(2*i+1)*len(normal)//8;r=normal[vi];fi=(r['length']-1)*(2*i+1)//8
   references[scene].append({'reference':f'Ref{i+1}','video_id':r['id'],'frame_id':fi,'frame_sha256':r['frames_sha256'][fi],'relative_path':r['relative_path'],'selection_video_index':vi,'normal_train_count':len(normal)})
  chosen={r['video_id'] for r in references[scene]};assert len(chosen)==4
  preflight.append(sanitized(next(r for r in normal if r['id'] not in chosen)))
 return {'references':references,'X_rotation':ROTATION},preflight

def input_identity(condition,row,seg,reference_manifest):
 assert condition in CONDITIONS
 refs=[] if condition=='C0' else reference_manifest['references'][row['scene'] if condition=='N' else ROTATION[row['scene']]]
 images=[{'role':'reference','name':r['reference'],'video_id':r['video_id'],'relative_path':r['relative_path'],'frame_id':r['frame_id'],'sha256':r['frame_sha256']} for r in refs]
 images += [{'role':'query','name':f'Frame{i+1}','video_id':row['id'],'relative_path':row['relative_path'],'frame_id':f,'sha256':row['frames_sha256'][f]} for i,f in enumerate(seg['frame_ids'])]
 assert len(images)==(8 if condition=='C0' else 12)
 assert row['id'] not in {r['video_id'] for r in refs}
 return images

def resolve_images(images):
 paths=[]
 for r in images:
  folder=DATA/r['relative_path'];p=folder/f"{r['frame_id']:03d}.jpg"
  if not p.exists():p=next(p for p in folder.glob('*.jpg') if int(p.stem)==r['frame_id'])
  assert sha(p)==r['sha256'],str(p);paths.append(p)
 return paths

def protected():
 value=json.loads((OUT/'protected_previous.json').read_text())
 actual={str(p.relative_to(ROOT)):sha(p) for stage in ['stage4','stage5','stage6'] for p in (ROOT/'experiments'/stage).rglob('*') if p.is_file()}
 assert actual==value['files'],'Stage4/5/6 artifacts changed'
 return len(actual)


def decision_first(prompt):
 marker='Explain the most relevant visible observations and your industrial-process assessment.'
 assert prompt.count(marker)==1
 scope='Refer to query images as Frame1 through Frame8 and to reference examples as Ref1 through Ref4.' if 'Ref1: <image>' in prompt else 'Refer only to supplied images as Frame1 through Frame8.'
 return prompt.split(marker)[0]+"Return the decision before any explanation. The first nonempty line must be exactly Output: 0 or Output: 1. Then write Evidence: followed by at most 40 words describing visible decision evidence and relevant uncertainty. Do not add an Observations or Assessment preamble. Do not repeat the Output field. "+scope+"\nOutput: <0 or 1>\nEvidence: <at most 40 words>\n"
