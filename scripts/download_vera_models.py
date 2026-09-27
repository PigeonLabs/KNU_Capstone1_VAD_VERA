import os,shutil,time,urllib.request,json,hashlib
from pathlib import Path
os.environ['HF_HOME']=str(Path('cache/vera/hf').resolve())
from huggingface_hub import snapshot_download
root=Path('cache/vera'); out=Path('experiments/vera_ipad');
if shutil.disk_usage('.').free < 40*1024**3: raise RuntimeError('Insufficient disk for model download and reserve')
p=snapshot_download('OpenGVLab/InternVL2-8B',revision='6fb9ad6924f69424e57fab2ab061d707688f0296',local_dir=str(root/'InternVL2-8B'),allow_patterns=['*.py','*.json','*.safetensors','*.model','*.txt'],max_workers=2)
print('InternVL downloaded',p,flush=True)
target=root/'imagebind_huge.pth'
if not target.exists():
 with urllib.request.urlopen('https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth',timeout=60) as r, target.with_suffix('.part').open('wb') as f:
  while block:=r.read(8*1024**2):
   if shutil.disk_usage('.').free<=10*1024**3:
    (out/'disk_pause.json').write_text('{"status":"paused_low_disk","automatic_resume":false}')
    raise RuntimeError('disk reserve reached')
   f.write(block)
 target.with_suffix('.part').replace(target)
print('ImageBind downloaded',flush=True)
manifest=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and ('InternVL2-8B' in p.parts or p.name=='imagebind_huge.pth') and '.cache' not in p.parts:
  h=hashlib.sha256()
  with p.open('rb') as f:
   while b:=f.read(8*1024**2): h.update(b)
  manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(out/'model_inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Inventory complete',flush=True)
