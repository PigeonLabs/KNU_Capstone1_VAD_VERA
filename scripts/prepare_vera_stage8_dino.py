"""Fetch the same pinned public DINOv2 backbone; no dataset access or inference."""
import sys,io,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage8_common import OUT,sha,write
import scripts.train_vera_questions as base
COMMIT='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
def main():
 base.OUT=OUT;base.guard();cache=ROOT/'cache/stage8/dino';cache.mkdir(parents=True,exist_ok=True)
 target=cache/COMMIT;url=f'https://codeload.github.com/facebookresearch/dinov2/zip/{COMMIT}'
 if not (target/'hubconf.py').exists():
  data=urllib.request.urlopen(url).read()
  with zipfile.ZipFile(io.BytesIO(data)) as z:
   for info in z.infolist():
    rel=Path(*Path(info.filename).parts[1:])
    if not rel.parts or '..' in rel.parts:continue
    p=target/rel
    if info.is_dir():p.mkdir(parents=True,exist_ok=True)
    else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(info))
  (cache/'source.zip').write_bytes(data)
 weights=cache/'dinov2_vitb14_pretrain.pth'
 if not weights.exists():
  with urllib.request.urlopen('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth') as response,weights.with_suffix('.tmp').open('wb') as f:
   while chunk:=response.read(8*1024**2):base.guard();f.write(chunk)
  weights.with_suffix('.tmp').replace(weights)
 write(OUT/'dino_download.json',{'source_url':url,'commit':COMMIT,'weights_sha256':sha(weights),'source_files':{str(p.relative_to(target)):sha(p) for p in target.rglob('*.py')},'local_only':True,'dataset_accessed':False})
 print('Pinned DINOv2 source and weights downloaded; no inference.')
if __name__=='__main__':main()
