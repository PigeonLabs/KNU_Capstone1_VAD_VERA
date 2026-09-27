import hashlib,json,time
from pathlib import Path
root=Path('/home/jeong/Desktop/IPAD/runs/vera_publication/KNU_Capstone1_VAD_VERA');source=Path('/home/jeong/Desktop/IPAD')
rows=json.loads((root/'experiments/vera_ipad/model_inventory.json').read_text());checked=[]
for r in rows:
 p=source/r['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(8*1024**2):h.update(block)
 assert h.hexdigest()==r['sha256'],r['path'];checked.append(r)
(root/'experiments/stage2_2/model_verification.json').write_text(json.dumps({'status':'passed','checked_at':time.time(),'files':checked},indent=2)+'\n')
print('Verified model files:',len(checked))
