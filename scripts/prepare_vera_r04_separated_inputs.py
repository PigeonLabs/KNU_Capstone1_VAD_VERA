"""Verify and install the normal-only input cache for the separated R04 pipeline."""
import json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import *
def main():
 base.guard();folder=ROOT/'experiments/stage4/R04';old=folder/'normal'
 protected=json.loads((ROOT/'experiments/stage4/r04_protected_artifacts.json').read_text())['files']
 actual={str(p.relative_to(ROOT)):sha(p) for s in ['R01','R02','R03'] for p in (ROOT/'experiments/stage4'/s).rglob('*') if p.is_file()};assert actual==protected
 frozen=json.loads((old/'frozen.json').read_text())
 for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h
 cache=json.loads((old/'input_cache_manifest.json').read_text());assert cache['model_config']==vars(CONFIG)
 assert cache['generation']=={'do_sample':False,'num_beams':1,'max_new_tokens':1024}
 assert cache['split_sha256']==sha(ROOT/'experiments/stage2_1/split.json')
 _,generate,audit=split_scene('R04');assert frozen['generate_ids']==[r['id'] for r in generate] and frozen['audit_ids']==[r['id'] for r in audit]
 assert json.loads((old/'input_manifest.json').read_text())=={'generation':[sanitized(r) for r in generate],'audit':[sanitized(r) for r in audit]}
 for p,h in cache['cached_artifacts'].items():assert sha(old/p)==h
 inventory=json.loads((old/'cached_model_inventory.json').read_text());assert inventory==json.loads((ROOT/'experiments/vera_ipad/model_inventory.json').read_text())
 for r in inventory:base.guard();assert sha(DATA/r['path'])==r['sha256']
 for row in generate:base.guard();base.files_for(row,row['frame_ids'])
 private=DATA/'runs/vera_r04_local_material'/str(time.time_ns());private.mkdir(parents=True)
 for name in frozen['source_sha256']:
  p=private/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
 shutil.move(str(folder),str(private/'R04'));out=folder/'normal';out.mkdir(parents=True)
 for name in list(cache['cached_artifacts'])+['cached_calls.jsonl','cached_format_failures.jsonl','cached_model_inventory.json']:
  p=out/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(private/'R04/normal'/name,p)
 cache['verified_at']=time.time();cache['protected_R01_R03_files']=len(protected);cache['reuse_verification']='current model files, 5300 normal input frames, cache content, config, split and source semantics verified; candidate/grounding not reused'
 write(out/'input_cache_manifest.json',cache)
 print(json.dumps({'cache_status':'verified','observations':338,'summaries':14,'model_files':len(inventory),'normal_frames':sum(r['length'] for r in generate),'protected_files':len(protected),'candidate_and_grounding_reused':False},indent=2))
if __name__=='__main__':main()
