"""Verify deterministic R04 observation inputs before adopting a canonical input cache."""
import ast,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import *

def observation_block(path):
 text=path.read_text();return text[text.index(' for row in generate:'):text.index(" write(out/'status.json',{'status':'running','phase':'candidate_rules'")]

def main():
 base.guard();out=ROOT/'experiments/stage4/R04/normal'
 protection={str(p.relative_to(ROOT)):sha(p) for s in ['R01','R02','R03'] for p in sorted((ROOT/'experiments/stage4'/s).rglob('*')) if p.is_file()}
 target=ROOT/'experiments/stage4/r04_protected_artifacts.json';assert not target.exists();write(target,{'files':protection,'scope':'R01-R03 Stage4 files before R04 execution; no inference or evaluation rerun permitted'})
 frozen=json.loads((out/'frozen.json').read_text())
 for name,h in frozen['source_sha256'].items():assert sha(ROOT/name)==h
 assert observation_block(ROOT/'scripts/build_vera_normal_context_v2.py')==observation_block(ROOT/'scripts/build_vera_normal_context_r04.py')
 old=ast.parse((ROOT/'scripts/build_vera_normal_context_v2.py').read_text());new=ast.parse((ROOT/'scripts/build_vera_normal_context_r04.py').read_text())
 for name in ['summarize_with_fallback','parse_visual_checks','assess']:
  assert ast.dump(next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name))==ast.dump(next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name))
 assert frozen['model_config']==vars(CONFIG) and frozen['generation']=={'do_sample':False,'num_beams':1,'max_new_tokens':1024}
 assert frozen['split_sha256']==sha(ROOT/'experiments/stage2_1/split.json')
 _,gen,audit=split_scene('R04');assert frozen['generate_ids']==[r['id'] for r in gen] and frozen['audit_ids']==[r['id'] for r in audit]
 assert json.loads((out/'input_manifest.json').read_text())=={'generation':[sanitized(r) for r in gen],'audit':[sanitized(r) for r in audit]}
 inventory=json.loads((ROOT/'experiments/vera_ipad/model_inventory.json').read_text());assert json.loads((out/'model_verification.json').read_text())['files']==inventory
 for item in inventory:base.guard();assert sha(DATA/item['path'])==item['sha256']
 expected={r['path']:r['sha256'] for r in json.loads((ROOT/'docs/publication_manifest.json').read_text())['files']}
 files=[];keys=set();windows=0;image_count=0
 for row in gen:
  base.guard();base.files_for(row,row['frame_ids']);image_count+=len(row['frame_ids']);observations=[]
  for seg in segments(row['length']):
   p=out/'observations'/row['original_split']/row['video']/f"{seg['center']:06d}.json";r=json.loads(p.read_text())
   assert r['video_id']==row['id'] and r['segment']==seg and r['request_hash']==digest({k:r[k] for k in ['prompt','video_id','segment']})
   # The unchanged source block reconstructs/validates the exact prompt on every cache hit.
   assert sha(p)==expected[str(p.relative_to(ROOT))];files.append(p);keys.add(r['request_hash']);windows+=1;observations.append({'center':seg['center'],'description':r['response']})
  p=out/'video_summaries'/row['original_split']/(row['video']+'.json');r=json.loads(p.read_text());assert sha(p)==expected[str(p.relative_to(ROOT))];files.append(p);keys.add(r['request_hash'])
  assert r['request_hash']==digest({'prompt':r['prompt'],'video_id':None,'segment':None})
  if r.get('derived_not_model_json'):
   idx=np.linspace(0,len(observations)-1,min(6,len(observations)),dtype=int).tolist()
   assert r['summary_fallback']['observations_sha256']==digest(observations)
   assert r['parsed']['facts']==[{'claim':observations[i]['description'],'center':observations[i]['center']} for i in idx]
  assert all(f['center'] in {s['center'] for s in segments(row['length'])} for f in r['parsed']['facts'])
 cache_calls=[r for r in readlines(out/'calls.jsonl') if r.get('request_hash') in keys]
 cached_failures=[r for r in readlines(out/'format_failures.jsonl') if r.get('request_hash') in keys]
 manifest={'status':'verified','mode':'deterministic_cached_intermediates','observations':windows,'video_summaries':len(gen),'input_frames_sha256_checked':image_count,'model_files_sha256_checked':len(inventory),'model_inventory_sha256':sha(ROOT/'experiments/vera_ipad/model_inventory.json'),'split_sha256':frozen['split_sha256'],'observation_summary_source_semantics_identical':True,'exact_prompt_and_sampling_cache_keys':True,'cached_artifacts':{str(p.relative_to(out)):sha(p) for p in files},'source_sha256':frozen['source_sha256'],'observation_block_sha256':digest(observation_block(ROOT/'scripts/build_vera_normal_context_r04.py')),'generation':frozen['generation'],'model_config':frozen['model_config'],'verified_at':time.time()}
 private=DATA/'runs/vera_r04_local_material';private.mkdir(exist_ok=True);archive=private/str(time.time_ns());archive.mkdir()
 shutil.move(str(out),str(archive/'normal'));out.mkdir()
 for relative in manifest['cached_artifacts']:
  p=out/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive/'normal'/relative,p)
 write(out/'input_cache_manifest.json',manifest)
 for r in cache_calls:append(out/'cached_calls.jsonl',r)
 for r in cached_failures:append(out/'cached_format_failures.jsonl',r)
 write(out/'cached_model_inventory.json',inventory)
 print(json.dumps({k:v for k,v in manifest.items() if k not in ['cached_artifacts','source_sha256','model_config']},indent=2))
if __name__=='__main__':main()
