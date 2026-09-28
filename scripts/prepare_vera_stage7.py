"""Prepare agreed decision-first experiment, not a retry of frozen Stage6."""
import json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage7_common import *
def main():
 assert not (OUT/'frozen.json').exists()
 assert json.loads((OUT/'consultation/consensus.json').read_text())['status']=='both_agree_to_execute'
 files={str(p.relative_to(ROOT)):sha(p) for stage in ['stage4','stage5','stage6'] for p in (ROOT/'experiments'/stage).rglob('*') if p.is_file()}
 write(OUT/'protected_previous.json',{'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'files':files})
 rows=json.loads((ROOT/'experiments/stage2_1/split.json').read_text())['records'];refs,pre=select_references(rows)
 assert refs==json.loads((ROOT/'experiments/stage6/references.json').read_text())
 manifest={'evaluation':[sanitized(r) for r in rows if r['split']=='evaluation' and r['scene'] in SCENES],'preflight_normal_train':pre}
 assert manifest==json.loads((ROOT/'experiments/stage6/inference_manifest.json').read_text())
 for scene in SCENES:
  for r in refs['references'][scene]:resolve_images([{'relative_path':r['relative_path'],'frame_id':r['frame_id'],'sha256':r['frame_sha256']}])
 write(OUT/'references.json',refs);write(OUT/'inference_manifest.json',manifest);write(OUT/'prompts.json',build_prompts(json.loads((ROOT/'experiments/stage5/prompts.json').read_text())))
 p=json.loads((ROOT/'experiments/stage6/protocol.json').read_text());p.update(title='Stage7 decision-first fixed normal visual reference comparison',fixed=p['fixed'].replace('greedy1024','greedy128'),protected_previous_files=protected())
 p['conditions']['C0']='Stage6 C0 semantic criteria and eight query images; response contract replaced identically across C0/N/X with decision first and Evidence <=40words'
 p['failure_policy']='First nonempty line after rstrip exactly Output: 0 or Output: 1. Scan every case-insensitive whole-word Output occurrence across full response; each occurrence must be on a complete exact Output line matching first value. Same-value repeats valid with duplicate flag; malformed/incomplete/conflicting fields invalid. Never recover invalid firstline; invalid binary stops. Evidence quality does not override explicit valid class. No repair/retry/imputation.'
 p['response_contract']='firstline Output: [01], then Evidence <=40words. Remove prior observations/assessment/final-line instructions. 128newtokens; greedy otherwise unchanged.'
 p['quality']='Save actual generated token IDs/count and EOS via read-only generate return capture. No altered generation args beyond128 budget. Missing/over40word/repeated Evidence and truncated-withoutEOS are quality diagnostics only. Do not infer semantic correctness from format or claim explanation caused decision.'
 p['limits']+=['decision-first changes predictions; old Stage5 P1 comparison descriptive only','malformed/incomplete later Output may still fail-stop even with valid firstline','60normalpreflight does not guarantee3216evaluationvalid','post-decision Evidence is not causal reasoning']
 p['fresh_inference']='All60preflight+3216evaluation calls newly executed; no Stage6 predictions reused; no furtherautomaticformatrepairsafterfreeze'
 write(OUT/'protocol.json',p);print(json.dumps({'prepared':True,'protected':protected(),'conditions':['C0','N','X']},indent=2))
if __name__=='__main__':main()
