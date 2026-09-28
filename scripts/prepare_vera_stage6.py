"""Prepare the agreed Stage6 protocol once, before inference; no label files opened."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage6_common import *

def main():
 assert not (OUT/'frozen.json').exists(),'Already frozen'
 draft=ROOT/'runs/stage6_collaboration';OUT.mkdir(parents=True,exist_ok=True)
 shutil.copy2(draft/'protected_previous.json',OUT/'protected_previous.json');protected()
 consultation=OUT/'consultation';consultation.mkdir(exist_ok=True)
 for name in ['consensus.json','round2_gpt_excerpt.md','round2_claude_excerpt.md','training_visual_inspection.json','research_checked.json']:shutil.copy2(draft/name,consultation/name)
 assert json.loads((consultation/'consensus.json').read_text())['status']=='both_agree_to_execute'
 rows=json.loads((ROOT/'experiments/stage2_1/split.json').read_text())['records']
 refs,preflight=select_references(rows);prompts=build_prompts(json.loads((ROOT/'experiments/stage5/prompts.json').read_text()))
 assert prompts==json.loads((draft/'draft_prompts.json').read_text())
 evaluation=[sanitized(r) for r in rows if r['split']=='evaluation' and r['scene'] in SCENES]
 assert {r['id'] for r in preflight}.isdisjoint(r['id'] for r in evaluation)
 for scene in SCENES:
  for r in refs['references'][scene]:resolve_images([{'relative_path':r['relative_path'],'frame_id':r['frame_id'],'sha256':r['frame_sha256']}])
 write(OUT/'references.json',refs);write(OUT/'prompts.json',prompts);write(OUT/'inference_manifest.json',{'evaluation':evaluation,'preflight_normal_train':preflight})
 protocol={
  'title':'Stage6 corrected scene wording and fixed normal visual references','approval':'User requested follow-up execution after genuine GPT6Pro/ClaudeOpus5.5High consensus; both explicitly AGREE TO EXECUTE',
  'conditions':{'C0':'Stage5 P1 with only neutral object phrase replaced from local normal TRAIN observations','N':'C0 plus four same-scene independent normal still references','X':'byte-identical N text, four normal references from next scene R01-R02-R03-R04-R01'},
  'reference_selection':'0-based sorted normal current-TRAIN rows; video floor((i+.5)*n/4), frame floor((F-1)*(i+.5)/4), i=0..3; no selection by output, query, validation or evaluation',
  'reference_interpretation':'independent still examples, not a temporal sequence or full operating specification; no inferred mandatory positions/motion/timing; noncomparable references do not decide query normality',
  'preflight':'first sorted normal current-TRAIN video outside reference videos per scene; five uniformly spaced existing centers; 4x5x3=60calls; any invalid stops before evaluation, no repair/retry',
  'evaluation':'same37videos/16862frames/1072windows per condition; 3216newcalls; Stage5 P1 cached baseline only',
  'input':'C0 query8; N/X ref4 then identical query8; num_patches_list=[1]*N; same RGB448 transform, one image patch each',
  'fixed':'InternVL2-8B BF16 eager; seed0 greedy1024; original stride16/clipped300frame/8sampling; existing query-only ImageBind cache and retrieval->smoothing->position',
  'label_policy':'all evaluation inference validated before opening any evaluation frame-label file; split video-level metadata used solely for training reference/preflight membership; labels absent from prompts/inference manifests',
  'failure_policy':'raw response retained; original Stage5 unique binary Output parser; no imputation, semantic repair, retry, reference replacement, changing precision/model, or subset presented as full metrics; incomplete/invalid stops full evaluation',
  'primary':'N-C0 scene macro initial AUROC (binary balanced accuracy)','secondary':'N-X initial AUROC; C0-P1 descriptive wording comparison',
  'metrics':'scene/macro/pooled initial/retrieved/smoothed/final AUROC/AP; initial confusion recall FPR; input-any vs target-any; ZERO/ONE controls; report every condition',
  'primary_uncertainty':'reuse exact Stage5 paired scene-video bootstrap2000draws seed0; undefined retained; exploratory only',
  'secondary_uncertainty':'separate scene and video any-anomaly-label stratified paired bootstrap2000draws seed0, preserving each stratum video count; no replacement of primary analysis',
  'permutation_diagnostic':'200 within-video permutations of initial segment predictions, seed1; same permutation across methods for each video/draw; same legacy postprocessing; distributions only, no confirmatory pvalue; no model calls',
  'selection':'none; no abnormal training or validation-based choice; no tuning after evaluation',
  'limits':['N-C0 adds image count/instruction/normal content jointly','X semantic distraction means N-X alone cannot establish gain over C0','C0 uses limited Codex training-only inspection, not human-verified process rules','references cover few states, not complete normal dynamics','long-window score alignment remains','already-observed test set; exploratory, not confirmation','normal current TRAIN can include original testing directory videos after approved disjoint re-split'],
  'protected_previous_files':protected(),'model_updates':0}
 write(OUT/'protocol.json',protocol)
 print(json.dumps({'prepared':True,'references':{s:[(r['video_id'],r['frame_id']) for r in rs] for s,rs in refs['references'].items()},'preflight':[r['id'] for r in preflight],'protected':protected()},indent=2))
if __name__=='__main__':main()
