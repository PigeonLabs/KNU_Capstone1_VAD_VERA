"""Construct three preregistered prompts using normal-training text only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import write,sha,digest,SCENES
OUT=ROOT/'experiments/stage5'
NEUTRAL={
 'R01':'a tabletop, a red power tool, and a green and black device',
 'R02':'a tabletop, a digital device, a green clamp, and a metal bar',
 'R03':'a tabletop, a yellow forklift-like object, a black and white support, and white cylindrical objects',
 'R04':'a tabletop, a blade, a clamp, a metal bar, and brown material',
}
COMMON='''Assess only these eight supplied images in their supplied time order. They come from a clipped context window of up to 10 seconds at an assumed 30 FPS. Some images may be repeated in very short videos.
Scene: {scene}.
Scene elements mentioned in normal training observations: {objects}. This list identifies the scene; it does not certify normality or require every listed object in every image.
{reference}
Question: Does anything visible in these images depart from a coherent industrial work process, considering object state, relative position, tool-workpiece relationships, and changes across the supplied images?
Do not invent defects, hidden functions, sounds, mandatory steps, timing limits, or events between sampled images. Do not infer a stopped process just because sampled appearances are unchanged. Mere motion, object presence, unfamiliar appearance, lighting change or occlusion alone is not proof of a process deviation. A familiar scene or an ordinary-looking action alone is not proof of normality.
Decision rule for every condition: Use label 1 for at least one specific, visibly supported process discrepancy not explained by a supported normal state or variation. It need not occupy most images or look dangerous. Use label 0 when no discrepancy is supported by the visible evidence. Missing visibility is evidence for neither conclusion; label 0 does not certify the entire video. Do not invent evidence for either answer. Do not give confidence scores.
{assessment}
Refer only to supplied images as Frame1 through Frame8. Keep the response within 120 words. The final line must be exactly Output: 0 or Output: 1, once only.
{format}
'''
REFERENCE='''Partial description from normal training observations (verbatim frozen Stage4 description):
{description}
The description is a record of sampled normal examples, not an exhaustive operating specification. Its statements are not observations of the current input. Matching its objects or background is insufficient to establish normality. A detail absent from it is not automatically abnormal. Its stationary observations do not establish mandatory immobility in other phases. Assess applicability using independently visible process context, not by requiring the expected normal state itself to be present before checking it. Do not invent missing normal dynamics or process order.
'''
DIRECT='Explain the most relevant visible observations and your industrial-process assessment.'
BALANCED='''Compare NORMAL (a supported normal state or variation) and DEVIATION (a specific visibly supported discrepancy). State the strongest evidence for each, using none observed where appropriate. Do not manufacture evidence to fill a field. A normal-looking background does not cancel a concrete discrepancy elsewhere. Apply the common decision rule to this evidence comparison.'''
DIRECT_FORMAT='''Observations: <visible facts and changes, with Frame references; at most 60 words>
Assessment: <decision evidence and relevant uncertainty; at most 60 words>
Output: <0 or 1>'''
BALANCED_FORMAT='''Observed changes: <visible facts with Frame references; at most 40 words>
Evidence consistent: <support for normality, or none observed; at most 30 words>
Evidence conflicting: <support for a process discrepancy, or none observed; at most 30 words>
Uncertainty: <visibility or sampling limitations; at most 20 words>
Output: <0 or 1>'''

def main():
 assert not (OUT/'frozen.json').exists(),'Cannot regenerate prompts after freeze'
 examples=json.loads((OUT/'consultation/training_context_examples.json').read_text());split=json.loads((ROOT/'experiments/stage2_1/split.json').read_text())['records'];allowed={r['id'] for r in split if r['split']=='train' and r['video_label']==0}
 for scene,records in examples.items():
  for r in records:
   p=ROOT/r['source'];assert sha(p)==r['sha256'];assert scene+'/'+p.parent.name+'/'+p.stem in allowed
 prefix=''.join(f'Frame{i}: <image>\n' for i in range(1,9))+'\n'
 prompts={};references={}
 for scene in SCENES:
  p=ROOT/f'experiments/stage4/{scene}/normal/normal_description.txt';description=p.read_text();references[scene]={'source':str(p.relative_to(ROOT)),'sha256':sha(p),'neutral_objects':NEUTRAL[scene],'normal_training_examples':examples[scene]}
  prompts[scene]={}
  for c in ['P1','P2','P3']:
   prompts[scene][c]=prefix+COMMON.format(scene=scene,objects=NEUTRAL[scene],reference='' if c=='P1' else REFERENCE.format(description=description),assessment=BALANCED if c=='P3' else DIRECT,format=BALANCED_FORMAT if c=='P3' else DIRECT_FORMAT)
 write(OUT/'prompts.json',prompts);write(OUT/'references.json',references)
 protocol={'title':'Stage5 industrial-process prompting and balanced visible evidence','approval':'User requested collaboration with GPT 6 Pro and Claude Opus 5.5 High, prompting-only Stage5 and publication on completion','scenes':SCENES,'conditions':{'P1':'scene-specific neutral objects + industrial question + direct assessment','P2':'P1 plus verbatim Stage4 normal description and non-exhaustive/non-tautological interpretation','P3':'P2 plus structured balanced evidence assessment; identical binary decision rule'},'primary_comparison':'P3 minus P2, scene macro initial AUROC (binary balanced accuracy)','secondary_comparisons':['P2-P1','P3-A'],'report_all':True,'prompt_selection':'none; all three fixed in advance; no validation/evaluation-based rewrite','normal_reference_policy':'no new normative process rule generation; only normal training excerpts and frozen normal descriptions; no abnormal training used','preflight':'first sorted normal audit-training video per scene, five uniformly spaced existing windows; parsing/runtime only, no prompt repair/retry if constant predictions','evaluation':'37 frozen evaluation videos, 16862 frames, 1072 windows per new condition; labels opened only after all new inference is complete','preserved_pipeline':'InternVL2-8B BF16 eager; deterministic 1024 tokens; seed0; original stride16, clipped300frame windows, uniform8frame sampling, ImageBind retrieval then smoothing then position weight unchanged','failure_policy':'record every raw response; unique explicit Output binary parsed with existing Stage4 parser; no imputation, semantic repair, inference retry or tuning; if any missing/invalid prediction full evaluation metrics withheld','uncertainty':'paired video bootstrap within each scene; same resamples for all conditions,2000 draws,seed0; all frames of resampled videos retained; undefined classless resamples recorded; exploratory CIs, no confirmatory p-value','metrics':['initial/retrieved/smoothed/final AUROC and AP by scene/macro/pooled','initial confusion recall FPR positive-response rate','input-any versus scored-segment-any labels and 0,1-4,5-8 sampled-positive strata','all-zero/all-one controls through identical VERA processing'],'protected_stage4_files':6837,'interpretation_limits':['already-observed evaluation; exploratory not untouched test','P1 versus Stage4 changes both task framing and format','P2-P1 adds description, interpretation and length; not pure knowledge effect','P3-P2 changes evidence structure and response allocation; not perfectly isolated token-length effect','scene names/object labels are model-derived and can be inaccurate','normal references do not establish complete process dynamics','binary initial AUROC equals balanced accuracy at one operating point','video bootstrap cannot eliminate residual same-session correlation'],'consultants':{'requested':['GPT 6 Pro','Claude Opus 5.5 High'],'confirmed_ui':['ChatGPT model menu 6 Pro','Claude Model: Opus 5.5 High'],'urls':['https://chatgpt.com/c/6ab9e50c-86e8-83e8-a617-4b42df385966','https://claude.ai/chat/7ce4b602-fe43-4e89-becb-7d6de1d0b7d6'],'role':'design advisors only; local InternVL performs all predictions; summarized advice and adopted decisions recorded separately'}}
 write(OUT/'protocol.json',protocol)
 print(json.dumps({'status':'prepared','scenes':SCENES,'prompts':12,'prompt_sha256':{s:{c:digest(p) for c,p in d.items()} for s,d in prompts.items()}},indent=2))
if __name__=='__main__':main()
