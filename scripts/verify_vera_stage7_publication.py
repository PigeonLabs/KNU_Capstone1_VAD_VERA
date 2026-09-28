"""Verify completed consensus experiment and prior evidence before publication."""
import json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write,SCENES
from scripts.publish_vera_stage import publication_files
OUT=ROOT/'experiments/stage7'

def main():
 assert json.loads((OUT/'status.json').read_text())['status']=='complete'
 audit=json.loads((OUT/'independent_verification.json').read_text());analysis=json.loads((OUT/'analysis_verification.json').read_text());assert audit['status']==analysis['status']=='passed'
 assert sha(ROOT/'scripts/audit_vera_stage7.py')==audit['audit_source_sha256']
 for p,h in analysis['source_sha256'].items():assert sha(ROOT/p)==h
 frozen=json.loads((OUT/'frozen.json').read_text())
 for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h
 protected=json.loads((OUT/'protected_previous.json').read_text());actual={str(p.relative_to(ROOT)):sha(p) for stage in ['stage4','stage5','stage6'] for p in (ROOT/'experiments'/stage).rglob('*') if p.is_file()};assert actual==protected['files']
 original=subprocess.check_output(['git','show',protected['baseline_commit']+':README.md'],cwd=ROOT,text=True);current=(ROOT/'README.md').read_text()
 assert current.count('## 7단계 —')==1
 assert original[original.index('## 1단계'):].rstrip()==current[current.index('## 1단계'):current.index('## 7단계 —')].rstrip()
 metrics=json.loads((OUT/'metrics.json').read_text());section=current[current.index('## 7단계 —'):]
 for scene in [*SCENES,'macro','pooled']:
  row='| '+scene+' | '+' | '.join(f"{metrics[scene][c]['final']['auroc']:.2f} / {metrics[scene][c]['final']['ap']:.2f}" for c in ['P1','C0','N','X'])+' |';assert row in section
 for p in [ROOT/'README.md',OUT/'results.md',OUT/'explanation_examples.md']:
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' in link or link.startswith('#'):continue
   assert (p.parent/link.split('#')[0]).exists(),(str(p),link)
 for cases in json.loads((OUT/'explanation_examples.json').read_text()).values():
  for case in cases:
   for r in case['responses'].values():assert sha(ROOT/r['source'])==r['sha256'] and json.loads((ROOT/r['source']).read_text())['response']==r['response']
 tests=sorted((OUT/'execution/regression_final').glob('*/command.json'))[-1];assert json.loads(tests.read_text())['status']=='complete';count=int(re.search(r'(\d+) passed',(tests.parent/'output.log').read_text()).group(1))
 consensus=json.loads((OUT/'consultation/consensus.json').read_text());assert consensus['status']=='both_agree_to_execute' and all(consensus[c]['decision']=='AGREE TO EXECUTE' for c in ['gpt','claude'])
 assert json.loads((ROOT/'experiments/stages.json').read_text())['stage7']['status']=='complete'
 files=publication_files();result={'status':'passed','prior_stage4_stage5_stage6_files':len(actual),'all_prior_README_experiment_sections_unchanged':True,'frozen_sources_verified':True,'analysis_sources_verified':True,'both_model_execution_agreements_recorded':True,'README_metrics_and_links_verified':True,'examples_verified':True,'tests_passed':count,'publication_files_checked':len(files)}
 write(OUT/'publication_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
