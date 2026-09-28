"""Final Stage5 publication gate; only completed, hash-verified actual evidence."""
import json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write,SCENES
from scripts.publish_vera_stage import publication_files
OUT=ROOT/'experiments/stage5'

def main():
 for name in ['independent_verification','analysis_verification']:
  assert json.loads((OUT/(name+'.json')).read_text())['status']=='passed'
 assert json.loads((OUT/'status.json').read_text())['status']=='complete'
 frozen=json.loads((OUT/'frozen.json').read_text())
 for path,h in frozen['source_sha256'].items():assert sha(ROOT/path)==h
 protection=json.loads((OUT/'protected_stage4.json').read_text());expected=protection['files'];actual={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/stage4').rglob('*') if p.is_file()};assert actual==expected
 original=subprocess.check_output(['git','show',protection['baseline_commit']+':README.md'],cwd=ROOT,text=True);current=(ROOT/'README.md').read_text()
 for s in SCENES:
  pattern=r'(?ms)^## 4단계 '+s+r' .*?(?=^## |\Z)';assert re.findall(pattern,original)[0].rstrip()==re.findall(pattern,current)[0].rstrip()
 section=re.findall(r'(?ms)^## 5단계 .*?(?=^## |\Z)',current);assert len(section)==1
 metrics=json.loads((OUT/'metrics.json').read_text())
 for scene in [*SCENES,'macro','pooled']:
  expected_row='| '+scene+' | '+' | '.join(f"{metrics[scene][c]['final']['auroc']:.2f} / {metrics[scene][c]['final']['ap']:.2f}" for c in ['A','B','C','P1','P2','P3'])+' |';assert expected_row in section[0]
 for p in [ROOT/'README.md',OUT/'results.md',OUT/'explanation_examples.md']:
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' in link or link.startswith('#'):continue
   assert (p.parent/link.split('#')[0]).exists(),(str(p),link)
 for cases in json.loads((OUT/'explanation_examples.json').read_text()).values():
  for case in cases:
   for r in case['responses'].values():assert sha(ROOT/r['source'])==r['sha256'] and json.loads((ROOT/r['source']).read_text())['response']==r['response']
 tests=sorted((OUT/'execution/regression_final').glob('*/command.json'))[-1];assert json.loads(tests.read_text())['status']=='complete';count=int(re.search(r'(\d+) passed',(tests.parent/'output.log').read_text()).group(1))
 assert json.loads((ROOT/'experiments/stages.json').read_text())['stage5']['status']=='complete'
 files=publication_files();result={'status':'passed','frozen_sources_verified':True,'README_matches_metrics':True,'markdown_links_verified':True,'examples_verified':True,'protected_stage4_files':len(expected),'stage4_scene_README_sections_unchanged':True,'repository_tests_passed':count,'publication_files_checked':len(files)}
 write(OUT/'publication_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
