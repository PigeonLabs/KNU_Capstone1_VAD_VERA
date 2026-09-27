"""Current R04 publication checks, preserving all R01-R03 artifacts and descriptions."""
import json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write
from scripts.publish_vera_stage import publication_files

def main():
 out=ROOT/'experiments/stage4';normal=out/'R04/normal';audit=json.loads((normal/'independent_verification.json').read_text());assert audit['status']=='passed' and audit['experiment_status']=='complete'
 evaluation=out/'R04/evaluation';assert json.loads((evaluation/'independent_verification.json').read_text())['status']=='passed'
 assert audit['candidate_text_immutable'] and audit['all_grounded_evidence_visually_checked']
 protection=json.loads((out/'r04_protected_artifacts.json').read_text())['files'];actual={str(p.relative_to(ROOT)):sha(p) for s in ['R01','R02','R03'] for p in (out/s).rglob('*') if p.is_file()};assert actual==protection
 old=subprocess.check_output(['git','show','HEAD:README.md'],cwd=ROOT,text=True);new=(ROOT/'README.md').read_text()
 for scene in ['R01','R02','R03']:
  pattern=r'(?ms)^## 4단계 '+scene+r' .*?(?=^## |\Z)';assert [s.rstrip() for s in re.findall(pattern,old)]==[s.rstrip() for s in re.findall(pattern,new)]
 section=re.findall(r'(?ms)^## 4단계 R04 .*?(?=^## |\Z)',new);assert len(section)==1
 assert 'revision' not in section[0].lower() and 'history' not in section[0].lower()
 assert '다중 영상 근거 요건 미충족' not in new and '실제 R04 성능이 아니다' not in section[0]
 metrics=json.loads((evaluation/'metrics.json').read_text())
 for condition in ['A','B','C']:
  m=metrics[condition];assert f"| {condition} | {m['final']['auroc']:.2f} | {m['final']['ap']:.2f} |" in section[0]
 for p in [out/'status.json',normal/'status.json',evaluation/'status.json']:assert json.loads(p.read_text())['status']=='complete'
 for stage in [normal,evaluation]:
  for file,h in json.loads((stage/'frozen.json').read_text())['source_sha256'].items():assert sha(ROOT/file)==h
 for relative,h in json.loads((normal/'input_cache_manifest.json').read_text())['cached_artifacts'].items():assert sha(normal/relative)==h
 tests=sorted((out/'execution/R04_separated_regression_final').glob('*/command.json'))[-1];assert json.loads(tests.read_text())['status']=='complete';passed=int(re.search(r'(\d+) passed',(tests.parent/'output.log').read_text()).group(1))
 for name in ['R04_normal_fact_grounded','R04_normal_independent_audit','R04_current_stage4_report']:assert not (out/'execution'/name).exists()
 files=publication_files()
 result={'status':'passed','R04_normal_and_evaluation':'complete','canonical_R04_only':True,'candidate_rules':audit['candidate_rules'],'duplicate_rules':audit['duplicate_rules'],'accepted_rules':audit['accepted_rules'],'grounding_by_rule':audit['grounding_by_rule'],'support_calls':audit['support_calls'],'audit_windows':audit['audit_windows'],'R04_evaluation_frames_per_condition':json.loads((evaluation/'status.json').read_text())['frames_per_condition'],'R01_R03_files_unchanged':len(protection),'R01_R03_readme_sections_unchanged':True,'cache_hashes_and_frozen_sources_verified':True,'repository_tests_passed':passed,'publication_policy_files_checked':len(files),'README_matches_current_artifacts':True}
 write(out/'R04/publication_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
