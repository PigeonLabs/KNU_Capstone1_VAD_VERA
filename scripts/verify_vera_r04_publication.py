"""Final current-tree checks before publishing the canonical R04 execution."""
import json,hashlib,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import sha,write
from scripts.publish_vera_stage import publication_files

def main():
 out=ROOT/'experiments/stage4';r04=out/'R04/normal';audit=json.loads((r04/'independent_verification.json').read_text());assert audit['status']=='passed'
 assert audit['candidate_rules']==3 and audit['invalid_rules']==3 and audit['duplicate_rules']==0 and audit['accepted_rules']==0
 assert audit['all_fact_ids_exist'] and all(m['distinct_videos']==1 for m in audit['mappings'])
 assert not (out/'R04/evaluation').exists() and not (r04/'normal_profile.json').exists()
 assert not (r04/'failure_verification.json').exists()
 for name in ['R04_normal','R04_failure_audit_report','controller_failure','summary_with_failure']:
  assert not (out/'execution'/name).exists()
 protection=json.loads((out/'r04_protected_artifacts.json').read_text())['files'];actual={str(p.relative_to(ROOT)):sha(p) for s in ['R01','R02','R03'] for p in (out/s).rglob('*') if p.is_file()};assert actual==protection
 old=subprocess.check_output(['git','show','HEAD:README.md'],cwd=ROOT,text=True);new=(ROOT/'README.md').read_text()
 for scene in ['R01','R02','R03']:
  pattern=r'(?ms)^## 4단계 '+scene+r' .*?(?=^## |\Z)';assert [s.rstrip() for s in re.findall(pattern,old)]==[s.rstrip() for s in re.findall(pattern,new)]
 section=re.findall(r'(?ms)^## 4단계 R04 .*?(?=^## |\Z)',new);assert len(section)==1
 assert '| 79 | 3 | 3 | 0 | 0 | 0 |' in section[0]
 assert 'revision' not in section[0].lower() and 'history' not in section[0].lower() and 'failure_verification.json' not in new
 assert 'R04 A/B/C 평가는 수행하지 않았고 AUROC/AP는 없다' in section[0]
 for p in [out/'status.json',r04/'status.json']:assert json.loads(p.read_text())['status']=='failed'
 for file,h in json.loads((r04/'frozen.json').read_text())['source_sha256'].items():assert sha(ROOT/file)==h
 cache=json.loads((r04/'input_cache_manifest.json').read_text())
 for relative,h in cache['cached_artifacts'].items():assert sha(r04/relative)==h
 tests=sorted((out/'execution/R04_repository_regression').glob('*/command.json'))[-1];assert json.loads(tests.read_text())['status']=='complete';assert re.search(r'82 passed',(tests.parent/'output.log').read_text())
 files=publication_files();assert all('/R04/revision' not in r['path'] for r in files)
 result={'status':'passed','canonical_R04_only':True,'R04_experiment_status':'failed','reason':'distinct-video evidence requirement not met','all_fact_ids_and_mapping_valid':True,'invalid_candidates':3,'duplicates':0,'accepted_rules':0,'visual_support_calls':0,'audit_windows':0,'evaluation_executed':False,'R01_R03_files_unchanged':len(protection),'R01_R03_readme_sections_unchanged':True,'cache_hashes_and_frozen_sources_verified':True,'repository_tests_passed':82,'publication_policy_files_checked':len(files),'README_matches_current_artifacts':True,'obsolete_R04_artifacts_absent':True}
 write(out/'R04/publication_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
