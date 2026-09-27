"""Describe current canonical Stage 4 artifacts; never rerun protected scene inference."""
import csv,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from scripts.vera_stage4_common import write,sha

def main():
 out=ROOT/'experiments/stage4';normal=out/'R04/normal';audit=json.loads((normal/'independent_verification.json').read_text());assert audit['status']=='passed' and audit['experiment_status']=='failed'
 cache=json.loads((normal/'input_cache_manifest.json').read_text());done=['R01','R02','R03'];scenes={};rows=[];sources={}
 for scene in done:
  p=out/scene/'evaluation';assert json.loads((p/'status.json').read_text())['status']=='complete';scenes[scene]=json.loads((p/'metrics.json').read_text());rows.extend(csv.DictReader((p/'frame_scores.csv').open()))
  for file in [p/'frame_scores.csv',p/'metrics.json',p/'frozen.json',out/scene/'normal/normal_profile.json']:sources[str(file.relative_to(ROOT))]=sha(file)
 for file in [normal/'independent_verification.json',normal/'input_cache_manifest.json',normal/'fact_table.json',out/'r04_protected_artifacts.json',out/'explanation_examples.json']:sources[str(file.relative_to(ROOT))]=sha(file)
 summary={'status':'partial_failure','completed_evaluation_scenes':done,'R04':audit,'scenes':scenes,'conditions':{},'sources_sha256':sources,'scope':'R01-R03 existing matched frame scores only; R04 current fact-ID run has no accepted normal rules and no evaluation; no protected inference or evaluation rerun.'}
 for c in ['A','B','C']:
  part=[r for r in rows if r['condition']==c];y=[int(r['label']) for r in part];pred=[float(r['final']) for r in part];counts={k:sum(scenes[s][c]['initial_binary'][k] for s in done) for k in ['tp','fp','tn','fn']};counts['recall']=counts['tp']/(counts['tp']+counts['fn']);counts['fpr']=counts['fp']/(counts['fp']+counts['tn'])
  summary['conditions'][c]={'frames':len(part),'macro':{k:float(np.mean([scenes[s][c]['final'][k] for s in done])) for k in ['auroc','ap']},'pooled':{'auroc':100*roc_auc_score(y,pred),'ap':100*average_precision_score(y,pred)},'initial_binary_pooled':counts}
 write(out/'summary.json',summary)
 logs=sorted((out/'execution/R04_repository_regression').glob('*/command.json'));test=logs[-1];assert json.loads(test.read_text())['status']=='complete';passed=int(re.search(r'(\d+) passed',(test.parent/'output.log').read_text()).group(1))
 r04='## 4단계 R04 — fact-ID 근거 기반 정상 기준·질문 생성\n\n**현재 공식 R04 실행: 실패 — 서로 다른 3개 정상 영상의 근거 요건 미충족.** 모델·BF16·샘플링·정상 분할·support/audit 채택 기준은 고정했고, 평가 결과에 따른 추가 조정은 수행하지 않았다.\n\n'
 r04+=f"정상 생성 영상 14개·관찰 {cache['observations']}개·영상 요약 {cache['video_summaries']}개를 검증된 deterministic 입력 캐시로 사용했다. 모델 파일 {cache['model_files_sha256_checked']}개와 원본 정상 프레임 {cache['input_frames_sha256_checked']:,}개, 관찰·요약 파일의 SHA256을 확인했다. 관찰·요약 코드 의미, 프롬프트, 샘플링, 분할이 같음을 확인했으며 expensive observation inference는 재실행하지 않았다. 원문 인용 요약 3개도 정확한 원문·중심 프레임을 유지했다. [캐시 검증](R04/normal/input_cache_manifest.json).\n\n"
 r04+='각 영상의 summary fact 순서대로 `R04_training_07_F01` 형식의 ID를 Python이 부여했다. 모델에는 fact-ID와 claim을 주고 ID만 선택하게 했다. 실제 video_id/center/claim은 프로그램의 fact table로만 변환한다. 없는 ID, 직접 좌표 필드, 세 영상 미만의 근거는 invalid 처리한다. 유효 후보의 expectation 또는 question에 lowercase·양끝 공백 제거·연속 공백 단일화만 적용해 첫 후보를 남긴다. 의미 유사도 모델이나 evidence 자동 교체는 사용하지 않았다.\n\n'
 r04+=f"| fact 수 | 생성 후보 | invalid 제거 | 중복 제거 | 유효 후보 | 최종 채택 |\n|---:|---:|---:|---:|---:|---:|\n| {audit['fact_table_rows']} | {audit['candidate_rules']} | {audit['invalid_rules']} | {audit['duplicate_rules']} | 0 | {audit['accepted_rules']} |\n\n"
 r04+='| 후보 | 선택한 fact-ID | 실제 영상 수 | 결과 |\n|---|---|---:|---|\n'
 for m in audit['mappings']:r04+=f"| {m['rule_id']} | "+', '.join('`'+e['fact_id']+'`' for e in m['mapped_evidence'])+f" | {m['distinct_videos']} | invalid: 최소 3개 영상 미충족 |\n"
 r04+='\n모든 ID는 실제 fact table에 존재하고 매핑도 정확했다. 하지만 각 후보가 같은 영상의 fact 세 개를 선택했다. 다른 영상의 ID를 임의로 골라 넣거나 후보 의미를 수정하지 않았다. 최상위 JSON 배열은 규칙 내용·ID를 그대로 둔 `rules` 객체 래핑만 적용했다. 새 실행의 원문과 형식 처리·재개 명령은 실행 로그에 보존했다.\n\n'
 r04+='시각 support 호출 **0회**, held-out 정상 audit 실행 **0구간**이다(점검용 정상 영상 4개는 그대로 분리). 유효 후보가 없어 normal description/questions를 만들지 않았다. **R04 A/B/C 평가는 수행하지 않았고 AUROC/AP는 없다.** 이를 정상 점수나 50%로 채우지 않는다. 합성 fixture로 수행한 평가 실행기 회귀 테스트는 실제 R04 성능이 아니다.\n\n'
 r04+='[fact table](R04/normal/fact_table.json), [원문 후보](R04/normal/candidates.json), [invalid 사유](R04/normal/invalid_candidates.json), [중복 필터](R04/normal/duplicate_candidates.json), [독립 검산](R04/normal/independent_verification.json), [모델 시간·VRAM](R04/normal/runtime_summary.json), [실행 명령](execution/R04_normal_fact_grounded).\n\n'
 r04+=f"R01–R03의 기존 산출물 {audit['protected_R01_R03_files']:,}개는 작업 전후 SHA256이 모두 동일하고, 해당 장면의 추론·평가를 재실행하지 않았다. [보호 목록](r04_protected_artifacts.json). 전체 회귀 테스트는 **{passed}개 통과**했으며 fact-ID·중복·support/audit 실패 처리·합성 평가 전체 경로·평가 라벨 지연 로딩을 포함한다. [테스트 로그]({str((test.parent/'output.log').relative_to(out))}).\n\n"
 aggregate='## 4단계 — 현재 결과 종합\n\nR01–R03의 기존 저장 점수를 읽어 동일 범위로 집계했다. A=기존 Q0, B=정상 설명+Q0, C=동일 정상 설명+장면별 질문이다. R04에는 새 fact-ID 프로토콜의 실제 상태만 표시한다.\n\n| 장면 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |\n|---|---:|---:|---:|\n'
 for s in done:aggregate+='| '+s+' | '+' | '.join(f"{scenes[s][c]['final']['auroc']:.2f} / {scenes[s][c]['final']['ap']:.2f}" for c in ['A','B','C'])+' |\n'
 aggregate+='| R04 | 미실행 | 미실행 | 미실행 |\n\nR01–R03의 동일 13,373프레임에 대한 집계이며, R01–R04 전체 성능은 아니다.\n\n| 집계 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |\n|---|---:|---:|---:|\n'
 for unit in ['macro','pooled']:aggregate+='| '+unit+' | '+' | '.join(f"{summary['conditions'][c][unit]['auroc']:.2f} / {summary['conditions'][c][unit]['ap']:.2f}" for c in ['A','B','C'])+' |\n'
 aggregate+='\n완료된 세 장면은 모든 조건의 이상 재현율이 0%였다. 정상 설명은 주로 정지·배치에 머물렀으며, 이 결과를 상세한 공정 정상성 설명 접근 전체의 실패로 일반화하지 않는다. 동일 VLM이 설명 생성·확인을 수행했고, 이미 관찰한 IPAD 재분할에서의 탐색적 오프라인 평가다. [원문 판정 사례](explanation_examples.md), [종합 수치·해시](summary.json).\n\n공개 결과 검산 명령은 `python scripts/verify_vera_stage4_summary.py`이며 NumPy/scikit-learn이 필요하다. R04 근거 검산은 `python scripts/audit_vera_r04_normal.py`로 수행한다. 모델 재실행 명령과 실제 환경·소스 해시는 실행 로그와 장면별 frozen 기록에 있다.\n'
 body=r04+aggregate;(out/'results.md').write_text(body)
 text=(ROOT/'README.md').read_text();text=re.sub(r'^\*\*4단계 실행 종료:.*$', '**4단계 현재 상태: R01–R03 평가 완료, R04 fact-ID 후보의 다중 영상 근거 요건 미충족.** [현재 공식 결과](experiments/stage4/results.md)에 실제 상태와 검증을 기록했다.',text,count=1,flags=re.M)
 # Rebuild only this generator's named sections when rerun.
 heads=list(re.finditer(r'^## .*$',text,re.M));remove=[]
 for i,h in enumerate(heads):
  if h.group() in ['## 4단계 R04 — fact-ID 근거 기반 정상 기준·질문 생성','## 4단계 — 현재 결과 종합']:remove.append((h.start(),heads[i+1].start() if i+1<len(heads) else len(text)))
 for a,b in reversed(remove):text=text[:a]+text[b:]
 row='| **4-R04-N** | **R04 fact-ID 정상 기준·질문 생성** | **실패** | 후보 3개 모두 1개 영상만 인용: 다중 영상 근거 요건 미충족 |\n'
 if row not in text:text=text.replace('\n진행 상태는','\n'+row+'\n진행 상태는',1)
 linked=re.sub(r'\]\(([^)]+)\)',lambda m:']('+('experiments/stage4/'+m.group(1) if '://' not in m.group(1) else m.group(1))+')',body)
 (ROOT/'README.md').write_text(text.rstrip()+'\n\n'+linked)
 write(out/'status.json',{'status':'failed','outcome':'partial_failure','completed_evaluation_scenes':done,'R04_protocol':'fact-ID grounding','R04_terminal_reason':'fewer than three distinct videos per candidate','all_four_scenes_complete':False,'frames_per_condition':13373,'finished_at':time.time()})
 p=ROOT/'experiments/stages.json';registry=json.loads(p.read_text());registry['stage4_R04_normal']={'title':'R04 fact-ID 정상 기준 생성: 다중 영상 근거 요건 미충족','status':'failed','execution_approved':True,'status_file':'experiments/stage4/R04/normal/status.json','artifact_directory':'experiments/stage4/R04/normal'};registry['stage4']={'title':'R01–R03 평가와 현재 R04 fact-ID 실행 결과','status':'failed','execution_approved':True,'status_file':'experiments/stage4/status.json','artifact_directory':'experiments/stage4'};write(p,registry)
 print(json.dumps({'R04':{k:audit[k] for k in ['candidate_rules','invalid_rules','duplicate_rules','accepted_rules','support_calls','audit_windows']},'tests_passed':passed,'R01_R03_unchanged':True},indent=2))
if __name__=='__main__':main()
