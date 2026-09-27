"""Report completed comparisons and the verified failed scene without imputing results."""
import csv,json,re,sys,time,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from scripts.vera_stage4_common import sha,write,SCENES

def main():
 out=ROOT/'experiments/stage4';done=['R01','R02','R03'];scenes={};allrows=[];sources={}
 failed=json.loads((out/'R04/normal/failure_verification.json').read_text());assert failed['status']=='failure_verified'
 for scene in done:
  p=out/scene/'evaluation'
  assert json.loads((p/'status.json').read_text())['status']=='complete'
  assert json.loads((p/'independent_verification.json').read_text())['status']=='passed'
  scenes[scene]=json.loads((p/'metrics.json').read_text());allrows.extend(csv.DictReader((p/'frame_scores.csv').open()))
  for file in [p/'frame_scores.csv',p/'metrics.json',p/'frozen.json',out/scene/'normal/normal_profile.json']:sources[str(file.relative_to(ROOT))]=sha(file)
 analyzer=ROOT/'scripts/analyze_vera_stage4_explanations.py'
 shutil.copyfile('/tmp/vera_stage4_work/analyze_vera_stage4_explanations.py',analyzer)
 subprocess.run([sys.executable,'scripts/logged_command.py','--stage','stage4','--step','explanation_analysis','--',sys.executable,str(analyzer)],cwd=ROOT,check=True)
 subprocess.run([sys.executable,'scripts/logged_command.py','--stage','stage4','--step','final_tests','--',sys.executable,'-m','pytest','-q'],cwd=ROOT,check=True)
 for file in [out/'explanation_examples.json',out/'R04/normal/failure_verification.json']:sources[str(file.relative_to(ROOT))]=sha(file)
 summary={'status':'partial_failure','completed_evaluation_scenes':done,'failed_scene':'R04','R04_failure':failed,'scenes':scenes,'conditions':{},'sources_sha256':sources,'scope':'Matched R01-R03 only; exploratory existing split; R04 extraction failed and no B/C evaluation exists; no best-prompt selection.'}
 for c in ['A','B','C']:
  rows=[r for r in allrows if r['condition']==c];y=[int(r['label']) for r in rows];scores=[float(r['final']) for r in rows]
  counts={k:sum(scenes[s][c]['initial_binary'][k] for s in done) for k in ['tp','fp','tn','fn']}
  counts['recall']=counts['tp']/(counts['tp']+counts['fn']);counts['fpr']=counts['fp']/(counts['fp']+counts['tn'])
  summary['conditions'][c]={'frames':len(rows),'videos':len({(r['scene'],r['original_split'],r['video']) for r in rows}),'macro':{k:float(np.mean([scenes[s][c]['final'][k] for s in done])) for k in ['auroc','ap']},'pooled':{'auroc':100*roc_auc_score(y,scores),'ap':100*average_precision_score(y,scores)},'initial_binary_pooled':counts}
 summary['normal_profiles']={s:{'accepted_rules':json.loads((out/s/'normal/status.json').read_text())['accepted_rules'],'normal_tokens':json.loads((out/s/'normal/normal_profile.json').read_text())['normal_tokens'],'summary_fallback_videos':json.loads((out/s/'normal/independent_verification.json').read_text()).get('summary_fallback_videos',[])} for s in done}
 summary['delta_macro_auroc']={'B_minus_A':summary['conditions']['B']['macro']['auroc']-summary['conditions']['A']['macro']['auroc'],'C_minus_B':summary['conditions']['C']['macro']['auroc']-summary['conditions']['B']['macro']['auroc']}
 write(out/'summary.json',summary)
 body='## 4단계 — 실제 실행 결과 종합 (R04 실패 포함)\n\n**R01–R03 A/B/C 평가는 완료됐고, R04는 정상 기준 생성 실패로 B/C 평가를 수행하지 못했다. R01–R04 전체 완료로 표시하지 않는다.** 장면별 정상 기준과 질문을 독립 생성했고, 정상 학습 자료만 사용했다. A는 기존 Q0, B는 정상 설명+Q0, C는 같은 정상 설명+장면별 질문이다. 모델은 동결했으며 optimizer 반복·검증 성능 기반 질문 재선택은 수행하지 않았다.\n\n| 장면 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) | B−A AUROC (%p) | C−B AUROC (%p) |\n|---|---:|---:|---:|---:|---:|\n'
 for s in done:
  m=scenes[s];body+='| '+s+' | '+' | '.join(f"{m[c]['final']['auroc']:.2f} / {m[c]['final']['ap']:.2f}" for c in ['A','B','C'])+f" | {m['B']['final']['auroc']-m['A']['final']['auroc']:+.2f} | {m['C']['final']['auroc']-m['B']['final']['auroc']:+.2f} |\n"
 body+='| R04 | — | 미실행 | 미실행 | — | — |\n\nR04는 정상 영상 14개·338구간 관찰 후 후보 5개 모두 동일한 잘못된 근거 인용으로 탈락했다. 요약에 없는 `R04/training/07` 중심 프레임 `0`을 인용했다. 이를 정상 판정이나 AUROC 50으로 대체하지 않았다. [실패 검산](R04/normal/failure_verification.json).\n\n'
 frames=summary['conditions']['A']['frames'];videos=summary['conditions']['A']['videos']
 body+=f'아래 집계는 **R01–R03만의 {videos}개 영상·{frames:,}프레임**을 A/B/C 동일하게 사용했다. 기존 3단계의 네 장면 전체 평균과 직접 비교하지 않는다. Macro는 세 장면의 단순 평균, pooled는 해당 프레임을 합친 값이다.\n\n| R01–R03 집계 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |\n|---|---:|---:|---:|\n'
 for unit in ['macro','pooled']:body+='| '+unit+' | '+' | '.join(f"{summary['conditions'][c][unit]['auroc']:.2f} / {summary['conditions'][c][unit]['ap']:.2f}" for c in ['A','B','C'])+' |\n'
 body+='\n| 초기 이진 판정 (R01–R03 프레임) | TP | FP | TN | FN | Recall (%) | FPR (%) |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for c in ['A','B','C']:
  b=summary['conditions'][c]['initial_binary_pooled'];body+=f"| {c} | {b['tp']} | {b['fp']} | {b['tn']} | {b['fn']} | {100*b['recall']:.2f} | {100*b['fpr']:.2f} |\n"
 body+='\nRecall/FPR는 구간 이진 판정을 프레임으로 확장한 초기 점수 기준이다. 후처리 연속 점수의 AUROC/AP와 구별한다. **완료된 세 장면에서 A/B/C 모두 실제 이상 프레임을 잡지 못했다.** R03의 B/C AUROC 상승은 기존 오탐 감소이며, R02 B에는 새 오탐이 생겼다. C는 세 장면의 모든 구간을 정상으로 판정했다. 따라서 이번 자동 추출 정상 설명·질문으로 실제 이상 검출이 개선됐다는 증거는 없다.\n\n'
 body+='| 장면 | 채택 정상 규칙 | 정상 설명 토큰 | 관찰 원문 인용 대체 요약 영상 |\n|---|---:|---:|---:|\n'
 for s,p in summary['normal_profiles'].items():body+=f"| {s} | {p['accepted_rules']} | {p['normal_tokens']} | {len(p['summary_fallback_videos'])} |\n"
 body+=f"| R04 | 0 (생성 실패) | — | {len(failed['summary_fallback_videos'])} |\n"
 body+='\n실제로 채택된 정상 설명은 물체의 정지·배치에 집중되어 있고 `when applicable`처럼 적용 조건이 불명확한 문구 및 중복 규칙이 있다. 공정의 상세한 단계 순서·필수 접촉·전이 조건이 충분히 추출되었다고 보기 어렵다. 따라서 이 결과로 정상성 설명 접근 자체가 효과 없다고 일반화할 수 없다. 같은 VLM이 설명 생성과 근거 확인을 수행해 사람의 독립 검증도 아니다. 이미 관찰한 IPAD 재분할의 탐색적 결과이며 8프레임 표본과 오프라인 문맥을 사용했다.\n\n'
 body+='형식 실패·수정·재개·요약 대체를 장면별로 보존했다. 원문 인용 대체는 모델 생성 요약으로 표현하지 않는다. R01의 완료 실행기는 보존하고 R02–R04는 별도 v2 실행기를 사용했다. 평가 점수를 보고 프롬프트나 설정을 변경하지 않았으며, R04 인용 오류에 대해 근거를 임의 교체하지 않았다.\n\n[실제 판정 차이·미탐 원문 예시](explanation_examples.md), [프레임 ID·정답·응답 소스 해시](explanation_examples.json), [종합 수치와 소스 해시](summary.json). 사례는 평가 종료 후 정답으로 고른 사후 분석이고 프롬프트 선택에 사용하지 않았다. 관찰 불가 표현·규칙 ID 언급 집계는 문자열 지표이며 설명의 사실성에 대한 정답이 아니다.\n'
 (out/'results.md').write_text(body)
 text=(ROOT/'README.md').read_text();text=re.sub(r'\*\*4단계 현재 결과:.*?\n','**4단계 실행 종료: R01–R03 A/B/C 평가 완료, R04 정상 기준 생성 실패.** [종합 결과와 한계](experiments/stage4/results.md)를 확인할 수 있다. 네 장면 전체 평가 완료는 아니다.\n',text,count=1)
 linked=body
 for target in ['R04/normal/failure_verification.json','explanation_examples.md','explanation_examples.json','summary.json']:linked=linked.replace('('+target+')','(experiments/stage4/'+target+')')
 assert '## 4단계 — 실제 실행 결과 종합' not in text;(ROOT/'README.md').write_text(text+'\n'+linked)
 write(out/'status.json',{'status':'failed','outcome':'partial_failure','completed_evaluation_scenes':done,'failed_normal_scene':'R04','all_four_scenes_complete':False,'conditions_evaluated':['A','B','C'],'frames_per_condition':frames,'finished_at':time.time()})
 registry_path=ROOT/'experiments/stages.json';registry=json.loads(registry_path.read_text());registry['stage4']={'title':'R01–R03 평가 완료·R04 정상 기준 생성 실패 종합','status':'failed','execution_approved':True,'status_file':'experiments/stage4/status.json','artifact_directory':'experiments/stage4'};write(registry_path,registry)
 print(json.dumps(summary['conditions'],indent=2))
if __name__=='__main__':main()
