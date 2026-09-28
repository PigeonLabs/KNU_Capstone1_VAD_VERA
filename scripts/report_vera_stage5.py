"""Korean report of completed Stage5 evidence only; Stage4 artifacts remain untouched."""
import csv,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import write,sha,SCENES
OUT=ROOT/'experiments/stage5'

def main():
 status=json.loads((OUT/'status.json').read_text());audit=json.loads((OUT/'independent_verification.json').read_text());analysis=json.loads((OUT/'analysis_verification.json').read_text())
 assert status['status']=='complete' and audit['status']==analysis['status']=='passed'
 metrics=json.loads((OUT/'metrics.json').read_text());bootstrap=json.loads((OUT/'bootstrap.json').read_text());controls=json.loads((OUT/'constant_control_metrics.json').read_text());alignment=json.loads((OUT/'alignment_diagnostics.json').read_text());runtime=json.loads((OUT/'runtime_summary.json').read_text());preflight=json.loads((OUT/'preflight_summary.json').read_text())
 tests=sorted((OUT/'execution/regression_final').glob('*/command.json'))[-1];assert json.loads(tests.read_text())['status']=='complete';count=int(re.search(r'(\d+) passed',(tests.parent/'output.log').read_text()).group(1))
 conditions=['A','B','C','P1','P2','P3'];stages=['initial','retrieved','smoothed','final'];frames=list(csv.DictReader((OUT/'frame_scores.csv').open()))
 text='## 5단계 — 산업 공정 질문·정상 설명·근거 비교 (완료)\n\n'
 text+='사용자 요청에 따라 **GPT 6 Pro와 Claude Opus 5.5 High에 실제 설계 검토를 요청**하고, 기존 로컬 InternVL2-8B BF16으로 R01–R04 프롬프트 실험을 수행했다. 두 모델은 자문만 담당했고 성능 추론에 대신 사용하지 않았다. 모델 UI 표기를 확인했으며 내부 제공자 빌드 ID는 확인하지 않았다. [협업 발췌·채택 근거](consultation/decisions.md).\n\n'
 text+='### 수행 방법\n\n| 조건 | 실제 입력·처리 |\n|---|---|\n| A/B/C | 기존 Stage4 프레임 점수와 해시를 그대로 재사용. 재추론 없음 |\n| P1 | 장면별 중립 물체 설명 + 산업 공정 질문 + 직접 판단 |\n| P2 | P1 + 기존 정상 설명 원문 + 불완전성·동어반복 방지 지침 |\n| P3 | P2의 동일 정보 + 정상/이상 시각 근거를 대칭적으로 비교 |\n\n'
 text+='R01은 공구·녹색/검정 장치, R02는 디지털 장치·클램프·금속 바, R03은 노란 지게차형 물체·받침·흰 원통, R04는 칼날·클램프·금속 바·갈색 재료에 대한 **정상 학습 근거만** 장면별로 사용했다. 명칭은 모델 관찰에서 유래했으며 실제 물체 기능을 독립 확인한 것은 아니다. 새 정상 동작 규칙이나 결함 목록을 추측하여 추가하지 않았다. [정상 근거·해시](references.json), [장면별 프롬프트 12개](prompts.json).\n\n'
 text+='모든 조건에 동일한 이진 결정 기준을 두었다. 구체적으로 보이는 공정 불일치가 있으면 1, 지지되는 불일치가 없으면 0이다. 익숙한 물체나 배경만으로 정상이라고 단정하지 않고, 관찰되지 않는 동작·순서·정확한 시간을 추론하지 않도록 했다. P3의 필드는 `Observed changes / Evidence consistent / Evidence conflicting / Uncertainty / Output`이다. **주 비교는 P3−P2의 장면 macro 초기 AUROC**이며, 이진 초기 AUROC는 balanced accuracy와 같다. P2−P1과 P3−A는 보조 비교다.\n\n'
 text+=f"정상 학습 20개 창 × 3조건 사전 점검 **{len(preflight['records'])}회** 후 고정된 37영상·16,862프레임·조건당 1,072창을 평가했다. 새 평가 추론 **{audit['inference_calls_verified']:,}회**, 파싱 실패 0회. validation/evaluation 성능 기반 선택·질문 최적화·재시도·학습 업데이트는 하지 않았다. **모든 장면·조건 추론 완료 후에만 원본 프레임 정답 파일을 열었다.** 기존 분할 manifest의 영상 단위 메타데이터는 읽지만, 추론 입력 manifest와 프롬프트에는 정답 필드를 전달하지 않는다. stride16, clipped10초(30FPS 가정), 8프레임, BF16 eager, seed0, greedy 최대1024토큰, ImageBind→smoothing→위치 가중치를 유지했다. [사전 고정 프로토콜](protocol.json), [소스·입력 fingerprint](frozen.json).\n\n"
 text+='### 확인된 결과 해석\n\n'
 r04=metrics['R04']['P1'];r04ci=bootstrap['intervals']['R04']['P1']['final']['auroc']
 text+=f"R04의 P1은 최종 AUROC **{r04['final']['auroc']:.2f}%**, AP **{r04['final']['ap']:.2f}%**로 기존 A/B보다 높은 점추정값을 보였다. 초기 검출률은 {100*r04['initial_binary']['recall']:.2f}%, 오탐률은 {100*r04['initial_binary']['fpr']:.2f}%이고 초기 AUROC는 {r04['initial']['auroc']:.2f}%다. 최종 AUROC의 영상 bootstrap 구간은 [{r04ci['low']:.2f}, {r04ci['high']:.2f}]%로 넓으므로 안정적인 개선이 입증되었다고 표현하지 않는다.\n\n"
 text+=f"장면 전체로는 P1의 최종 macro AUROC가 {metrics['macro']['P1']['final']['auroc']:.2f}%에 머물렀다. R02 P1은 {metrics['R02']['P1']['final']['auroc']:.2f}%로 낮아졌고, R01 P3는 TP {metrics['R01']['P3']['initial_binary']['tp']} / FP {metrics['R01']['P3']['initial_binary']['fp']}프레임이었다. **정상 설명 추가(P2)와 근거 비교(P3)가 공통적으로 성능을 높인다는 가설은 지지되지 않았다.**\n\n"
 text+='실제 응답에서는 정적인 배치만으로 정상을 판단하는 경향이 남았다. R01/testing/09 중심0의 P3는 같은 응답에서 공구가 움직였다고 하면서 정상 근거에는 정지했다고 적었고, 움직임 자체를 이상 근거로 사용했다. R02/testing/02 중심208은 입력 8장과 점수 구간 모두 정상 라벨인데도 손의 기기 접촉을 이상으로 판정했다. 이는 출력 형식을 준수해도 근거의 일관성과 정상 동작 해석이 보장되지 않는 사례다. 원문 설명의 시각적 사실 여부를 사람이 재판정한 것은 아니며, 이 사후 사례를 추가 튜닝에 사용하지 않았다. [실제 응답 비교](explanation_examples.md).\n\n'
 text+='### 실제 최종 점수 AUROC / AP (%)\n\n| 장면 | A | B | C | P1 | P2 | P3 |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for scene in [*SCENES,'macro','pooled']:text+='| '+scene+' | '+' | '.join(f"{metrics[scene][c]['final']['auroc']:.2f} / {metrics[scene][c]['final']['ap']:.2f}" for c in conditions)+' |\n'
 text+='\n### 초기 프레임 판정\n\n| 장면 | 조건 | TP | FP | FN | TN | Recall (%) | FPR (%) | 초기 AUROC / AP (%) |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for scene in [*SCENES,'pooled']:
  for c in conditions:
   b=metrics[scene][c]['initial_binary'];m=metrics[scene][c]['initial'];text+=f"| {scene} | {c} | {b['tp']} | {b['fp']} | {b['fn']} | {b['tn']} | {100*b['recall']:.2f} | {100*b['fpr']:.2f} | {m['auroc']:.2f} / {m['ap']:.2f} |\n"
 text+='\n### 네 점수 단계 (장면 macro, AUROC / AP %)\n\n| 조건 | initial | retrieved | smoothed | final |\n|---|---:|---:|---:|---:|\n'
 for c in conditions:text+='| '+c+' | '+' | '.join(f"{metrics['macro'][c][s]['auroc']:.2f} / {metrics['macro'][c][s]['ap']:.2f}" for s in stages)+' |\n'
 text+='\n### 영상 단위 불확실성\n\n장면 내 영상을 복원추출하고 모든 방법에 같은 추출을 적용한 **2,000회 paired bootstrap, seed0**이다. 각 영상의 모든 프레임을 함께 가져왔으며 프레임 단위 bootstrap은 하지 않았다. 단일 클래스 표본은 undefined로 남겼다. 아래는 장면 macro AUROC의 차이와 95% percentile 구간(percentage point)이다.\n\n| 비교 | 점수 | 차이 | 95% 구간 | 유효 / undefined |\n|---|---|---:|---|---:|\n'
 for pair in ['P3-P2','P2-P1','P3-A']:
  for stage in ['initial','final']:
   d=bootstrap['deltas_percentage_points']['macro'][pair][stage]['auroc'];lo=d['low'];hi=d['high'];interval=f"[{lo:.2f}, {hi:.2f}]" if lo is not None else 'undefined';text+=f"| {pair} | {stage} | {d['point']:+.2f} | {interval} | {d['valid_resamples']} / {d['undefined_resamples']} |\n"
 primary=bootstrap['deltas_percentage_points']['macro']['P3-P2']['initial']['auroc']
 if primary['low'] is not None and primary['low']>0:interpret='사전 지정한 P3−P2의 초기 macro AUROC 차이 구간은 0보다 높았다. 다만 이미 본 평가셋의 탐색적 결과이며 확증적 유의성으로 주장하지 않는다.'
 elif primary['high'] is not None and primary['high']<0:interpret='사전 지정한 P3−P2의 초기 macro AUROC 차이 구간은 0보다 낮았다. 이 비교에서 근거 구조화가 개선되었다고 볼 수 없다.'
 else:interpret='사전 지정한 P3−P2의 초기 macro AUROC 차이 구간은 0을 포함하거나 정의되지 않았다. 이 비교에서 일관된 개선을 확인했다고 결론 내리지 않는다.'
 text+='\n'+interpret+' [전체 장면·AP·단계별 구간](bootstrap.json), [실제 영상 재표집 목록](bootstrap_resamples.json).\n\n'
 text+='### 시간 정렬과 상수 대조군\n\n| 장면 | 입력 정상 / 점수 정상 | 입력 정상 / 점수 이상 | 입력 이상 / 점수 정상 | 입력 이상 / 점수 이상 |\n|---|---:|---:|---:|---:|\n'
 for scene in [*SCENES,'pooled']:
  t=alignment[scene]['A']['input_any_target_any'];text+='| '+scene+' | '+' | '.join(str(t[k]) for k in ['input0_target0','input0_target1','input1_target0','input1_target1'])+' |\n'
 text+='\n입력 이상은 샘플 8장 중 하나라도 이상 라벨인 경우, 점수 이상은 중심부터 최대16프레임 구간에 이상 라벨이 있는 경우다. 공식 프레임 지표를 바꾸는 재라벨링이 아니라 추가 진단이다. [0 / 1–4 / 5–8개 입력 이상 프레임별 판정·recall·FPR](alignment_diagnostics.json).\n\n| 계산 대조군 | macro 초기 AUROC / AP (%) | macro 최종 AUROC / AP (%) |\n|---|---:|---:|\n'
 for c in ['ZERO','ONE']:text+=f"| {c} | {controls['macro'][c]['initial']['auroc']:.2f} / {controls['macro'][c]['initial']['ap']:.2f} | {controls['macro'][c]['final']['auroc']:.2f} / {controls['macro'][c]['final']['ap']:.2f} |\n"
 text+='\nZERO/ONE은 새 모델 추론이 아니라 모든 창을 0/1로 놓은 계산 대조군이다. ONE의 비상수 최종 점수는 경계 smoothing·위치 가중치가 만드는 순위를 보여준다. 반올림 없는 부동소수점 계산은 retrieved 단계의 수치상 동점에도 미세한 차이를 만들 수 있다. 최종 AUROC 상승만으로 시각 검출 능력 향상을 주장하지 않는다.\n\n'
 text+='### 검증·재현·한계\n\n'
 text+=f"- 전체 테스트 **{count}개 통과**, 독립 점수 재구성·라벨 정렬·해시 검증 통과. [테스트 로그]({str((tests.parent/'output.log').relative_to(OUT))}), [독립 검증](independent_verification.json), [통계 검증](analysis_verification.json).\n- Stage4 전체 **{audit['protected_stage4_files_unchanged']:,}개 파일 SHA256 불변**. R01–R04 기존 추론·평가 재실행 없음. [보호 목록](protected_stage4.json).\n- 모델 호출 합계 {runtime['calls']:,}회, model.chat 합산 {runtime['model_seconds']/60:.1f}분, peak allocated {runtime['peak_allocated_gib']:.2f} GiB. 시간은 전처리·로딩·검산을 제외한다.\n"
 text+='- 요청한 출력 필드가 모두 나타난 응답: '+', '.join(f"{c} {v['all_requested_fields_present']}/{v['records']}" for c,v in audit['requested_output_field_diagnostics'].items())+'. 이는 표현 형식 진단이며 판정값을 바꾸는 기준은 아니다.\n'
 text+='- 이미 여러 차례 관찰한 IPAD 재분할 평가이며 독립적인 새 테스트가 아니다. 영상 수가 적고 같은 촬영 세션의 상관이 남을 수 있다.\n- P1 대 기존 방법은 질문 의미·문장·형식이 함께 다르다. P2−P1은 정상 설명·해석 지침·길이를 함께 추가한다. P3−P2도 구조와 필드별 길이 배분을 바꾸므로 완전한 단일 요인 분해라고 주장하지 않는다.\n- 정상 요약은 같은 VLM이 만든 불완전한 관찰이며 정적인 물체 설명에 치우쳤다. 새 정상 공정 지식을 획득했다고 표현하지 않는다.\n- 잘못된 출력은 정상·이상으로 대체하지 않는다. 토큰 수는 디코딩 텍스트를 다시 토큰화한 길이이며 제공자가 노출하지 않은 정확한 종료 사유를 주장하지 않는다.\n\n'
 text+='[실제 판정 사례](explanation_examples.md), [원문 응답](inference), [프레임별 6조건 점수](frame_scores.csv), [모든 지표](metrics.json), [상수 대조군](constant_control_metrics.json), [실행 명령·stdout](execution).\n\n재현(기존 데이터·모델·특징 캐시 필요):\n\n```bash\npython scripts/run_vera_stage5.py --phase preflight\npython scripts/run_vera_stage5.py --phase evaluation\npython scripts/audit_vera_stage5.py\npython scripts/analyze_vera_stage5.py\n```\n'
 (OUT/'results.md').write_text(text)
 readme=ROOT/'README.md';original=readme.read_text();backup=ROOT/'runs/README_before_stage5.md';backup.parent.mkdir(exist_ok=True)
 if not backup.exists():backup.write_text(original)
 assert '## 5단계 —' not in original,'Report already appended; do not duplicate'
 linked=re.sub(r'\]\(([^)]+)\)',lambda m:']('+('experiments/stage5/'+m.group(1) if '://' not in m.group(1) else m.group(1))+')',text)
 header='**5단계 완료: GPT 6 Pro·Claude Opus 5.5 High 설계 협업 후 R01–R04의 고정 프롬프트 3종을 평가했습니다.** [실제 결과·한계](experiments/stage5/results.md).\n\n'
 original=original.replace('\n진행 상태는','\n| **5** | **R01–R04 산업 질문·정상 설명·근거 비교** | **완료** | 고정 3조건 / 37영상 / 16,862프레임, [결과](experiments/stage5/results.md) |\n\n진행 상태는',1)
 table_start=original.index('## 실험 단계');table_end=original.index('진행 상태는',table_start)
 original=original[:table_start]+re.sub(r'(?m)(^\|[^\n]*\|)\n(?:[ \t]*\n)+(?=\|)',r'\1\n',original[table_start:table_end])+original[table_end:]
 original=original.replace('# VERA의 IPAD 전이 평가\n\n','# VERA의 IPAD 전이 평가\n\n'+header,1)
 readme.write_text(original.rstrip()+'\n\n'+linked)
 registry=json.loads((ROOT/'experiments/stages.json').read_text());registry['stage5']={'title':'산업 질문·정상 설명·시각 근거 비교','status':'complete','execution_approved':True,'status_file':'experiments/stage5/status.json','artifact_directory':'experiments/stage5'};write(ROOT/'experiments/stages.json',registry)
 print(json.dumps({'status':'report_complete','primary':primary,'tests':count,'stage4_protected':audit['protected_stage4_files_unchanged']},indent=2))
if __name__=='__main__':main()
