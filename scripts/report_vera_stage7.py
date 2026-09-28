"""Evidence-based Korean Stage7 report, preserving every previous experiment section."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.vera_stage4_common import write,SCENES
OUT=ROOT/'experiments/stage7';METHODS=['P1','C0','N','X'];STAGES=['initial','retrieved','smoothed','final']

def main():
 assert json.loads((OUT/'status.json').read_text())['status']=='complete'
 audit=json.loads((OUT/'independent_verification.json').read_text());analysis=json.loads((OUT/'analysis_verification.json').read_text());assert audit['status']==analysis['status']=='passed'
 m=json.loads((OUT/'metrics.json').read_text());b=json.loads((OUT/'bootstrap.json').read_text());null=json.loads((OUT/'permutation_diagnostics.json').read_text());controls=json.loads((OUT/'constant_control_metrics.json').read_text());refs=json.loads((OUT/'references.json').read_text());runtime=json.loads((OUT/'runtime_summary.json').read_text())
 tests=sorted((OUT/'execution/regression_final').glob('*/command.json'))[-1];assert json.loads(tests.read_text())['status']=='complete';count=int(re.search(r'(\d+) passed',(tests.parent/'output.log').read_text()).group(1))
 text='## 7단계 — 판정 우선 출력·정상 시각 참조 비교 (완료)\n\n'
 text+='5단계에서 정상 요약의 정적인 묘사, 낮은 검출률, 근거 비교 조건의 오탐, 입력과 무관한 ONE 대조군의 후처리 순위가 확인됐다. 사용자의 후속 실험 요청에 따라 **실제 GPT 6 Pro와 Claude Opus 5.5 High의 독립 검토·교차 검토를 수행하고, 둘 모두 실행에 동의한 C0/N/X 설계를 실행**했다. Claude는 첫 검토에서 Research mode를 사용했다. 동의는 실행 설계에 대한 것이며 성능 향상을 보증하는 것이 아니다. 제공자 내부 모델 빌드 ID는 검증하지 않았다. [합의 기록](consultation/consensus.json), [GPT 실제 응답 발췌](consultation/round2_gpt_excerpt.md), [Claude 실제 응답 발췌](consultation/round2_claude_excerpt.md).\n\n'
 text+='### 보완 근거와 수행 방법\n\n'
 text+='6단계는 평가 78번째 응답의 반복·판정 누락으로 중단했으며 실패 산출물을 보존했다. 평가 라벨을 열기 전에 별도 7단계 설계에 합의했다. 모든 조건의 기존 Observations/Assessment/마지막 판정 지침을 제거하고 **첫 줄 Output, 이후 Evidence 최대 40단어**로 바꿨다. 첫 줄이 정확하지 않으면 실패, 같은 값의 반복 판정은 품질 플래그, 상충·불완전·형식 오류의 뒤쪽 판정은 실패로 고정했다. 설명 누락·잘림은 명시적 이진 판정을 대체하거나 제외하지 않는다. 생성 한도 128토큰을 실행 중 바꾸지 않았고 6단계 예측은 재사용하지 않았다.\n\n'
 text+='Codex가 정상 학습 영상만 직접 확인했다. R01의 세 프레임에는 녹색 벨트형 표면 위 작은 물체의 위치 변화가 보였고, R02의 세 프레임 중 한 장에는 다이얼 기기가 있고 두 장에는 없었다. 기존 물체 명칭이 잘못된 대상에 주의를 돌렸을 가능성을 분리하기 위해 **중립 문구와 판정 우선 출력을 적용한 C0**를 만들었다. 장면당 정상 학습 영상 하나의 1–3장 관찰이며 독립적인 사람의 판정이나 완전한 공정 명세는 아니다. 기기가 항상 있어야 한다거나 필수 동작·순서를 새로 정하지 않았다. [관찰·원본 SHA256](../stage6/consultation/training_visual_inspection.json).\n\n'
 text+='| 조건 | 실제 입력 및 차이 |\n|---|---|\n| P1 | Stage5의 기존 점수·응답 재사용, 재추론 없음 |\n| C0 | 중립 물체 명칭과 판정 우선 출력, query 8장 |\n| N | C0에 같은 장면의 정상 TRAIN 참조 4장 추가, 총 12장 |\n| X | N과 텍스트·순서·이미지 수 동일, 참조 출처만 다음 장면으로 변경 |\n\n'
 text+='N/X의 프롬프트는 장면별로 **바이트 단위 동일**하다. 참조는 독립적인 정상 still 예시이며 연속 영상이나 현재 장면의 필수 명세로 표현하지 않았다. 비교 불가능하면 query 근거만 사용하도록 했다. X 출처는 R01→R02→R03→R04→R01로 고정했다. **주 비교는 N−C0의 초기 장면 macro AUROC**, 핵심 보조 비교는 N−X, C0−P1은 물체 명칭·출력 순서·토큰 한도가 함께 바뀐 서술적 비교다. N−C0는 이미지·설명·추가 시각 토큰을 함께 더한 효과다. X는 무관한 장면 때문에 혼란을 줄 수 있어 N−X만 높다고 성능 개선을 주장하지 않는다. [정확한 프롬프트](prompts.json), [사전 고정 프로토콜](protocol.json).\n\n'
 text+='참조는 현재 분할의 정상 TRAIN 영상 ID를 정렬하고 0-based 인덱스 `floor((i+0.5)*n/4)`의 영상에서 `floor((F-1)*(i+0.5)/4)` 프레임을 선택했다(i=0..3). 사람이 참조를 교체하거나 query·평가 결과로 고르지 않았다. 이전 승인된 재분할 때문에 경로의 `testing`은 원래 폴더명일 수 있으며, **현재 evaluation 분할과는 겹치지 않는다.**\n\n| 참조 장면 | Ref1 | Ref2 | Ref3 | Ref4 |\n|---|---|---|---|---|\n'
 for scene in SCENES:text+='| '+scene+' | '+' | '.join(f"`{r['video_id']}:{r['frame_id']}`" for r in refs['references'][scene])+' |\n'
 text+='\n[참조 선택·해시](references.json). 각 장면의 참조 영상과 겹치지 않는 첫 정상 TRAIN 영상에서 다섯 창을 골라 60회 사전 점검을 수행했다. 이후 **37영상·16,862프레임·조건당 1,072창, 신규 평가 3,216회**를 완료했다. 파싱 실패·재시도·모델 학습·validation 성능 기반 선택은 없었다. 모든 평가 추론 후에만 원본 프레임 라벨 파일을 열었으며, 영상 단위 분할 메타데이터가 프롬프트에 전달되지 않음을 검증했다.\n\n'
 text+='InternVL2-8B BF16/eager, seed0, greedy128, stride16, clipped10초(30FPS 가정)·8프레임, RGB448와 이미지당1패치를 유지했다. 참조 이미지는 VLM에만 추가했고, 후처리는 **기존 query-only ImageBind 특징→retrieval→Gaussian smoothing→위치 Gaussian** 그대로다. 짧은 창·logits·참조 검색 bank는 수행하지 않았다. [동결 fingerprint](frozen.json), [실행 명령과 출력](execution).\n\n'
 all_normal=all(m[scene][c]['initial_binary']['tp']==0 and m[scene][c]['initial_binary']['fp']==0 for scene in SCENES for c in ['C0','N','X'])
 if all_normal:
  text+='### 핵심 결과: 실행 완료, 이상 검출 개선 없음\n\n**새 세 조건 모두 1,072개 구간 전부를 정상(0)으로 판정했다.** 총 3,216개 평가 응답이 유효했지만 이상 recall은 모두 0%, FPR도 0%다. 모든 점수 단계에서 AUROC 50%이며 AP는 장면별 이상 프레임 비율과 같다. macro AP 20.20%는 네 장면 비율의 평균이고 pooled AP 19.59%는 전체 프레임 비율이다. 단순 accuracy가 높게 보이더라도 이상을 검출한 것이 아니다. 정상 참조는 **이번 결정 우선 출력 프로토콜의 관측된 이진 판정**을 바꾸지 못했다. 정상 참조 방법 일반의 무효나 모델 내부에서 이미지가 무시됐음을 증명하지는 않는다. 이진 판정이 바닥값에 머물렀으므로 내부의 측정하지 않은 선호 변화는 판단할 수 없고, 설명 문장은 조건별로 달라질 수 있다. 초기 0은 검색·smoothing·위치 가중 후에도 0이므로 ZERO 대조군과 일치한다. 이번 새 세 조건에는 ONE 대조군에서 보인 위치 기반 순위 상승이 발생하지 않았다.\n\nStage5 P1 대비 물체 문구·출력 순서·토큰 한도가 함께 달라져 어느 하나를 원인으로 분리할 수 없다. R04의 P1 최종 AUROC/AP 69.43/59.05 대비 이번 세 조건은 50.00/32.39였다. P1과의 차이는 서술적 비교다. 평가 결과를 본 뒤 프롬프트·참조·한도·판정 기준을 바꾸거나 추가 추론하지 않았다.\n\n'
 text+='### 최종 AUROC / AP (%)\n\n| 장면 | P1 | C0 | N | X |\n|---|---:|---:|---:|---:|\n'
 for scene in [*SCENES,'macro','pooled']:text+='| '+scene+' | '+' | '.join(f"{m[scene][c]['final']['auroc']:.2f} / {m[scene][c]['final']['ap']:.2f}" for c in METHODS)+' |\n'
 text+='\n### 초기 이진 판정\n\n| 장면 | 조건 | TP | FP | FN | TN | Recall (%) | FPR (%) | AUROC / AP (%) |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for scene in [*SCENES,'pooled']:
  for c in METHODS:
   v=m[scene][c]['initial_binary'];m0=m[scene][c]['initial'];text+=f"| {scene} | {c} | {v['tp']} | {v['fp']} | {v['fn']} | {v['tn']} | {100*v['recall']:.2f} | {100*v['fpr']:.2f} | {m0['auroc']:.2f} / {m0['ap']:.2f} |\n"
 text+='\n### 단계별 장면 macro AUROC / AP (%)\n\n| 조건 | initial | retrieved | smoothed | final |\n|---|---:|---:|---:|---:|\n'
 for c in METHODS:text+='| '+c+' | '+' | '.join(f"{m['macro'][c][s]['auroc']:.2f} / {m['macro'][c][s]['ap']:.2f}" for s in STAGES)+' |\n'
 text+='\n### 주 비교와 불확실성\n\nStage5와 **정확히 같은 2,000개 장면 내 영상 단위 paired bootstrap 재표집**을 재사용했다. 영상의 전체 프레임을 함께 복원추출하고 단일 클래스 표본은 undefined로 보존했다. 아래는 macro AUROC 차이(%p)다.\n\n| 비교 | 점수 | 차이 | 95% percentile 구간 | 유효 / undefined |\n|---|---|---:|---|---:|\n'
 for pair in ['N-C0','N-X','C0-P1']:
  for stage in ['initial','final']:
   d=b['primary']['deltas_percentage_points']['macro'][pair][stage]['auroc'];ci=f"[{d['low']:.2f}, {d['high']:.2f}]" if d['low'] is not None else 'undefined';text+=f"| {pair} | {stage} | {d['point']:+.2f} | {ci} | {d['valid_resamples']} / {d['undefined_resamples']} |\n"
 d=b['primary']['deltas_percentage_points']['macro']['N-C0']['initial']['auroc']
 if d['low'] is not None and d['low']>0:verdict='주 비교의 구간은 0보다 높아 이 평가에서 참조 추가 방식의 초기 판별 개선 신호가 관측됐다. 다만 재사용한 평가셋의 탐색 결과이며 새로운 데이터에서의 확증이나 정상성 이해의 증거로 해석하지 않는다.'
 elif d['high'] is not None and d['high']<0:verdict='주 비교의 구간은 0보다 낮아 이 평가에서 정상 참조 추가가 초기 판별을 악화시켰다. 성능 개선으로 표현하지 않는다.'
 else:verdict='주 비교의 구간이 0을 포함하거나 정의되지 않아, 정상 참조 추가가 안정적인 초기 판별 개선을 만들었다고 결론 내리지 않는다.'
 text+='\n'+verdict+'\n\n'
 if all_normal:text+='N−C0와 N−X의 paired 차이 구간 [0,0]은 관측된 상수 예측이 같아서 퇴화한 결과다. 모집단에서 두 방법이 동등하거나 다른 데이터에서도 차이가 없다는 증거가 아니다.\n\n'
 sd=b['stratified_secondary']['deltas_percentage_points']['macro']['N-C0']['initial']['auroc'];text+=f"별도 보조 분석으로 장면과 영상 라벨(이상 프레임 포함 여부)을 함께 층화한 2,000회 paired bootstrap(seed0)을 계산했다. N−C0 초기 macro AUROC 구간은 [{sd['low']:.2f}, {sd['high']:.2f}]%p, 유효 {sd['valid_resamples']}회다. 주 분석을 대체하지 않았다. [전체 장면·AP·단계별 구간](bootstrap.json), [재표집 목록](bootstrap_resamples.json).\n\n"
 text+='### 상수 대조군과 순열 진단\n\n'
 text+=f"기존 ZERO/ONE을 동일 영상·후처리에서 재구성했다. ONE의 최종 macro AUROC는 {controls['macro']['ONE']['final']['auroc']:.2f}%, AP는 {controls['macro']['ONE']['final']['ap']:.2f}%였다. 영상 내 초기 구간 판정 순서를 200회 섞고(seed1) 같은 순열을 모든 조건에 적용했다. 이는 모델 재추론이 아니며 통계적 유의성 검정의 p값도 아니다. 각 영상의 양성 구간 수는 유지하지만 마지막 불완전 구간 길이 때문에 양성 프레임 수는 달라질 수 있다. 아래는 최종 macro AUROC의 진단 분포다.\n\n| 조건 | 관측값 | 순열 중앙값 | 순열 2.5–97.5 percentile |\n|---|---:|---:|---|\n"
 for c in METHODS:
  v=null['summary']['macro'][c]['final']['auroc'];text+=f"| {c} | {v['observed']:.2f} | {v['median']:.2f} | [{v['low']:.2f}, {v['high']:.2f}] |\n"
 text+='\n[상수 결과](constant_control_metrics.json), [순열 분포](permutation_diagnostics.json), [실제 순열 인덱스](permutation_indices.json), [입력/점수 라벨 정렬과 이상 입력 수별 진단](alignment_diagnostics.json). 최종 후처리 순위만으로 시각 검출이 좋아졌다고 주장하지 않는다.\n\n'
 quality=json.loads((OUT/'response_quality.json').read_text())
 text+='### 응답 품질\n\n| 조건 | 평가 유효 | 동일 판정 반복 | 설명 누락 | 설명 40단어 초과 | 반복 문장 | 128토큰·EOS 없음 |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for c in ['C0','N','X']:
  q=quality[c];text+=f"| {c} | {q['binary_valid']}/{q['calls']} | {q['duplicate_output']} | {q['evidence_missing']} | {q['evidence_over40words']} | {q['evidence_repeated_sentence']} | {q['explanation_truncated']} |\n"
 text+='\n설명 내용의 진위·판정과의 의미적 일치는 전수 검증하지 않았다. 잘림은 반환 토큰 수와 EOS로 기록했고, 이미 명시된 판정은 추정으로 바꾸지 않았다. [전체 품질 계측](response_quality.json).\n\n### 검증·재현·한계\n\n'
 text+=f"- 전체 테스트 **{count}개 통과**, {audit['inference_calls_verified']:,}개 신규 응답·{audit['raw_image_hashes_verified']:,}개 원본 이미지 해시·참조 출처·프레임 정렬·모든 점수 검산 통과. [테스트 로그]({str((tests.parent/'output.log').relative_to(OUT))}), [독립 검증](independent_verification.json), [통계 검증](analysis_verification.json).\n- Stage4·Stage5·실패 Stage6 **{audit['protected_previous_files_unchanged']:,}개 파일 SHA256 불변**. 기존 결과 재추론·덮어쓰기 없음. [보호 목록](protected_previous.json).\n- 총 모델 호출 {runtime['calls']:,}회, model.chat 합산 {runtime['model_seconds']/60:.1f}분, peak allocated {runtime['peak_allocated_gib']:.2f} GiB. 로딩·전처리·검산은 model.chat 시간에서 제외한다.\n"
 text+='- 정상 still 참조는 전체 공정의 동작·속도·순서를 정의하지 않는다. 다른 장면 참조는 무관한 시각 정보 자체로 혼란을 줄 수 있다.\n- 이미 관찰한 재분할 평가셋이며 장면별 독립 영상 수가 적다. 어떤 결과도 새로운 공정에 대한 일반화 증거로 표현하지 않는다.\n- 설명의 출력 형식 준수는 시각적 사실의 정확성과 다르다. 원문은 대체·의미 복구 없이 보존했다. 실제 생성 토큰 ID/수와 EOS를 생성 반환 텐서에서 기록했다. 재토큰화 길이는 별도 필드다. Evidence는 판정 뒤의 설명이며 판정을 유발한 추론의 증거가 아니다.\n- 정상 템플릿의 효과가 모델마다 다르다는 [MMAD 원문 §4.3](https://arxiv.org/html/2410.09453v3)은 실험 동기로만 사용했으며 현재 모델·IPAD의 개선 근거로 대신 사용하지 않았다. [독립 확인 범위](../stage6/consultation/research_checked.json).\n\n'
 text+='두 협업 모델 모두 실제 수치를 전달받은 사후 검토에서 위의 제한된 결론에 동의했다. 이는 파일 감사의 대체가 아니며, 모델들은 로컬 원본을 직접 확인하지 않았다. [결과 해석 교차 검토](consultation/result_review.json).\n\n'
 text+='실제 누락 사례는 장면별 첫 사례를 고정 규칙으로 추출했다. 예를 들어 R04/testing/02 center64는 점수 구간의 13/16프레임과 입력 5/8장이 정답상 이상인데 세 조건 모두 0이었다. N 응답은 blade-like piece가 brown sheet를 자르고 관계가 coherent하다는 설명을 냈다. R03 첫 누락 사례는 입력 8/8장이 이상 라벨이었다. 모델은 반복적으로 일관성·정지 상태를 설명 근거로 제시했으나, 그 묘사의 시각적 진위나 실패 원인을 전수 검증한 것은 아니다. 정상 still 참조 추가가 이 관측된 판정 편향을 해소하지 못했다는 범위에서 해석한다.\n\n'
 text+='[실제 응답·참조 비교 사례](explanation_examples.md), [모든 지표](metrics.json), [프레임별 점수](frame_scores.csv), [원문 추론 기록](inference).\n\n재현(기존 로컬 원본·모델·특징 캐시 필요):\n\n```bash\npython scripts/run_vera_stage7.py --phase preflight\npython scripts/run_vera_stage7.py --phase evaluation\npython scripts/audit_vera_stage7.py\npython scripts/analyze_vera_stage7.py\npython scripts/explain_vera_stage7.py\n```\n'
 (OUT/'results.md').write_text(text)
 readme=ROOT/'README.md';original=readme.read_text();assert '## 7단계 —' not in original
 backup=ROOT/'runs/README_before_stage7.md';backup.parent.mkdir(exist_ok=True)
 if not backup.exists():backup.write_text(original)
 header='**7단계 평가 완료: 판정 우선 출력·정상 참조 3조건 모두 전 구간 정상(0), 성능 개선 없음.** [실제 결과·한계](experiments/stage7/results.md).\n\n'
 original=original.replace('# VERA의 IPAD 전이 평가\n\n','# VERA의 IPAD 전이 평가\n\n'+header,1)
 row='| **7** | **R01–R04 물체 설명 수정·정상 시각 참조** | **완료** | C0/N/X 모두 이상 recall 0%, AUROC 50%, [결과](experiments/stage7/results.md) |\n'
 original=original.replace('\n\n진행 상태는','\n'+row+'\n진행 상태는',1)
 linked=re.sub(r'\]\(([^)]+)\)',lambda v:']('+('experiments/stage7/'+v.group(1) if '://' not in v.group(1) else v.group(1))+')',text)
 readme.write_text(original.rstrip()+'\n\n'+linked)
 registry=json.loads((ROOT/'experiments/stages.json').read_text());registry['stage7']={'title':'물체 설명 수정·정상 시각 참조 비교','status':'complete','execution_approved':True,'status_file':'experiments/stage7/status.json','artifact_directory':'experiments/stage7'};write(ROOT/'experiments/stages.json',registry)
 print(json.dumps({'report':'complete','primary':d,'tests':count},indent=2))
if __name__=='__main__':main()
