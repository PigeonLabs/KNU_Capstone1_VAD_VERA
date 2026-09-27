## 4단계 R04 — 규칙 생성·영상별 근거 선택 분리 (완료)

**현재 공식 R04 실행은 정상 기준 생성과 A/B/C 평가까지 완료했다.** 정상 영상 14개의 관찰 338개·검증된 요약 14개를 입력 캐시로 사용했다. 모델 파일 22개·정상 원본 프레임 5,300개·관찰 및 요약 SHA256을 확인했고 모델/config·분할·샘플링·프롬프트·관찰 코드 블록도 동일함을 검증했다. 관찰 추론은 재실행하지 않고 규칙과 근거 선택은 새로 생성했다.

실행 순서는 **영상별로 묶은 정상 summary → 규칙 텍스트만 생성 → 문자열 중복 제거 → 규칙 문구/해시 고정 → 규칙별·영상별 독립 grounding → Python의 distinct-video 계산 → 선택된 모든 근거의 시각 검증 → held-out 정상 audit → 정상 설명·질문 고정 → R04 평가**였다. grounding에는 한 영상의 facts만 주고 정확한 ID 하나 또는 NONE만 허용했다. 잘못된 출력은 NONE과 별개의 failed이며 의미 보정이나 재선택을 하지 않았다.

### 실제 후보 규칙

| ID | 조건 / 정상 기대 | 질문 | 처리 |
|---|---|---|---|
| N1 | 조건: A metal blade is stationary in the center of a metal tabletop with a green surface.<br>정상: A metal blade is stationary in the center of a metal tabletop with a green surface. | Is there a visible deviation from the expected pattern? | grounding 및 시각 검증 수행 |
| N2 | 조건: A brown wooden board is placed on the blade, cut, and then removed.<br>정상: A brown wooden board is placed on the blade, cut, and then removed. | Is there a visible deviation from the expected pattern? | 동일 질문 중복 제거; grounding 미실행 |
| N3 | 조건: A brown cardboard piece is placed on the machine, which cuts it into smaller pieces.<br>정상: A brown cardboard piece is placed on the machine, which cuts it into smaller pieces. | Is there a visible deviation from the expected pattern? | 동일 질문 중복 제거; grounding 미실행 |
| N4 | 조건: A brown cardboard piece is fed into the machine, cut, and falls onto the table.<br>정상: A brown cardboard piece is fed into the machine, cut, and falls onto the table. | Is there a visible deviation from the expected pattern? | 동일 질문 중복 제거; grounding 미실행 |
| N5 | 조건: A wooden board is placed on a metal bar, then tilted, and finally cut.<br>정상: A wooden board is placed on a metal bar, then tilted, and finally cut. | Is there a visible deviation from the expected pattern? | 동일 질문 중복 제거; grounding 미실행 |

후보 5개, schema invalid 0개, 중복 제거 4개, 최종 채택 1개다. normalization은 lowercase·strip·연속 공백 단일화이며 expectation 또는 question이 같으면 처음 것을 유지했다. N2–N5는 질문이 N1과 같아 grounding 전에 제외했다. 해당 규칙의 영상별 결과를 NONE으로 만들어 넣지 않았다.

### 영상별 grounding과 시각 검증

| 규칙 | 정상 영상 | 선택 fact | 중심 프레임 | grounding | visual support |
|---|---|---|---:|---|---|
| N1 | R04/training/01 | R04_training_01_F01 | 0 | selected | supported |
| N1 | R04/training/03 | R04_training_03_F05 | 160 | selected | supported |
| N1 | R04/training/07 | NONE | — | none | 미실행 |
| N1 | R04/training/08 | NONE | — | none | 미실행 |
| N1 | R04/training/09 | R04_training_09_F06 | 80 | selected | supported |
| N1 | R04/training/10 | R04_training_10_F04 | 16 | selected | supported |
| N1 | R04/training/11 | R04_training_11_F02 | 64 | selected | supported |
| N1 | R04/training/12 | R04_training_12_F02 | 16 | selected | supported |
| N1 | R04/training/14 | R04_training_14_F02 | 16 | selected | supported |
| N1 | R04/training/16 | R04_training_16_F01 | 0 | selected | supported |
| N1 | R04/training/19 | R04_training_19_F01 | 0 | selected | supported |
| N1 | R04/training/22 | R04_training_22_F04 | 64 | selected | supported |
| N1 | R04/training/23 | R04_training_23_F02 | 64 | selected | supported |
| N1 | R04/training/24 | R04_training_24_F01 | 0 | selected | supported |

| 규칙 | grounded distinct videos | NONE | grounding failed | visual supported / contradicted / unobservable / failed | audit supported / contradicted / unobservable / failed | 채택 |
|---|---:|---:|---:|---|---|---|
| N1 | 12 | 2 | 0 | 12 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 채택 |

선택된 모든 generation-video 근거를 실제 8프레임 창에서 검사했다. distinct supported 영상 ≥3, 명확한 contradiction/failed 없음 조건을 유지했다. 별도 정상 영상 4개의 100개 창을 모두 audit했고, unobservable을 contradiction으로 바꾸지 않았다. 같은 VLM의 판단이므로 독립 사람 검증은 아니다.

### 고정 정상 설명과 질문

```text
Normal reference for R04 (normal training observations, not exhaustive process specifications):
N1. When A metal blade is stationary in the center of a metal tabletop with a green surface.: A metal blade is stationary in the center of a metal tabletop with a green surface.
Allowed variation / uncertainty: apply only when the stated condition is visible. Other phases, occlusion and unobserved details are not automatically violations. Exact timing and a mandatory global step order have not been established.

1. Is there a visible deviation from the expected pattern?
```

정상 설명은 정지한 칼날의 배치에 집중하고 조건과 기대 상태가 유사하며 질문도 포괄적이다. 상세한 공정 순서·필수 동작이 검증됐다고 해석하지 않는다. 이런 한계에도 평가 결과를 보고 문구를 수정하지 않았다.

### R04 실제 평가

A=기존 Q0와 기존 ImageBind 특징을 검증 후 재사용, B=고정 정상 설명+Q0, C=동일 정상 설명+채택 질문. InternVL2-8B BF16 동결, 30 FPS 가정, stride 16·10초 창·8프레임, 기존 retrieval/smoothing/위치 가중치를 유지했다. 평가 프레임 정답은 B/C 추론이 모두 끝난 뒤 점수 계산에 사용했다.

| 조건 | AUROC (%) | AP (%) | 초기 TP | FP | FN | 초기 recall (%) |
|---|---:|---:|---:|---:|---:|---:|
| A | 50.00 | 32.39 | 0 | 0 | 1130 | 0.00 |
| B | 54.31 | 38.73 | 0 | 16 | 1130 | 0.00 |
| C | 50.00 | 32.39 | 0 | 0 | 1130 | 0.00 |

B의 단계별 점수는 다음과 같다.

| 점수 단계 | AUROC (%) | AP (%) |
|---|---:|---:|
| initial | 49.66 | 32.39 |
| retrieved | 49.63 | 32.29 |
| smoothed | 54.14 | 34.51 |
| final | 54.31 | 38.73 |

A/C는 모든 구간을 정상으로 판정했다. B의 유일한 양성 구간은 R04/testing/15의 중심 32, 점수 구간 [32,48)로 프레임 정답 기준 FP 16개·TP 0개다. 다만 8개 입력 프레임 중 5개에는 이상 라벨이 있어 이를 정상 영상만 보고 낸 오탐으로 단정하지 않는다. B의 최종 순위 지표 상승은 smoothing/위치 가중치 이후 나타났고, 초기 프레임 단위 이상 recall은 0%였다. 이 결과만으로 직접적인 이상 검출 능력이 개선되었다고 결론 내리지 않는다. 상세 입력·점수 정렬과 원문은 [실제 판정 사례](explanation_examples.md)에 보존했다.

전체 테스트 **88개 통과**. R01–R03 산출물 **5,635개 SHA256 불변**, 추론·평가 재실행 없음. [테스트 로그](execution/R04_separated_regression_final/20260927T161810Z_a3762195/output.log), [보호 목록](r04_protected_artifacts.json), [normal 독립 검산](R04/normal/independent_verification.json), [평가 독립 검산](R04/evaluation/independent_verification.json).

[후보 원문](R04/normal/candidates.json), [고정 규칙과 해시](R04/normal/candidate_rules_frozen.json), [14개 grounding 원문/결과](R04/normal/grounding_summary.json), [fact table](R04/normal/fact_table.json), [정상 설명](R04/normal/normal_description.txt), [질문](R04/normal/questions.txt), [평가 프롬프트](R04/evaluation/prompts.json), [정렬된 단계별 프레임 점수](R04/evaluation/frame_scores.csv), [실행 명령](execution/R04_separated_normal).

## 4단계 — 현재 결과 종합

R01–R03는 기존 고정 결과를 그대로 읽었으며 R04만 이번에 평가했다. 네 장면의 동일 16,862프레임에서 A/B/C를 비교한다.

| 장면 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |
|---|---:|---:|---:|
| R01 | 50.00 / 16.47 | 50.00 / 16.47 | 50.00 / 16.47 |
| R02 | 50.00 / 7.91 | 47.37 / 7.91 | 50.00 / 7.91 |
| R03 | 46.41 / 24.05 | 50.00 / 24.05 | 50.00 / 24.05 |
| R04 | 50.00 / 32.39 | 54.31 / 38.73 | 50.00 / 32.39 |
| macro | 49.10 / 20.20 | 50.42 / 21.79 | 50.00 / 20.20 |
| pooled | 48.88 / 19.59 | 50.73 / 22.04 | 50.00 / 19.59 |

이미 관찰한 IPAD 재분할의 탐색적 오프라인 평가다. 장면별 실행 프로토콜에는 차이가 있어 동일한 정상 기준 생성 방식의 네 장면 반복으로 해석하지 않는다. [종합 수치·해시](summary.json), [실제 판정 사례](explanation_examples.md). 공개 점수 재검산: `python scripts/verify_vera_stage4_summary.py`.
