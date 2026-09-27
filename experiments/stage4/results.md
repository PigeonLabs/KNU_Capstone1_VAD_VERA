## 4단계 R04 — fact-ID 근거 기반 정상 기준·질문 생성

**현재 공식 R04 실행: 실패 — 서로 다른 3개 정상 영상의 근거 요건 미충족.** 모델·BF16·샘플링·정상 분할·support/audit 채택 기준은 고정했고, 평가 결과에 따른 추가 조정은 수행하지 않았다.

정상 생성 영상 14개·관찰 338개·영상 요약 14개를 검증된 deterministic 입력 캐시로 사용했다. 모델 파일 22개와 원본 정상 프레임 5,300개, 관찰·요약 파일의 SHA256을 확인했다. 관찰·요약 코드 의미, 프롬프트, 샘플링, 분할이 같음을 확인했으며 expensive observation inference는 재실행하지 않았다. 원문 인용 요약 3개도 정확한 원문·중심 프레임을 유지했다. [캐시 검증](R04/normal/input_cache_manifest.json).

각 영상의 summary fact 순서대로 `R04_training_07_F01` 형식의 ID를 Python이 부여했다. 모델에는 fact-ID와 claim을 주고 ID만 선택하게 했다. 실제 video_id/center/claim은 프로그램의 fact table로만 변환한다. 없는 ID, 직접 좌표 필드, 세 영상 미만의 근거는 invalid 처리한다. 유효 후보의 expectation 또는 question에 lowercase·양끝 공백 제거·연속 공백 단일화만 적용해 첫 후보를 남긴다. 의미 유사도 모델이나 evidence 자동 교체는 사용하지 않았다.

| fact 수 | 생성 후보 | invalid 제거 | 중복 제거 | 유효 후보 | 최종 채택 |
|---:|---:|---:|---:|---:|---:|
| 79 | 3 | 3 | 0 | 0 | 0 |

| 후보 | 선택한 fact-ID | 실제 영상 수 | 결과 |
|---|---|---:|---|
| N1 | `R04_training_01_F01`, `R04_training_01_F02`, `R04_training_01_F03` | 1 | invalid: 최소 3개 영상 미충족 |
| N2 | `R04_training_03_F01`, `R04_training_03_F02`, `R04_training_03_F03` | 1 | invalid: 최소 3개 영상 미충족 |
| N3 | `R04_training_07_F01`, `R04_training_07_F02`, `R04_training_07_F03` | 1 | invalid: 최소 3개 영상 미충족 |

모든 ID는 실제 fact table에 존재하고 매핑도 정확했다. 하지만 각 후보가 같은 영상의 fact 세 개를 선택했다. 다른 영상의 ID를 임의로 골라 넣거나 후보 의미를 수정하지 않았다. 최상위 JSON 배열은 규칙 내용·ID를 그대로 둔 `rules` 객체 래핑만 적용했다. 새 실행의 원문과 형식 처리·재개 명령은 실행 로그에 보존했다.

시각 support 호출 **0회**, held-out 정상 audit 실행 **0구간**이다(점검용 정상 영상 4개는 그대로 분리). 유효 후보가 없어 normal description/questions를 만들지 않았다. **R04 A/B/C 평가는 수행하지 않았고 AUROC/AP는 없다.** 이를 정상 점수나 50%로 채우지 않는다. 합성 fixture로 수행한 평가 실행기 회귀 테스트는 실제 R04 성능이 아니다.

[fact table](R04/normal/fact_table.json), [원문 후보](R04/normal/candidates.json), [invalid 사유](R04/normal/invalid_candidates.json), [중복 필터](R04/normal/duplicate_candidates.json), [독립 검산](R04/normal/independent_verification.json), [모델 시간·VRAM](R04/normal/runtime_summary.json), [실행 명령](execution/R04_normal_fact_grounded).

R01–R03의 기존 산출물 5,635개는 작업 전후 SHA256이 모두 동일하고, 해당 장면의 추론·평가를 재실행하지 않았다. [보호 목록](r04_protected_artifacts.json). 전체 회귀 테스트는 **82개 통과**했으며 fact-ID·중복·support/audit 실패 처리·합성 평가 전체 경로·평가 라벨 지연 로딩을 포함한다. [테스트 로그](execution/R04_repository_regression/20260927T155649Z_479f8739/output.log).

## 4단계 — 현재 결과 종합

R01–R03의 기존 저장 점수를 읽어 동일 범위로 집계했다. A=기존 Q0, B=정상 설명+Q0, C=동일 정상 설명+장면별 질문이다. R04에는 새 fact-ID 프로토콜의 실제 상태만 표시한다.

| 장면 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |
|---|---:|---:|---:|
| R01 | 50.00 / 16.47 | 50.00 / 16.47 | 50.00 / 16.47 |
| R02 | 50.00 / 7.91 | 47.37 / 7.91 | 50.00 / 7.91 |
| R03 | 46.41 / 24.05 | 50.00 / 24.05 | 50.00 / 24.05 |
| R04 | 미실행 | 미실행 | 미실행 |

R01–R03의 동일 13,373프레임에 대한 집계이며, R01–R04 전체 성능은 아니다.

| 집계 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |
|---|---:|---:|---:|
| macro | 48.80 / 16.14 | 49.12 / 16.14 | 50.00 / 16.14 |
| pooled | 48.64 / 16.25 | 48.86 / 16.25 | 50.00 / 16.25 |

완료된 세 장면은 모든 조건의 이상 재현율이 0%였다. 정상 설명은 주로 정지·배치에 머물렀으며, 이 결과를 상세한 공정 정상성 설명 접근 전체의 실패로 일반화하지 않는다. 동일 VLM이 설명 생성·확인을 수행했고, 이미 관찰한 IPAD 재분할에서의 탐색적 오프라인 평가다. [원문 판정 사례](explanation_examples.md), [종합 수치·해시](summary.json).

공개 결과 검산 명령은 `python scripts/verify_vera_stage4_summary.py`이며 NumPy/scikit-learn이 필요하다. R04 근거 검산은 `python scripts/audit_vera_r04_normal.py`로 수행한다. 모델 재실행 명령과 실제 환경·소스 해시는 실행 로그와 장면별 frozen 기록에 있다.
