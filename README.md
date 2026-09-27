# VERA의 IPAD 전이 평가

**4단계 현재 상태: R01–R03 평가 완료, R04 fact-ID 후보의 다중 영상 근거 요건 미충족.** [현재 공식 결과](experiments/stage4/results.md)에 실제 상태와 검증을 기록했다.

**1단계: VERA 논문 추론 방법론 재현 완료.** 동결된 InternVL2-8B와 저자가 공개한 UCF-Crime 학습 질문을 IPAD R01–R04에 적용했다. 최종 macro AUROC는 **52.02%**, macro AP는 **43.79%**였다. 초기 이진 판정에서 2,001개 구간 중 **7개만 이상**으로 판정했다.

이 저장소는 VERA 실험의 코드·분석 자료·실행 로그를 관리한다. 1단계는 공개 질문 전이 평가이고 2-2단계의 질문 학습은 별도 기록했다. 원 논문 벤치마크 수치 재현을 의미하지 않는다. 미래 프레임과 전체 영상 문맥을 사용하는 **오프라인 평가**다. 원본 이미지·모델 가중치·특징 캐시는 로컬에 보존하고, 공개 자료에는 SHA256 목록을 포함한다.

## 실험 단계

| 단계 | 목적 | 상태 | 확인된 결과 / 다음 결정 |
|---|---|---|---|
| **1** | **논문 추론 방법론 재현 및 IPAD 전이 평가** | **완료** | 63개 영상, 31,550프레임, 2,001구간; macro AUROC 52.02% |
| **2-1** | **정상·이상 영상을 포함하는 IPAD 재분할** | **완료** | 학습 120개 / 검증 17개 / 평가 37개, 영상·동일 프레임 교집합 0 |
| **2-2** | **learner–optimizer 반복 질문 최적화** | **완료** | 10 epoch·600 반복, 유효 optimizer 530회, 형식 오류 70회 |
| **2-3** | **검증 정확도에 따른 질문 선택** | **완료** | 업데이트 0의 질문 선택, 검증 11/17 (64.71%) |
| **3** | **선택 질문의 IPAD 재분할 평가** | **완료** | 37개 영상·16,862프레임, macro AUROC 49.10% / AP 20.20% |
| **4-R01-N** | **R01 정상 기준·질문 생성** | **완료** | 생성 22개 / 점검 6개 정상 영상, 규칙 2/5개 채택 |
| **4-R01-E** | **R01 정상 설명·질문 A/B/C 비교** | **완료** | AUROC A 50.00 / B 50.00 / C 50.00% |
| **4-R02-N** | **R02 정상 기준·질문 생성** | **완료** | 생성 17개 / 점검 5개 정상 영상, 규칙 5/5개 채택 |
| **4-R02-E** | **R02 정상 설명·질문 A/B/C 비교** | **완료** | AUROC A 50.00 / B 47.37 / C 50.00% |
| **4-R03-N** | **R03 정상 기준·질문 생성** | **완료** | 생성 13개 / 점검 4개 정상 영상, 규칙 1/5개 채택 |
| **4-R03-E** | **R03 정상 설명·질문 A/B/C 비교** | **완료** | AUROC A 46.41 / B 50.00 / C 50.00% |


| **4-R04-N** | **R04 fact-ID 정상 기준·질문 생성** | **실패** | 후보 3개 모두 1개 영상만 인용: 다중 영상 근거 요건 미충족 |

진행 상태는 [실행 단계 기록](experiments/stages.json)을 따른다. 실제 수행한 실험만 기록하며, 제안이나 미실행 실험은 GitHub에 미리 게시하지 않는다. 실제 단계가 끝날 때 로그·결과·한국어 README를 갱신하고 `main`에 커밋·push한다.

## 1단계 — 개요와 진행 방법

기준 자료는 [원 논문 v3](https://arxiv.org/abs/2412.01095v3)와 [공식 구현](https://github.com/vera-framework/VERA)이다. 논문에 명시된 값과 공식 코드가 다를 때는 논문 값을 우선했다. 상세 수식·경계 처리·코드 차이표는 [실험 프로토콜](docs/vera_ipad_protocol.md)을 따른다.

| 항목 | 고정 설정 |
|---|---|
| 대상 | IPAD R01–R04, R02의 라벨 불일치 영상 12·13·14 제외 |
| 모델 | InternVL2-8B, 동결, BF16, eager attention, 양자화·LoRA 없음 |
| 질문 | 저자가 UCF-Crime에서 학습한 질문 5개와 learner 프롬프트 |
| 샘플링 | 중심 간격 16프레임, 중심 주변 10초에서 8프레임; 경계 자르기, 마지막 불완전 구간 포함 |
| FPS | 원본 촬영 FPS 미검증으로 30 FPS 가정, 10초 = 300프레임 |
| 장면 문맥 | ImageBind FP32, 같은 영상 안에서 `K=max(1, floor(0.1h))`, `softmax(similarity/10)` |
| 시간 문맥 | 정규화 Gaussian 크기 15·표준편차 10 → 프레임 확장 → 영상 중앙 기준 위치 가중치 |
| 평가 | 장면별·장면 macro·전체 프레임 pooled AUROC 및 AP; 중간 점수 반올림 없음 |
| 재현 기록 | seed 0, 소스·모델·입력·라벨 SHA256, 구간별 설명/시간/VRAM, 단계별 프레임 점수 |

실행 순서는 **환경·가중치 확보 → 정상 학습 영상 점검 → 설정 동결 → 전체 추론 → 평가 → 교집합 비교·검산**이었다. 라벨은 평가 단계에서만 사용했다. 첫 preflight는 SentencePiece 0.2.2 호환 문제로 실패했고, 0.2.1로 고정한 뒤 정상 실행했다. 실패 로그도 보존했다. 질문 학습·모델 학습은 수행하지 않았으므로 해당 학습 이력은 없다.

[환경 구성 및 재실행 명령](docs/environment_setup.md)에서 전체 절차를 확인할 수 있다. 공개 결과만으로 검산하려면 다음을 실행한다(이미지·GPU 불필요).

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python numpy scikit-learn
.venv/bin/python scripts/verify_publication.py
```

## 1단계 — 결과

모든 수치는 백분율이다. AP는 average precision이며, JSON의 `auprc` 필드도 같은 계산을 사용한다. macro는 R01–R04 지표의 단순 평균, pooled는 전체 프레임을 합쳐 계산한 값이다.

| 처리 단계 | macro AUROC | macro AP | pooled AUROC | pooled AP |
|---|---:|---:|---:|---:|
| 초기 이진 판정 | 50.21 | 41.77 | 50.31 | 43.13 |
| 장면 문맥 검색 | 50.70 | 42.25 | 51.04 | 43.89 |
| Gaussian smoothing | 52.03 | 43.95 | 52.89 | 46.47 |
| **위치 가중치 적용 (최종)** | **52.02** | **43.79** | **52.88** | **46.20** |

| 장면 | 영상 수 | 프레임 수 | 이상 판정 구간 / 전체 구간 | 최종 AUROC | 최종 AP |
|---|---:|---:|---:|---:|---:|
| R01 | 15 | 3,685 | 0 / 238 | 50.00 | 34.03 |
| R02 | 12 | 7,706 | 0 / 487 | 50.00 | 33.90 |
| R03 | 17 | 12,005 | 6 / 758 | 55.27 | 48.63 |
| R04 | 19 | 8,154 | 1 / 518 | 52.80 | 58.60 |
| **전체** | **63** | **31,550** | **7 / 2,001** | **52.88 pooled** | **46.20 pooled** |

정상·이상 여부 파싱 실패나 전체 평가 누락 없이 완료했다. 다만 전체 63개 중 59개 영상은 초기 점수가 모두 0이었다. R01은 15개 중 14개 영상이 300프레임보다 짧다. 사회적 이상을 묻는 질문과 산업 공정의 차이, 긴 문맥 창, 시각 인식 자체가 가능한 원인이지만, 현재 결과만으로 원인을 단정하지 않는다. 초기 판정의 낮은 이상 응답률을 실제 관측 결과로 기록했다.

### 기존 IPAD/DINOv2와 동일 프레임 비교

기존 결과가 존재하는 **동일한 30,353프레임**에서 모든 지표를 재계산했다. 아래 VERA 값은 위 전체 31,550프레임 결과와 평가 범위가 다르다. 기존 방법의 학습은 이번 실행에서 다시 수행하지 않았다.

| 방법 / 저장 점수 열 | macro AUROC | macro AP |
|---|---:|---:|
| VERA 최종 | 52.02 | 44.96 |
| IPAD `negative_psnr_with_phase` | 72.53 | 66.00 |
| DINOv2 reconstruction `dino_patch6_with_phase` | 69.98 | 62.71 |
| DINOv2 reconstruction `dino_patch12_with_phase` | 72.94 | 65.85 |
| DINOv2 reconstruction `dino_cls_with_phase` | 75.19 | 68.26 |
| DINOv2 reconstruction `dino_multilevel_with_phase` | 72.61 | 65.60 |
| DINOv2 prototype `conditional_nn_with_phase` | 75.83 | 69.03 |
| DINOv2 prototype `conditional_soft_with_phase` | 74.61 | 67.93 |
| DINOv2 prototype `unconditional_nn_with_phase` | 77.48 | 67.43 |
| DINOv2 prototype `unconditional_soft_with_phase` | 71.40 | 61.08 |

비교 원본과 해시는 [reconstruction/IPAD 비교](experiments/vera_ipad/comparisons.json), [prototype 비교](experiments/vera_ipad/prototype_comparison.json)에 있다. `experiments/stage1_reproduction`, `stage2_dinov2`는 **기존 IPAD 연구의 비교 자료 폴더명**이며, 이 저장소의 VERA 실험 단계 번호와 별개다.

### 처리 시간과 메모리

RTX PRO 6000 Blackwell 96GB에서 VLM 구간 처리 평균 **1.867초**, p95 **2.207초**, ImageBind 평균 **0.189초**였다. VLM peak allocated/reserved VRAM은 **18.62/20.21 GiB**, ImageBind는 **4.79/5.02 GiB**였다. 각 구간 측정 합계는 VLM 3,735.87초, ImageBind 377.72초이며 로딩을 포함한 전체 벽시계 시간과 다르다. 다른 GPU 작업을 중지하지 않았다. 이 수치는 미래 문맥을 쓰는 오프라인 처리 시간이며 실시간·인과적 성능으로 해석하지 않는다.

## 증거와 로그 위치

| 확인할 내용 | 파일/디렉터리 |
|---|---|
| 원 실행 상태·이벤트·실패 | [status](experiments/vera_ipad/status.json), [events](experiments/vera_ipad/events.jsonl), [preflight 실패](experiments/vera_ipad/preflight.log), [재시도](experiments/vera_ipad/preflight_02.log), [run](experiments/vera_ipad/run.log) |
| 구간별 원문 설명·판정·선택 프레임·시간 | [inference](experiments/vera_ipad/inference), [정상 영상 점검](experiments/vera_ipad/diagnostic) |
| 검색 이웃·가중치·단계별 점수 | [postprocessing](experiments/vera_ipad/postprocessing), [전체 frame_scores.csv](experiments/vera_ipad/frame_scores.csv) |
| 평가 결과·해석 | [metrics](experiments/vera_ipad/metrics.json), [results](experiments/vera_ipad/results.md), [interpretation](experiments/vera_ipad/interpretation.md), [runtime](experiments/vera_ipad/runtime_summary.json) |
| 고정 설정·질문·환경 | [frozen](experiments/vera_ipad/frozen.json), [questions](experiments/vera_ipad/questions.txt), [environment](experiments/vera_ipad/environment.txt), [VERA lock](experiments/vera_ipad/requirements-vera.lock.txt) |
| 입력·모델·소스의 해시 | [test manifest](experiments/vera_ipad/test_manifest.json), [normal manifest](experiments/vera_ipad/normal_manifest.json), [labels](experiments/vera_ipad/label_inventory.json), [models](experiments/vera_ipad/model_inventory.json), [source](experiments/vera_ipad/source_inventory.json) |
| 완료 시점 검증 | [verification](experiments/vera_ipad/verification.json), [독립 검산](experiments/vera_ipad/final_verification.json) |
| 이번 게시를 위한 검증·실행 기록 | [stage1_publication](experiments/stage1_publication), [시점별 정리](experiments/stage1_publication/stage1_ledger.json) |
| 실제 공개 파일 SHA256 | [publication_manifest](docs/publication_manifest.json) — 자기 자신 제외 |

1단계 당시 모든 설치 명령의 터미널 원문이나 최초 테스트 stdout이 수집된 것은 아니다. 남아 있는 실제 로그와 검증 요약을 그대로 보존하고, 이번 게시 검증의 명령·stdout·종료 상태는 별도로 기록한다. 게시 전 **30개 테스트 통과**(기존 23개 + 게시·로그 도구 7개), 전체 프레임 지표 및 비교 지표의 독립 재계산 일치를 확인했다. 새 로그 도구 테스트의 기대값 인덱스 오류로 발생한 최초 2개 실패와 수정 후 통과 기록도 보존했다. 과거 기록을 사후 생성한 실행 로그처럼 표시하지 않는다. 이후 실행은 `scripts/logged_command.py`로 시도별 로그를 남긴다.

## 운영 원칙

- 기존 데이터와 완료 결과를 보존한다. 디스크 여유가 **10 GiB 이하이면 실험을 중단**하며 자동 재개하지 않는다.
- 새 실험의 설정을 먼저 고정하고 최종 평가 정답으로 질문·시간 창·임계값을 선택하지 않는다. 이미 결과를 본 동일 테스트셋의 후속 비교는 탐색적 분석으로 명시한다.
- 승인된 각 실험 단계의 완료·진단·실패 상태를 구분하고 README와 근거를 함께 커밋·push한다. `scripts/publish_vera_stage.py`는 게시 대상·상태·원격 저장소를 검사하고, 원격 변경을 덮어쓰거나 force-push하지 않는다.
- [공개 자료 정책](docs/artifact_policy.md)에 따라 데이터 이미지, 가중치, 특징 벡터/캐시, 로컬 논문 PDF, 가상환경, 자격증명은 업로드하지 않는다. 코드·분석용 점수·텍스트 설명·로그·SHA256 목록을 게시한다.
- `experiments/vera_ipad` 안의 당시 문서에는 이전의 로컬 보관 정책이 남아 있다. 현재 게시 권한은 이 README와 [AGENTS.md](AGENTS.md), [로그·게시 정책](docs/logging_and_publication.md)을 따른다. 제3자 코드 출처는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있다.

## 2-1단계 — 정상·이상 데이터 재분할 (완료)

2026-09-27 사용자 승인에 따라 기존 정상 training과 testing을 합친 뒤, **영상 전체를 단위로** 학습·검증·평가를 다시 분리했다. 원본 파일은 이동·변경하지 않았고 [split.json](experiments/stage2_1/split.json)이 새 소속을 정의한다. 비디오 라벨은 기존 프레임 라벨의 최댓값으로 만들었으며, 프레임별 이상 위치는 질문 학습 입력으로 제공하지 않는다.

| 새 분할 | 정상 영상 | 이상 영상 | 합계 | 역할 |
|---|---:|---:|---:|---|
| 학습 | 85 | 35 | 120 | learner–optimizer의 비디오 라벨 기반 질문 업데이트용 |
| 검증 | 11 | 6 | 17 | 질문의 비디오 분류 정확도 비교용 |
| 최종 평가 | 26 | 11 | 37 | 학습·질문 선택에서 제외한 프레임 평가용 |

장면과 비디오 라벨로 층화하고 seed 0으로 섞었다. 각 층에서 20%를 올림하여 평가로, 남은 영상의 10%를 올림하여 검증으로 배정했다. R02/testing/12·13·14는 라벨 불일치로 제외했다. 총 174개 영상의 실제 이미지 SHA256을 기존 목록과 대조했고, 영상 중복 및 분할 사이의 **바이트 동일 프레임 교집합이 0**임을 확인했다. 전체 영상에서 `i × floor(F/8)`, `i=0..7`로 뽑는 학습 프레임도 목록에 고정했다.

이 분할의 기존 testing 영상은 1단계에서 이미 평가된 자료다. 따라서 결과는 **IPAD 재분할에 대한 탐색적 평가**이며, 기존 표준 테스트 성능이나 완전히 미관찰 데이터에 대한 일반화 성능으로 표현하지 않는다. 촬영 세션 ID가 없어 동일 세션의 유사 영상 여부까지 검증하지는 못했다.

근거: [분할 요약](experiments/stage2_1/summary.json), [완료·해시](experiments/stage2_1/status.json), [실행 로그](experiments/stage2_1/execution), [분할 코드](scripts/split_vera_ipad.py). 원 실행은 `scripts/logged_command.py --stage stage2_1 --step split -- <VERA Python> scripts/split_vera_ipad.py`이며, 완료된 분할의 덮어쓰기를 거부한다.

## 2-2단계 — learner–optimizer 반복 질문 최적화 (완료)

IPAD 학습 120개 영상에 대해 **10 epoch, 배치 2, 총 600회** 질문 업데이트를 수행했다. 모델 가중치는 동결된 InternVL2-8B BF16이며, 동일 모델이 learner와 optimizer 역할을 순차적으로 수행했다. 원 논문의 수동 초기 질문 2개(Q0)에서 시작했으며 공개 UCF 학습 질문으로 시작하지 않았다. 각 영상 전체에서 `i × floor(F/8)`로 선택한 8프레임을 learner에 전달했다. Optimizer에는 두 영상의 16프레임, learner 예측 두 개, 학습용 비디오 라벨 두 개, 현재 질문을 전달했다. 프레임별 이상 위치는 입력하지 않았다.

생성은 `do_sample=False`, `num_beams=1`, `max_new_tokens=1024`, seed 0으로 고정했다. Q0와 100회마다 생성된 질문을 검증 17개 영상으로 평가했다. **모델 파라미터 업데이트는 0회**이며 학습된 것은 질문 텍스트다.

- 반영된 learner 예측: **1,200개**, 실제 learner 호출: **1202회**(미완료 시도 2회 포함), optimizer 호출: **600회**, 검증 호출: **119회**.
- 유효 optimizer 응답: **530회**, 형식 오류로 이전 질문 유지: **70회**, 실제 질문 문자열 변경: **382회**.
- 출력은 최대 5개의 완결된 번호 질문으로 검사했다. **공식 코드의 앞 5줄 자르기와 달리**, 초과 질문·불완전 형식은 수정하거나 임의로 잘라 쓰지 않고 이전 질문을 유지했다. 이 구현 차이와 모든 원문/오류를 남겼다.
- 균등 8프레임에 이상 프레임이 없는 학습 이상 영상은 35개 중 1개였고, 검증 이상 영상 6개는 모두 이상 프레임을 포함했다. [샘플링 점검](experiments/stage2_2/sampling_label_diagnostic.json)은 기술 통계이며 샘플링 변경이나 질문 입력에 사용하지 않았다.
- 최초 실행은 환경 기록 단계에서 `pip` 모듈 부재로 실패했다. 학습 시작 전 표준 패키지 메타데이터 조회로 수정했으며, 이전 소스·설정·실패 로그는 `startup_failures/001`에 보존했다.
- 186회 시작 시 learner가 같은 줄 끝에 명시적 `Output: 0`을 반환했지만 기존 줄 시작 파서가 거부해 중단됐다. 마지막 명시적 이진 Output을 줄 위치와 무관하게 읽도록 수정하고, 이전 성공 응답 404개의 판정이 모두 같음을 확인한 뒤 185회 완료 지점에서 재개했다. 모델·질문·분할·생성 예산은 변경하지 않았다. [수정·재개 근거](experiments/stage2_2/revisions/001_inline_output_fix/verification.json).
- 218회 시작 시에는 이진 Output 뒤에 결론 문장이 붙어 중단됐다. 유일한 명시적 이진 Output 필드를 읽고 모호한 값은 거부하도록 보강했으며, 이전 성공 판정 485개가 같음을 확인하고 217회 완료 지점에서 재개했다. [두 번째 수정 근거](experiments/stage2_2/revisions/002_scalar_output_field/verification.json).
- 관련 회귀 테스트 **37개 통과**, 첫 파서 보강 **12개 통과**, 두 번째 보강 **18개 통과**, 최종 전체 회귀 **48개 통과**, 실제 모델 파일 22개 SHA256 일치, 모든 반복·배치·응답·검증의 독립 재검산 통과.

| 검증 시점 (업데이트 수) | 정답 / 17 | 검증 정확도 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 11 / 17 | 64.71% | 0 | 0 | 11 | 6 |
| 100 | 11 / 17 | 64.71% | 2 | 2 | 9 | 4 |
| 200 | 11 / 17 | 64.71% | 0 | 0 | 11 | 6 |
| 300 | 11 / 17 | 64.71% | 0 | 0 | 11 | 6 |
| 400 | 11 / 17 | 64.71% | 0 | 0 | 11 | 6 |
| 500 | 11 / 17 | 64.71% | 1 | 1 | 10 | 5 |
| 600 | 11 / 17 | 64.71% | 0 | 0 | 11 | 6 |

TP/FP/TN/FN은 이상 탐지 기준의 참양성/거짓양성/참음성/거짓음성이다. 검증 정상 영상이 11/17이므로 모두 정상으로 판정해도 정확도는 64.71%이며, 이 기준선과 함께 해석한다.

호출별 VLM 시간은 이미지 전처리·GPU 전송 이후의 `model.chat` 구간이다. 전체 실행 시간은 각 command.json에 따로 기록했으며, reserved VRAM은 같은 프로세스의 이전 호출 캐시를 포함한다.

학습 배치 정확도는 반복마다 질문이 바뀌는 online 통계이므로 고정된 모델의 평가 정확도로 해석하지 않는다. 검증은 질문 선택용이며 최종 평가 영상은 학습·검증 입력에서 제외했다.

근거: [고정 설정·배치 순서](experiments/stage2_2/frozen.json), [전체 반복 이력](experiments/stage2_2/iterations.jsonl), [반복별 CSV](experiments/stage2_2/history.csv), [질문 버전 목록](experiments/stage2_2/questions_catalog.json), [learner 원문](experiments/stage2_2/learner_responses.jsonl), [optimizer 원문](experiments/stage2_2/optimizer_responses.jsonl), [검증 결과](experiments/stage2_2/validation), [학습 요약·검산](experiments/stage2_2/training_summary.json), [실행 로그](experiments/stage2_2/execution), [실패 기록](experiments/stage2_2/failures.jsonl).

## 2-3단계 — 질문 선택 (완료)

Q0 및 100·200·300·400·500·600회 업데이트의 7개 후보를 **동일한 검증 17개 영상의 비디오 분류 정확도**로 비교했다. 사전에 고정한 규칙대로 정확도가 가장 높은 후보를 선택했고, 동률은 가장 이른 후보를 유지했다. 최종 평가 프레임 라벨이나 AUROC는 선택에 사용하지 않았다.

선택 결과는 **업데이트 0의 질문**, 검증 정확도 **11/17 = 64.71%**이다. 초기 Q0가 선택되었으므로 optimizer가 생성한 질문이 최종 선택됐다고 표현하지 않는다. 선택 후보의 검증 혼동행렬은 TP=0, FP=0, TN=11, FN=6이다. 검증 표본이 17개로 작아 선택 결과의 불확실성이 크다.

선택한 원문은 다음과 같다.

```text
1. Is there any suspicious person or object that looks unusual in this scene?
2. Is there any behavior that looks unusual in this scene?
```

[선택 이력과 후보](experiments/stage2_3/selection.json), [동결 질문](experiments/stage2_3/questions.txt), [완료·해시](experiments/stage2_3/status.json), [실행 로그](experiments/stage2_3/execution). 질문 SHA256을 고정했다.

## 3단계 — IPAD 재분할 평가 (완료)

동결된 선택 질문을 학습·질문 선택에서 제외한 **37개 IPAD 영상, 16,862프레임**에 적용했다. 추가 데이터셋은 사용하지 않았다. 구간 간격 16프레임, 30 FPS 가정의 10초 창에서 8프레임, ImageBind FP32 검색, 논문 기준 Gaussian smoothing과 위치 가중치는 1단계와 동일하다. ImageBind 특징은 평가 영상 전체에서 다시 계산했다. 프레임 라벨은 VLM·특징 추출이 모두 끝난 뒤 평가기에만 전달했다.

| 점수 단계 | macro AUROC | macro AP | pooled AUROC | pooled AP |
|---|---:|---:|---:|---:|
| 초기 이진 판정 | 49.95 | 20.20 | 49.94 | 19.59 |
| 장면 문맥 검색 | 49.76 | 20.20 | 49.70 | 19.59 |
| Gaussian smoothing | 49.10 | 20.20 | 48.88 | 19.59 |
| 최종 위치 가중치 | 49.10 | 20.20 | 48.88 | 19.59 |

단위는 %. macro는 4개 장면의 단순 평균이며 AP는 average precision이다.

| 장면 | 영상 수 | 프레임 수 | 이상 판정 구간 / 전체 | 최종 AUROC | 최종 AP |
|---|---:|---:|---:|---:|---:|
| R01 | 11 | 2,514 | 0 / 163 | 50.00 | 16.47 |
| R02 | 9 | 5,284 | 0 / 335 | 50.00 | 7.91 |
| R03 | 8 | 5,575 | 1 / 352 | 46.41 | 24.05 |
| R04 | 9 | 3,489 | 0 / 222 | 50.00 | 32.39 |

### 1단계와 동일 프레임 비교

전체 평가 분할이 바뀌었으므로 1단계의 전체 63개 영상 지표와 직접 차이를 계산하지 않는다. 기존 testing 중 새 분할에서도 평가로 남은 **14개 영상·6,252프레임**에서만 1단계와 동일 범위로 다시 비교했다.

| 질문 | 공통 범위 macro AUROC | 공통 범위 macro AP |
|---|---:|---:|
| 1단계 공개 UCF 질문 | 52.83 | 53.39 |
| 이번 선택 질문 | 50.00 | 52.03 |

새 평가 영상에는 기존 정상 training에서 분리한 영상도 포함되어 있다. 기존 testing은 1단계에서 이미 관찰했으며 촬영 세션 단위 분리도 확인할 수 없으므로, 이 결과는 **IPAD 재분할 탐색적 평가**다. 원 논문 UCF/XD 성능이나 표준 IPAD 테스트셋 성능의 재현으로 표현하지 않는다. 모든 추론은 미래·전체 영상 문맥을 사용하는 오프라인 방식이다. VLM 시간은 전처리 이후 `model.chat`, ImageBind 시간은 자체 전처리를 포함한 `encode` 구간이며 모델 로딩·입력 해시 검사·로그 쓰기는 별도다. 전체 벽시계 시간은 실행 command.json에 보존했다.

근거: [고정 질문·설정·소스](experiments/stage3/frozen.json), [실제 모델 파일 해시 검증](experiments/stage3/model_verification.json), [프레임 점수](experiments/stage3/frame_scores.csv), [전체 지표](experiments/stage3/metrics.json), [동일 범위 비교](experiments/stage3/stage1_common_support.json), [원문 설명](experiments/stage3/inference), [후처리](experiments/stage3/postprocessing), [처리시간·VRAM](experiments/stage3/runtime_summary.json), [독립 검산](experiments/stage3/independent_verification.json), [실행 로그](experiments/stage3/execution). 특징 벡터와 이미지·모델은 로컬 보존, SHA256과 프레임 ID만 공개했다.

## 4단계 R01 — 정상 기준 생성·점검 (완료)

해당 장면의 학습 정상 영상만 사용했다. 정렬한 영상 ID를 장면별 seed 0으로 섞고 20%를 올림하여 내부 점검용으로 분리했다. **생성 22개, 점검 6개**이며 다른 장면·학습 이상·검증·평가 영상은 정상 설명 생성에 사용하지 않았다.

기존 16프레임 간격·10초 창·8프레임 입력으로 생성 영상의 **328개 구간**을 관찰했다. 영상별 요약에서 최대 5개 조건부 정상 규칙과 질문을 생성했다. 각 규칙에 인용된 서로 다른 정상 영상 3개 이상의 구간을 다시 시각적으로 확인하고, 내부 정상 점검 영상의 전체 창에서 명확한 반례가 있으면 필수 기준에서 제외했다. 정확한 공정 시간이나 관찰되지 않은 필수 순서는 만들지 않았다.

**후보 5개 중 2개 채택**, 정상 설명 86토큰(상한 1,024)이다. 규칙 검증은 같은 동결 VLM의 판단이며 사람의 독립 정답 주석이 아니다. 영상 인용·프레임 정렬·규칙 채택 조건은 별도 코드로 재검산했다. 모델 가중치 업데이트와 learner–optimizer 질문 반복은 수행하지 않았다. JSON 형식 오류는 원문을 보존하고 최대 2회 형식 복구만 허용했으며 성능을 보고 후보를 다시 선택하지 않았다.

| 규칙 | 생성 근거 영상 수 | 정상 점검 반례 구간 수 | 채택 |
|---|---:|---:|---|
| N1 | 3 | 0 | 예 |
| N2 | 0 | 0 | 아니오 |
| N3 | 3 | 0 | 예 |
| N4 | 0 | 0 | 아니오 |
| N5 | 0 | 0 | 아니오 |

실행 중 영상 요약이 6개 요청을 초과해 구간별 관찰을 나열하면서 중단됐다. 원문·2회 형식 복구 실패·이전 소스를 보존하고, 유효한 중간 요약을 버리지 않도록 항목 수 검사만 해당 영상 구간 수까지 허용했다. 기존 298개 관찰 구간과 19개 성공 요약은 그대로 유지했다. 최종 규칙·설명 토큰·질문 상한, 모델·입력·분할·생성 설정은 바꾸지 않았다. [수정·재개 검증](experiments/stage4/R01/normal/revisions/001_summary_cardinality/verification.json).

후보 생성에서는 요약에 없는 구간 번호를 인용해 중단됐다. 빈 AssertionError 대신 실제 허용된 근거 위치를 형식 복구에 전달하도록 오류 메시지를 보완했다. 인용 허용 조건은 유지했고 정상 관찰·분할·모델·생성 설정은 변경하지 않았다. [인용 검사 수정 기록](experiments/stage4/R01/normal/revisions/002_citation_feedback/verification.json).

인용 형식 복구가 최초 응답으로 되돌아가던 실행기 오류를 수정하여 직전 수정본을 누적 사용하고 모든 잘못된 인용을 함께 알리도록 했다. 회귀 테스트 10개가 통과했다. 328개 관찰·22개 요약은 재추론 없이 보존했다. [복구 로직 수정·검증](experiments/stage4/R01/normal/revisions/003_cumulative_format_repair/verification.json), [실패 원문](experiments/stage4/R01/normal/failures.jsonl).

형식 복구 후에도 남은 잘못된 인용은 후보별로 제외했다. 근거 위치를 임의 대체하지 않고 유효한 후보에만 시각적 근거 검사를 수행했다. 후보 제외 회귀를 포함한 테스트 11개가 통과했다. [후보 제외·재개 기록](experiments/stage4/R01/normal/revisions/004_reject_invalid_candidates/verification.json), [잘못된 인용 후보](experiments/stage4/R01/normal/invalid_candidates.json).

시각적 근거 JSON에서 모델이 프레임 번호를 수백 개 나열하다 잘리는 문제가 발생했다. 실행기가 원본 frame ID를 이미 보존하므로 검사 출력은 규칙 ID·지지/반례/관찰 불가·짧은 근거의 한 줄 형식으로 바꾸고 기존 검사 시도를 별도 보존했다. 파싱 실패는 관찰 불가나 정상으로 치환하지 않으며, 내부 점검 실패가 남은 규칙도 제외했다. 관련 테스트 12개 통과. [검사 형식 변경 기록](experiments/stage4/R01/normal/revisions/005_compact_visual_checks/verification.json).

모델이 설명 없이 명확한 한 단어 판정을 반환하는 경우가 있어, 단일 규칙에 한해 supported/contradicted/unobservable을 그대로 읽도록 보완했다. 설명 누락은 별도로 기록하고 여러 규칙에 대한 한 단어 응답은 전파하지 않는다. 내부 점검은 규칙별로 호출했다. 테스트 13개 통과. [명시적 판정 처리 기록](experiments/stage4/R01/normal/revisions/006_scalar_visual_verdict/verification.json).

정상 규칙 확인 단계의 판정·설명 누락 집계: `{"supported": 186, "contradicted": 0, "unobservable": 6, "failed": 0, "rationale_missing": 108}`. 설명이 없는 명시적 판정에는 실행기가 보존한 원본 프레임 목록과 앞선 관찰을 연결하며, 시각적 이유를 새로 만들어 넣지 않았다.

내부 점검의 `N1: supported`처럼 ID와 판정만 명시된 응답도 설명 누락으로 구분하여 읽었다. 원문은 그대로이며 중복·잘못된 ID·모호한 판정은 실패로 남긴다. 테스트 14개 통과. [ID 포함 명시 판정 처리](experiments/stage4/R01/normal/revisions/007_named_scalar_verdict/verification.json).

채택된 규칙 중 적용 조건이 `when applicable`처럼 추상적인 항목이 있다. 근거 검사를 통과했다는 사실만으로 상세한 공정 단계 정의가 확보됐다고 해석하지 않는다. 이 제한을 유지한 동결 프롬프트의 효과를 기록한다.

실제 생성된 정상 설명:

```text
Normal reference for R01 (normal training observations, not exhaustive process specifications):
N1. When when applicable: the power tool is stationary
N3. When when applicable: the red power tool is stationary
Allowed variation / uncertainty: apply only when the stated condition is visible. Other phases, occlusion and unobserved details are not automatically violations. Exact timing and a mandatory global step order have not been established.
```

실제 생성된 장면별 질문:

```text
1. Is the power tool stationary?
2. Is the red power tool stationary?
```

근거: [고정 설정](experiments/stage4/R01/normal/frozen.json), [입력 목록](experiments/stage4/R01/normal/input_manifest.json), [관찰 원문](experiments/stage4/R01/normal/observations), [후보와 규칙 판정](experiments/stage4/R01/normal/rules.json), [정상 설명·질문](experiments/stage4/R01/normal/normal_profile.json), [모든 호출](experiments/stage4/R01/normal/calls.jsonl), [독립 검산](experiments/stage4/R01/normal/independent_verification.json), [시간·VRAM](experiments/stage4/R01/normal/runtime_summary.json), [실행 로그](experiments/stage4/execution).

## 4단계 R01 — A/B/C 평가 (완료)

**11개 영상·2,514프레임**의 동일 범위로 비교했다. A는 기존 Q0, B는 R01 정상 설명+Q0, C는 동일 정상 설명+R01 전용 질문이다. 설명·질문은 검증 전에 고정했고, 검증 결과에 따른 질문 선택·수정 없이 B/C를 모두 평가했다. A는 3단계 응답·점수를 재사용했으며 모든 점수를 재계산하여 일치를 확인했다. 질문과 무관한 기존 ImageBind 특징도 파일 해시·창·프레임 ID를 검증해 재사용했다.

동결 InternVL2-8B BF16, seed 0, 결정적 생성, 16프레임 간격·30 FPS 가정·10초 창·8프레임 및 논문 기준 후처리는 동일하다. 정상 참조 이미지 추가나 수치 학습은 없다. 프레임 정답은 B/C 최종 추론이 모두 끝난 뒤 평가에만 사용했다.

| 조건 | 최종 AUROC (%) | 최종 AP (%) | 초기 frame 이상 재현율 (%) | 초기 frame 정상 오탐률 (%) | 이상 판정 구간 / 전체 |
|---|---:|---:|---:|---:|---:|
| A | 50.00 | 16.47 | 0.00 | 0.00 | 0 / 163 |
| B | 50.00 | 16.47 | 0.00 | 0.00 | 0 / 163 |
| C | 50.00 | 16.47 | 0.00 | 0.00 | 0 / 163 |

주 비교 B−A AUROC **+0.00%p**, 보조 비교 C−B **+0.00%p**이다.

| 점수 단계 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |
|---|---:|---:|---:|
| initial | 50.00 / 16.47 | 50.00 / 16.47 | 50.00 / 16.47 |
| retrieved | 50.00 / 16.47 | 50.00 / 16.47 | 50.00 / 16.47 |
| smoothed | 50.00 / 16.47 | 50.00 / 16.47 | 50.00 / 16.47 |
| final | 50.00 / 16.47 | 50.00 / 16.47 | 50.00 / 16.47 |

검증 영상의 분류 결과(질문 선택에 사용하지 않음):

| 조건 | 정확도 (%) | TP / FP / TN / FN |
|---|---:|---|
| A | 80.00 | 0 / 0 / 4 / 1 |
| B | 80.00 | 0 / 0 / 4 / 1 |
| C | 80.00 | 0 / 0 / 4 / 1 |

재현율/오탐률은 초기 이진 점수를 원본 프레임에 확장한 값이다. 미세한 차이를 통계적 우월성으로 단정하지 않는다. 기존에 관찰한 IPAD 재분할의 탐색적 오프라인 비교이며, 미래 프레임·전체 영상 문맥을 사용한다. 모델 설명과 정상 규칙은 독립적인 사람 주석이 아니다.

근거: [고정 조건](experiments/stage4/R01/evaluation/frozen.json), [B/C 실제 프롬프트](experiments/stage4/R01/evaluation/prompts.json), [원문 응답](experiments/stage4/R01/evaluation/inference), [전체 프레임 점수](experiments/stage4/R01/evaluation/frame_scores.csv), [전체 지표](experiments/stage4/R01/evaluation/metrics.json), [검증 지표](experiments/stage4/R01/evaluation/validation_metrics.json), [재사용 해시](experiments/stage4/R01/evaluation/reused_sources.json), [독립 검산](experiments/stage4/R01/evaluation/independent_verification.json), [시간·VRAM](experiments/stage4/R01/evaluation/runtime_summary.json), [실행 로그](experiments/stage4/execution).

## 4단계 R02 — 정상 기준 생성·점검 (완료)

해당 장면의 학습 정상 영상만 사용했다. 정렬한 영상 ID를 장면별 seed 0으로 섞고 20%를 올림하여 내부 점검용으로 분리했다. **생성 17개, 점검 5개**이며 다른 장면·학습 이상·검증·평가 영상은 정상 설명 생성에 사용하지 않았다.

기존 16프레임 간격·10초 창·8프레임 입력으로 생성 영상의 **634개 구간**을 관찰했다. 영상별 요약에서 최대 5개 조건부 정상 규칙과 질문을 생성했다. 각 규칙에 인용된 서로 다른 정상 영상 3개 이상의 구간을 다시 시각적으로 확인하고, 내부 정상 점검 영상의 전체 창에서 명확한 반례가 있으면 필수 기준에서 제외했다. 정확한 공정 시간이나 관찰되지 않은 필수 순서는 만들지 않았다.

**후보 5개 중 5개 채택**, 정상 설명 181토큰(상한 1,024)이다. 규칙 검증은 같은 동결 VLM의 판단이며 사람의 독립 정답 주석이 아니다. 영상 인용·프레임 정렬·규칙 채택 조건은 별도 코드로 재검산했다. 모델 가중치 업데이트와 learner–optimizer 질문 반복은 수행하지 않았다. JSON 형식 오류는 원문을 보존하고 최대 2회 형식 복구만 허용했으며 성능을 보고 후보를 다시 선택하지 않았다.

| 규칙 | 생성 근거 영상 수 | 정상 점검 반례 구간 수 | 채택 |
|---|---:|---:|---|
| N1 | 3 | 0 | 예 |
| N2 | 3 | 0 | 예 |
| N3 | 3 | 0 | 예 |
| N4 | 3 | 0 | 예 |
| N5 | 3 | 0 | 예 |

정상 규칙 확인 단계의 판정·설명 누락 집계: `{"supported": 950, "contradicted": 0, "unobservable": 0, "failed": 0, "rationale_missing": 158}`. 설명이 없는 명시적 판정에는 실행기가 보존한 원본 프레임 목록과 앞선 관찰을 연결하며, 시각적 이유를 새로 만들어 넣지 않았다.

채택된 규칙 중 적용 조건이 `when applicable`처럼 추상적인 항목이 있다. 근거 검사를 통과했다는 사실만으로 상세한 공정 단계 정의가 확보됐다고 해석하지 않는다. 이 제한을 유지한 동결 프롬프트의 효과를 기록한다.

요약 형식 복구를 소진한 정상 영상 **2개**는 균등한 최대 6개 관찰 구간의 문장을 그대로 인용하는 고정 대체 절차를 사용했다. 영상 전체 관찰 원문은 보존했고, 이 표현은 모델이 생성한 JSON 요약이 아님을 명시했다. 독립 검사에서 문장·구간 위치가 원문과 정확히 일치함을 확인했다. [대체 처리 기록](experiments/stage4/R02/normal/summary_fallbacks.jsonl).

R02 근거 응답은 단일 규칙에 대한 `supported | 설명` 형태였으나 ID 생략으로 거부됐다. 단일 규칙의 명시 판정·설명을 읽도록 보완하고 기존 원문을 재사용했다. 여러 규칙에 한 응답을 전파하지 않는다. [수정·재개 기록](experiments/stage4/R02/normal/revisions/002_unnamed_single_verdict/verification.json).

실제 생성된 정상 설명:

```text
Normal reference for R02 (normal training observations, not exhaustive process specifications):
N1. When when applicable: The digital device remains stationary on the metal table throughout the video.
N2. When when applicable: The digital device is placed on a metal table in an industrial setting and remains stationary throughout the video.
N3. When when applicable: The digital device is placed on a metal table with a blue and white striped surface.
N4. When when applicable: The digital device remains stationary on the metal table throughout the video.
N5. When when applicable: The digital device is placed on a metal table in an industrial setting with a green clamp and a silver metal bar.
Allowed variation / uncertainty: apply only when the stated condition is visible. Other phases, occlusion and unobserved details are not automatically violations. Exact timing and a mandatory global step order have not been established.
```

실제 생성된 장면별 질문:

```text
1. Is there a visible deviation from the digital device remaining stationary on the metal table throughout the video?
2. Is there a visible deviation from the digital device being placed on a metal table in an industrial setting and remaining stationary throughout the video?
3. Is there a visible deviation from the digital device being placed on a metal table with a blue and white striped surface?
4. Is there a visible deviation from the digital device remaining stationary on the metal table throughout the video?
5. Is there a visible deviation from the digital device being placed on a metal table in an industrial setting with a green clamp and a silver metal bar?
```

근거: [고정 설정](experiments/stage4/R02/normal/frozen.json), [입력 목록](experiments/stage4/R02/normal/input_manifest.json), [관찰 원문](experiments/stage4/R02/normal/observations), [후보와 규칙 판정](experiments/stage4/R02/normal/rules.json), [정상 설명·질문](experiments/stage4/R02/normal/normal_profile.json), [모든 호출](experiments/stage4/R02/normal/calls.jsonl), [독립 검산](experiments/stage4/R02/normal/independent_verification.json), [시간·VRAM](experiments/stage4/R02/normal/runtime_summary.json), [실행 로그](experiments/stage4/execution).

## 4단계 R02 — A/B/C 평가 (완료)

**9개 영상·5,284프레임**의 동일 범위로 비교했다. A는 기존 Q0, B는 R02 정상 설명+Q0, C는 동일 정상 설명+R02 전용 질문이다. 설명·질문은 검증 전에 고정했고, 검증 결과에 따른 질문 선택·수정 없이 B/C를 모두 평가했다. A는 3단계 응답·점수를 재사용했으며 모든 점수를 재계산하여 일치를 확인했다. 질문과 무관한 기존 ImageBind 특징도 파일 해시·창·프레임 ID를 검증해 재사용했다.

동결 InternVL2-8B BF16, seed 0, 결정적 생성, 16프레임 간격·30 FPS 가정·10초 창·8프레임 및 논문 기준 후처리는 동일하다. 정상 참조 이미지 추가나 수치 학습은 없다. 프레임 정답은 B/C 최종 추론이 모두 끝난 뒤 평가에만 사용했다.

| 조건 | 최종 AUROC (%) | 최종 AP (%) | 초기 frame 이상 재현율 (%) | 초기 frame 정상 오탐률 (%) | 이상 판정 구간 / 전체 |
|---|---:|---:|---:|---:|---:|
| A | 50.00 | 7.91 | 0.00 | 0.00 | 0 / 335 |
| B | 47.37 | 7.91 | 0.00 | 0.33 | 1 / 335 |
| C | 50.00 | 7.91 | 0.00 | 0.00 | 0 / 335 |

주 비교 B−A AUROC **-2.63%p**, 보조 비교 C−B **+2.63%p**이다.

| 점수 단계 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |
|---|---:|---:|---:|
| initial | 50.00 / 7.91 | 49.84 / 7.91 | 50.00 / 7.91 |
| retrieved | 50.00 / 7.91 | 49.67 / 7.91 | 50.00 / 7.91 |
| smoothed | 50.00 / 7.91 | 47.37 / 7.91 | 50.00 / 7.91 |
| final | 50.00 / 7.91 | 47.37 / 7.91 | 50.00 / 7.91 |

검증 영상의 분류 결과(질문 선택에 사용하지 않음):

| 조건 | 정확도 (%) | TP / FP / TN / FN |
|---|---:|---|
| A | 75.00 | 0 / 0 / 3 / 1 |
| B | 75.00 | 0 / 0 / 3 / 1 |
| C | 75.00 | 0 / 0 / 3 / 1 |

재현율/오탐률은 초기 이진 점수를 원본 프레임에 확장한 값이다. 미세한 차이를 통계적 우월성으로 단정하지 않는다. 기존에 관찰한 IPAD 재분할의 탐색적 오프라인 비교이며, 미래 프레임·전체 영상 문맥을 사용한다. 모델 설명과 정상 규칙은 독립적인 사람 주석이 아니다.

근거: [고정 조건](experiments/stage4/R02/evaluation/frozen.json), [B/C 실제 프롬프트](experiments/stage4/R02/evaluation/prompts.json), [원문 응답](experiments/stage4/R02/evaluation/inference), [전체 프레임 점수](experiments/stage4/R02/evaluation/frame_scores.csv), [전체 지표](experiments/stage4/R02/evaluation/metrics.json), [검증 지표](experiments/stage4/R02/evaluation/validation_metrics.json), [재사용 해시](experiments/stage4/R02/evaluation/reused_sources.json), [독립 검산](experiments/stage4/R02/evaluation/independent_verification.json), [시간·VRAM](experiments/stage4/R02/evaluation/runtime_summary.json), [실행 로그](experiments/stage4/execution).

## 4단계 R03 — 정상 기준 생성·점검 (완료)

해당 장면의 학습 정상 영상만 사용했다. 정렬한 영상 ID를 장면별 seed 0으로 섞고 20%를 올림하여 내부 점검용으로 분리했다. **생성 13개, 점검 4개**이며 다른 장면·학습 이상·검증·평가 영상은 정상 설명 생성에 사용하지 않았다.

기존 16프레임 간격·10초 창·8프레임 입력으로 생성 영상의 **572개 구간**을 관찰했다. 영상별 요약에서 최대 5개 조건부 정상 규칙과 질문을 생성했다. 각 규칙에 인용된 서로 다른 정상 영상 3개 이상의 구간을 다시 시각적으로 확인하고, 내부 정상 점검 영상의 전체 창에서 명확한 반례가 있으면 필수 기준에서 제외했다. 정확한 공정 시간이나 관찰되지 않은 필수 순서는 만들지 않았다.

**후보 5개 중 1개 채택**, 정상 설명 82토큰(상한 1,024)이다. 규칙 검증은 같은 동결 VLM의 판단이며 사람의 독립 정답 주석이 아니다. 영상 인용·프레임 정렬·규칙 채택 조건은 별도 코드로 재검산했다. 모델 가중치 업데이트와 learner–optimizer 질문 반복은 수행하지 않았다. JSON 형식 오류는 원문을 보존하고 최대 2회 형식 복구만 허용했으며 성능을 보고 후보를 다시 선택하지 않았다.

| 규칙 | 생성 근거 영상 수 | 정상 점검 반례 구간 수 | 채택 |
|---|---:|---:|---|
| N1 | 3 | 0 | 예 |
| N2 | 0 | 0 | 아니오 |
| N3 | 0 | 0 | 아니오 |
| N4 | 0 | 0 | 아니오 |
| N5 | 0 | 0 | 아니오 |

정상 규칙 확인 단계의 판정·설명 누락 집계: `{"supported": 184, "contradicted": 0, "unobservable": 0, "failed": 0, "rationale_missing": 163}`. 설명이 없는 명시적 판정에는 실행기가 보존한 원본 프레임 목록과 앞선 관찰을 연결하며, 시각적 이유를 새로 만들어 넣지 않았다.

채택된 규칙 중 적용 조건이 `when applicable`처럼 추상적인 항목이 있다. 근거 검사를 통과했다는 사실만으로 상세한 공정 단계 정의가 확보됐다고 해석하지 않는다. 이 제한을 유지한 동결 프롬프트의 효과를 기록한다.

요약 형식 복구를 소진한 정상 영상 **2개**는 균등한 최대 6개 관찰 구간의 문장을 그대로 인용하는 고정 대체 절차를 사용했다. 영상 전체 관찰 원문은 보존했고, 이 표현은 모델이 생성한 JSON 요약이 아님을 명시했다. 독립 검사에서 문장·구간 위치가 원문과 정확히 일치함을 확인했다. [대체 처리 기록](experiments/stage4/R03/normal/summary_fallbacks.jsonl).

실제 생성된 정상 설명:

```text
Normal reference for R03 (normal training observations, not exhaustive process specifications):
N1. When when applicable: The yellow forklift remains stationary on the green tabletop throughout the frames.
Allowed variation / uncertainty: apply only when the stated condition is visible. Other phases, occlusion and unobserved details are not automatically violations. Exact timing and a mandatory global step order have not been established.
```

실제 생성된 장면별 질문:

```text
1. Is there a visible deviation from the yellow forklift's position?
```

근거: [고정 설정](experiments/stage4/R03/normal/frozen.json), [입력 목록](experiments/stage4/R03/normal/input_manifest.json), [관찰 원문](experiments/stage4/R03/normal/observations), [후보와 규칙 판정](experiments/stage4/R03/normal/rules.json), [정상 설명·질문](experiments/stage4/R03/normal/normal_profile.json), [모든 호출](experiments/stage4/R03/normal/calls.jsonl), [독립 검산](experiments/stage4/R03/normal/independent_verification.json), [시간·VRAM](experiments/stage4/R03/normal/runtime_summary.json), [실행 로그](experiments/stage4/execution).

## 4단계 R03 — A/B/C 평가 (완료)

**8개 영상·5,575프레임**의 동일 범위로 비교했다. A는 기존 Q0, B는 R03 정상 설명+Q0, C는 동일 정상 설명+R03 전용 질문이다. 설명·질문은 검증 전에 고정했고, 검증 결과에 따른 질문 선택·수정 없이 B/C를 모두 평가했다. A는 3단계 응답·점수를 재사용했으며 모든 점수를 재계산하여 일치를 확인했다. 질문과 무관한 기존 ImageBind 특징도 파일 해시·창·프레임 ID를 검증해 재사용했다.

동결 InternVL2-8B BF16, seed 0, 결정적 생성, 16프레임 간격·30 FPS 가정·10초 창·8프레임 및 논문 기준 후처리는 동일하다. 정상 참조 이미지 추가나 수치 학습은 없다. 프레임 정답은 B/C 최종 추론이 모두 끝난 뒤 평가에만 사용했다.

| 조건 | 최종 AUROC (%) | 최종 AP (%) | 초기 frame 이상 재현율 (%) | 초기 frame 정상 오탐률 (%) | 이상 판정 구간 / 전체 |
|---|---:|---:|---:|---:|---:|
| A | 46.41 | 24.05 | 0.00 | 0.38 | 1 / 352 |
| B | 50.00 | 24.05 | 0.00 | 0.00 | 0 / 352 |
| C | 50.00 | 24.05 | 0.00 | 0.00 | 0 / 352 |

주 비교 B−A AUROC **+3.59%p**, 보조 비교 C−B **+0.00%p**이다.

| 점수 단계 | A AUROC / AP (%) | B AUROC / AP (%) | C AUROC / AP (%) |
|---|---:|---:|---:|
| initial | 49.81 / 24.05 | 50.00 / 24.05 | 50.00 / 24.05 |
| retrieved | 49.06 / 24.05 | 50.00 / 24.05 | 50.00 / 24.05 |
| smoothed | 46.41 / 24.05 | 50.00 / 24.05 | 50.00 / 24.05 |
| final | 46.41 / 24.05 | 50.00 / 24.05 | 50.00 / 24.05 |

검증 영상의 분류 결과(질문 선택에 사용하지 않음):

| 조건 | 정확도 (%) | TP / FP / TN / FN |
|---|---:|---|
| A | 50.00 | 0 / 0 / 2 / 2 |
| B | 50.00 | 0 / 0 / 2 / 2 |
| C | 50.00 | 0 / 0 / 2 / 2 |

재현율/오탐률은 초기 이진 점수를 원본 프레임에 확장한 값이다. 미세한 차이를 통계적 우월성으로 단정하지 않는다. 기존에 관찰한 IPAD 재분할의 탐색적 오프라인 비교이며, 미래 프레임·전체 영상 문맥을 사용한다. 모델 설명과 정상 규칙은 독립적인 사람 주석이 아니다.

근거: [고정 조건](experiments/stage4/R03/evaluation/frozen.json), [B/C 실제 프롬프트](experiments/stage4/R03/evaluation/prompts.json), [원문 응답](experiments/stage4/R03/evaluation/inference), [전체 프레임 점수](experiments/stage4/R03/evaluation/frame_scores.csv), [전체 지표](experiments/stage4/R03/evaluation/metrics.json), [검증 지표](experiments/stage4/R03/evaluation/validation_metrics.json), [재사용 해시](experiments/stage4/R03/evaluation/reused_sources.json), [독립 검산](experiments/stage4/R03/evaluation/independent_verification.json), [시간·VRAM](experiments/stage4/R03/evaluation/runtime_summary.json), [실행 로그](experiments/stage4/execution).

## 4단계 R04 — fact-ID 근거 기반 정상 기준·질문 생성

**현재 공식 R04 실행: 실패 — 서로 다른 3개 정상 영상의 근거 요건 미충족.** 모델·BF16·샘플링·정상 분할·support/audit 채택 기준은 고정했고, 평가 결과에 따른 추가 조정은 수행하지 않았다.

정상 생성 영상 14개·관찰 338개·영상 요약 14개를 검증된 deterministic 입력 캐시로 사용했다. 모델 파일 22개와 원본 정상 프레임 5,300개, 관찰·요약 파일의 SHA256을 확인했다. 관찰·요약 코드 의미, 프롬프트, 샘플링, 분할이 같음을 확인했으며 expensive observation inference는 재실행하지 않았다. 원문 인용 요약 3개도 정확한 원문·중심 프레임을 유지했다. [캐시 검증](experiments/stage4/R04/normal/input_cache_manifest.json).

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

[fact table](experiments/stage4/R04/normal/fact_table.json), [원문 후보](experiments/stage4/R04/normal/candidates.json), [invalid 사유](experiments/stage4/R04/normal/invalid_candidates.json), [중복 필터](experiments/stage4/R04/normal/duplicate_candidates.json), [독립 검산](experiments/stage4/R04/normal/independent_verification.json), [모델 시간·VRAM](experiments/stage4/R04/normal/runtime_summary.json), [실행 명령](experiments/stage4/execution/R04_normal_fact_grounded).

R01–R03의 기존 산출물 5,635개는 작업 전후 SHA256이 모두 동일하고, 해당 장면의 추론·평가를 재실행하지 않았다. [보호 목록](experiments/stage4/r04_protected_artifacts.json). 전체 회귀 테스트는 **82개 통과**했으며 fact-ID·중복·support/audit 실패 처리·합성 평가 전체 경로·평가 라벨 지연 로딩을 포함한다. [테스트 로그](experiments/stage4/execution/R04_repository_regression/20260927T155649Z_479f8739/output.log).

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

완료된 세 장면은 모든 조건의 이상 재현율이 0%였다. 정상 설명은 주로 정지·배치에 머물렀으며, 이 결과를 상세한 공정 정상성 설명 접근 전체의 실패로 일반화하지 않는다. 동일 VLM이 설명 생성·확인을 수행했고, 이미 관찰한 IPAD 재분할에서의 탐색적 오프라인 평가다. [원문 판정 사례](experiments/stage4/explanation_examples.md), [종합 수치·해시](experiments/stage4/summary.json).

공개 결과 검산 명령은 `python scripts/verify_vera_stage4_summary.py`이며 NumPy/scikit-learn이 필요하다. R04 근거 검산은 `python scripts/audit_vera_r04_normal.py`로 수행한다. 모델 재실행 명령과 실제 환경·소스 해시는 실행 로그와 장면별 frozen 기록에 있다.
