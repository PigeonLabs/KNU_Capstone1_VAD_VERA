# VERA의 IPAD 전이 평가

**1단계: VERA 논문 추론 방법론 재현 완료.** 동결된 InternVL2-8B와 저자가 공개한 UCF-Crime 학습 질문을 IPAD R01–R04에 적용했다. 최종 macro AUROC는 **52.02%**, macro AP는 **43.79%**였다. 초기 이진 판정에서 2,001개 구간 중 **7개만 이상**으로 판정했다.

이 저장소는 VERA 실험의 코드·분석 자료·실행 로그를 관리한다. 1단계는 공개 질문 전이 평가이고 2-2단계의 질문 학습은 별도 기록했다. 원 논문 벤치마크 수치 재현을 의미하지 않는다. 미래 프레임과 전체 영상 문맥을 사용하는 **오프라인 평가**다. 원본 이미지·모델 가중치·특징 캐시는 로컬에 보존하고, 공개 자료에는 SHA256 목록을 포함한다.

## 실험 단계

| 단계 | 목적 | 상태 | 확인된 결과 / 다음 결정 |
|---|---|---|---|
| **1** | **논문 추론 방법론 재현 및 IPAD 전이 평가** | **완료** | 63개 영상, 31,550프레임, 2,001구간; macro AUROC 52.02% |
| **2-1** | **정상·이상 영상을 포함하는 IPAD 재분할** | **완료** | 학습 120개 / 검증 17개 / 평가 37개, 영상·동일 프레임 교집합 0 |
| **2-2** | **learner–optimizer 반복 질문 최적화** | **완료** | 10 epoch·600 반복, 유효 optimizer 530회, 형식 오류 70회 |

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
