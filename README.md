# VERA의 IPAD 전이 평가

**6단계 실패 종료: 정상 참조 이미지 비교 중 78번째 평가 응답이 반복되어 판정을 반환하지 못했습니다. 성능 지표는 계산하지 않았습니다.** [실행 결과](experiments/stage6/results.md).

**5단계 완료: GPT 6 Pro·Claude Opus 5.5 High 설계 협업 후 R01–R04의 고정 프롬프트 3종을 평가했습니다.** [실제 결과·한계](experiments/stage5/results.md).

**4단계 완료: R01–R03 기존 결과 보존, R04 규칙 생성·영상별 grounding과 A/B/C 평가 완료.** [현재 공식 결과](experiments/stage4/results.md).

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
| **4-R04-N** | **R04 규칙 생성·영상별 근거 검증** | **완료** | 후보 5개 / 중복 4개 / 채택 1개, grounding 12/14영상 |
| **4-R04-E** | **R04 정상 설명·질문 A/B/C 평가** | **완료** | AUROC A 50.00 / B 54.31 / C 50.00% |
| **5** | **R01–R04 산업 질문·정상 설명·근거 비교** | **완료** | 고정 3조건 / 37영상 / 16,862프레임, [결과](experiments/stage5/results.md) |
| **6** | **중립 장면 문구·정상 참조 이미지 비교** | **실패** | 정상 점검 60/60, 평가 77개 유효 후 1개 출력 실패로 즉시 중단; 지표 없음 |

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

A/C는 모든 구간을 정상으로 판정했다. B의 유일한 양성 구간은 R04/testing/15의 중심 32, 점수 구간 [32,48)로 프레임 정답 기준 FP 16개·TP 0개다. 다만 8개 입력 프레임 중 5개에는 이상 라벨이 있어 이를 정상 영상만 보고 낸 오탐으로 단정하지 않는다. B의 최종 순위 지표 상승은 smoothing/위치 가중치 이후 나타났고, 초기 프레임 단위 이상 recall은 0%였다. 이 결과만으로 직접적인 이상 검출 능력이 개선되었다고 결론 내리지 않는다. 상세 입력·점수 정렬과 원문은 [실제 판정 사례](experiments/stage4/explanation_examples.md)에 보존했다.

전체 테스트 **88개 통과**. R01–R03 산출물 **5,635개 SHA256 불변**, 추론·평가 재실행 없음. [테스트 로그](experiments/stage4/execution/R04_separated_regression_final/20260927T161810Z_a3762195/output.log), [보호 목록](experiments/stage4/r04_protected_artifacts.json), [normal 독립 검산](experiments/stage4/R04/normal/independent_verification.json), [평가 독립 검산](experiments/stage4/R04/evaluation/independent_verification.json).

[후보 원문](experiments/stage4/R04/normal/candidates.json), [고정 규칙과 해시](experiments/stage4/R04/normal/candidate_rules_frozen.json), [14개 grounding 원문/결과](experiments/stage4/R04/normal/grounding_summary.json), [fact table](experiments/stage4/R04/normal/fact_table.json), [정상 설명](experiments/stage4/R04/normal/normal_description.txt), [질문](experiments/stage4/R04/normal/questions.txt), [평가 프롬프트](experiments/stage4/R04/evaluation/prompts.json), [정렬된 단계별 프레임 점수](experiments/stage4/R04/evaluation/frame_scores.csv), [실행 명령](experiments/stage4/execution/R04_separated_normal).

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

이미 관찰한 IPAD 재분할의 탐색적 오프라인 평가다. 장면별 실행 프로토콜에는 차이가 있어 동일한 정상 기준 생성 방식의 네 장면 반복으로 해석하지 않는다. [종합 수치·해시](experiments/stage4/summary.json), [실제 판정 사례](experiments/stage4/explanation_examples.md). 공개 점수 재검산: `python scripts/verify_vera_stage4_summary.py`.

## 5단계 — 산업 공정 질문·정상 설명·근거 비교 (완료)

사용자 요청에 따라 **GPT 6 Pro와 Claude Opus 5.5 High에 실제 설계 검토를 요청**하고, 기존 로컬 InternVL2-8B BF16으로 R01–R04 프롬프트 실험을 수행했다. 두 모델은 자문만 담당했고 성능 추론에 대신 사용하지 않았다. 모델 UI 표기를 확인했으며 내부 제공자 빌드 ID는 확인하지 않았다. [협업 발췌·채택 근거](experiments/stage5/consultation/decisions.md).

### 수행 방법

| 조건 | 실제 입력·처리 |
|---|---|
| A/B/C | 기존 Stage4 프레임 점수와 해시를 그대로 재사용. 재추론 없음 |
| P1 | 장면별 중립 물체 설명 + 산업 공정 질문 + 직접 판단 |
| P2 | P1 + 기존 정상 설명 원문 + 불완전성·동어반복 방지 지침 |
| P3 | P2의 동일 정보 + 정상/이상 시각 근거를 대칭적으로 비교 |

R01은 공구·녹색/검정 장치, R02는 디지털 장치·클램프·금속 바, R03은 노란 지게차형 물체·받침·흰 원통, R04는 칼날·클램프·금속 바·갈색 재료에 대한 **정상 학습 근거만** 장면별로 사용했다. 명칭은 모델 관찰에서 유래했으며 실제 물체 기능을 독립 확인한 것은 아니다. 새 정상 동작 규칙이나 결함 목록을 추측하여 추가하지 않았다. [정상 근거·해시](experiments/stage5/references.json), [장면별 프롬프트 12개](experiments/stage5/prompts.json).

모든 조건에 동일한 이진 결정 기준을 두었다. 구체적으로 보이는 공정 불일치가 있으면 1, 지지되는 불일치가 없으면 0이다. 익숙한 물체나 배경만으로 정상이라고 단정하지 않고, 관찰되지 않는 동작·순서·정확한 시간을 추론하지 않도록 했다. P3의 필드는 `Observed changes / Evidence consistent / Evidence conflicting / Uncertainty / Output`이다. **주 비교는 P3−P2의 장면 macro 초기 AUROC**이며, 이진 초기 AUROC는 balanced accuracy와 같다. P2−P1과 P3−A는 보조 비교다.

정상 학습 20개 창 × 3조건 사전 점검 **60회** 후 고정된 37영상·16,862프레임·조건당 1,072창을 평가했다. 새 평가 추론 **3,216회**, 파싱 실패 0회. validation/evaluation 성능 기반 선택·질문 최적화·재시도·학습 업데이트는 하지 않았다. **모든 장면·조건 추론 완료 후에만 원본 프레임 정답 파일을 열었다.** 기존 분할 manifest의 영상 단위 메타데이터는 읽지만, 추론 입력 manifest와 프롬프트에는 정답 필드를 전달하지 않는다. stride16, clipped10초(30FPS 가정), 8프레임, BF16 eager, seed0, greedy 최대1024토큰, ImageBind→smoothing→위치 가중치를 유지했다. [사전 고정 프로토콜](experiments/stage5/protocol.json), [소스·입력 fingerprint](experiments/stage5/frozen.json).

### 확인된 결과 해석

R04의 P1은 최종 AUROC **69.43%**, AP **59.05%**로 기존 A/B보다 높은 점추정값을 보였다. 초기 검출률은 16.99%, 오탐률은 9.37%이고 초기 AUROC는 53.81%다. 최종 AUROC의 영상 bootstrap 구간은 [48.17, 90.76]%로 넓으므로 안정적인 개선이 입증되었다고 표현하지 않는다.

장면 전체로는 P1의 최종 macro AUROC가 50.67%에 머물렀다. R02 P1은 37.99%로 낮아졌고, R01 P3는 TP 0 / FP 871프레임이었다. **정상 설명 추가(P2)와 근거 비교(P3)가 공통적으로 성능을 높인다는 가설은 지지되지 않았다.**

실제 응답에서는 정적인 배치만으로 정상을 판단하는 경향이 남았다. R01/testing/09 중심0의 P3는 같은 응답에서 공구가 움직였다고 하면서 정상 근거에는 정지했다고 적었고, 움직임 자체를 이상 근거로 사용했다. R02/testing/02 중심208은 입력 8장과 점수 구간 모두 정상 라벨인데도 손의 기기 접촉을 이상으로 판정했다. 이는 출력 형식을 준수해도 근거의 일관성과 정상 동작 해석이 보장되지 않는 사례다. 원문 설명의 시각적 사실 여부를 사람이 재판정한 것은 아니며, 이 사후 사례를 추가 튜닝에 사용하지 않았다. [실제 응답 비교](experiments/stage5/explanation_examples.md).

### 실제 최종 점수 AUROC / AP (%)

| 장면 | A | B | C | P1 | P2 | P3 |
|---|---:|---:|---:|---:|---:|---:|
| R01 | 50.00 / 16.47 | 50.00 / 16.47 | 50.00 / 16.47 | 50.00 / 16.47 | 50.00 / 16.47 | 8.93 / 16.47 |
| R02 | 50.00 / 7.91 | 47.37 / 7.91 | 50.00 / 7.91 | 37.99 / 7.84 | 46.06 / 6.89 | 46.31 / 6.83 |
| R03 | 46.41 / 24.05 | 50.00 / 24.05 | 50.00 / 24.05 | 45.25 / 19.97 | 50.00 / 24.05 | 50.00 / 24.05 |
| R04 | 50.00 / 32.39 | 54.31 / 38.73 | 50.00 / 32.39 | 69.43 / 59.05 | 54.95 / 41.13 | 60.51 / 37.33 |
| macro | 49.10 / 20.20 | 50.42 / 21.79 | 50.00 / 20.20 | 50.67 / 25.83 | 50.25 / 22.14 | 41.44 / 21.17 |
| pooled | 48.88 / 19.59 | 50.73 / 22.04 | 50.00 / 19.59 | 54.07 / 19.72 | 44.26 / 18.30 | 37.23 / 16.21 |

### 초기 프레임 판정

| 장면 | 조건 | TP | FP | FN | TN | Recall (%) | FPR (%) | 초기 AUROC / AP (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R01 | A | 0 | 0 | 414 | 2100 | 0.00 | 0.00 | 50.00 / 16.47 |
| R01 | B | 0 | 0 | 414 | 2100 | 0.00 | 0.00 | 50.00 / 16.47 |
| R01 | C | 0 | 0 | 414 | 2100 | 0.00 | 0.00 | 50.00 / 16.47 |
| R01 | P1 | 0 | 0 | 414 | 2100 | 0.00 | 0.00 | 50.00 / 16.47 |
| R01 | P2 | 0 | 0 | 414 | 2100 | 0.00 | 0.00 | 50.00 / 16.47 |
| R01 | P3 | 0 | 871 | 414 | 1229 | 0.00 | 41.48 | 29.26 / 16.47 |
| R02 | A | 0 | 0 | 418 | 4866 | 0.00 | 0.00 | 50.00 / 7.91 |
| R02 | B | 0 | 16 | 418 | 4850 | 0.00 | 0.33 | 49.84 / 7.91 |
| R02 | C | 0 | 0 | 418 | 4866 | 0.00 | 0.00 | 50.00 / 7.91 |
| R02 | P1 | 32 | 448 | 386 | 4418 | 7.66 | 9.21 | 49.22 / 7.82 |
| R02 | P2 | 16 | 256 | 402 | 4610 | 3.83 | 5.26 | 49.28 / 7.83 |
| R02 | P3 | 75 | 997 | 343 | 3869 | 17.94 | 20.49 | 48.73 / 7.75 |
| R03 | A | 0 | 16 | 1341 | 4218 | 0.00 | 0.38 | 49.81 / 24.05 |
| R03 | B | 0 | 0 | 1341 | 4234 | 0.00 | 0.00 | 50.00 / 24.05 |
| R03 | C | 0 | 0 | 1341 | 4234 | 0.00 | 0.00 | 50.00 / 24.05 |
| R03 | P1 | 101 | 571 | 1240 | 3663 | 7.53 | 13.49 | 47.02 / 23.37 |
| R03 | P2 | 0 | 0 | 1341 | 4234 | 0.00 | 0.00 | 50.00 / 24.05 |
| R03 | P3 | 0 | 0 | 1341 | 4234 | 0.00 | 0.00 | 50.00 / 24.05 |
| R04 | A | 0 | 0 | 1130 | 2359 | 0.00 | 0.00 | 50.00 / 32.39 |
| R04 | B | 0 | 16 | 1130 | 2343 | 0.00 | 0.68 | 49.66 / 32.39 |
| R04 | C | 0 | 0 | 1130 | 2359 | 0.00 | 0.00 | 50.00 / 32.39 |
| R04 | P1 | 192 | 221 | 938 | 2138 | 16.99 | 9.37 | 53.81 / 34.78 |
| R04 | P2 | 0 | 32 | 1130 | 2327 | 0.00 | 1.36 | 49.32 / 32.39 |
| R04 | P3 | 47 | 245 | 1083 | 2114 | 4.16 | 10.39 | 46.89 / 31.71 |
| pooled | A | 0 | 16 | 3303 | 13543 | 0.00 | 0.12 | 49.94 / 19.59 |
| pooled | B | 0 | 32 | 3303 | 13527 | 0.00 | 0.24 | 49.88 / 19.59 |
| pooled | C | 0 | 0 | 3303 | 13559 | 0.00 | 0.00 | 50.00 / 19.59 |
| pooled | P1 | 325 | 1240 | 2978 | 12319 | 9.84 | 9.15 | 50.35 / 19.70 |
| pooled | P2 | 16 | 288 | 3287 | 13271 | 0.48 | 2.12 | 49.18 / 19.52 |
| pooled | P3 | 122 | 2113 | 3181 | 11446 | 3.69 | 15.58 | 44.05 / 19.07 |

### 네 점수 단계 (장면 macro, AUROC / AP %)

| 조건 | initial | retrieved | smoothed | final |
|---|---:|---:|---:|---:|
| A | 49.95 / 20.20 | 49.76 / 20.20 | 49.10 / 20.20 | 49.10 / 20.20 |
| B | 49.87 / 20.20 | 49.83 / 20.18 | 50.38 / 20.74 | 50.42 / 21.79 |
| C | 50.00 / 20.20 | 50.00 / 20.20 | 50.00 / 20.20 | 50.00 / 20.20 |
| P1 | 50.01 / 20.61 | 50.26 / 20.98 | 49.86 / 25.86 | 50.67 / 25.83 |
| P2 | 49.65 / 20.19 | 48.98 / 20.18 | 49.77 / 21.34 | 50.25 / 22.14 |
| P3 | 43.72 / 19.99 | 43.50 / 19.84 | 40.24 / 20.43 | 41.44 / 21.17 |

### 영상 단위 불확실성

장면 내 영상을 복원추출하고 모든 방법에 같은 추출을 적용한 **2,000회 paired bootstrap, seed0**이다. 각 영상의 모든 프레임을 함께 가져왔으며 프레임 단위 bootstrap은 하지 않았다. 단일 클래스 표본은 undefined로 남겼다. 아래는 장면 macro AUROC의 차이와 95% percentile 구간(percentage point)이다.

| 비교 | 점수 | 차이 | 95% 구간 | 유효 / undefined |
|---|---|---:|---|---:|
| P3-P2 | initial | -5.93 | [-8.73, -2.28] | 1566 / 434 |
| P3-P2 | final | -8.82 | [-16.51, 0.69] | 1566 / 434 |
| P2-P1 | initial | -0.36 | [-4.83, 2.24] | 1566 / 434 |
| P2-P1 | final | -0.41 | [-18.72, 11.88] | 1566 / 434 |
| P3-A | initial | -6.23 | [-8.95, -3.05] | 1566 / 434 |
| P3-A | final | -7.66 | [-13.99, 1.74] | 1566 / 434 |

사전 지정한 P3−P2의 초기 macro AUROC 차이 구간은 0보다 낮았다. 이 비교에서 근거 구조화가 개선되었다고 볼 수 없다. [전체 장면·AP·단계별 구간](experiments/stage5/bootstrap.json), [실제 영상 재표집 목록](experiments/stage5/bootstrap_resamples.json).

### 시간 정렬과 상수 대조군

| 장면 | 입력 정상 / 점수 정상 | 입력 정상 / 점수 이상 | 입력 이상 / 점수 정상 | 입력 이상 / 점수 이상 |
|---|---:|---:|---:|---:|
| R01 | 133 | 0 | 2 | 28 |
| R02 | 281 | 0 | 26 | 28 |
| R03 | 250 | 0 | 16 | 86 |
| R04 | 128 | 0 | 21 | 73 |
| pooled | 792 | 0 | 65 | 215 |

입력 이상은 샘플 8장 중 하나라도 이상 라벨인 경우, 점수 이상은 중심부터 최대16프레임 구간에 이상 라벨이 있는 경우다. 공식 프레임 지표를 바꾸는 재라벨링이 아니라 추가 진단이다. [0 / 1–4 / 5–8개 입력 이상 프레임별 판정·recall·FPR](experiments/stage5/alignment_diagnostics.json).

| 계산 대조군 | macro 초기 AUROC / AP (%) | macro 최종 AUROC / AP (%) |
|---|---:|---:|
| ZERO | 50.00 / 20.20 | 50.00 / 20.20 |
| ONE | 50.00 / 20.20 | 58.95 / 24.90 |

ZERO/ONE은 새 모델 추론이 아니라 모든 창을 0/1로 놓은 계산 대조군이다. ONE의 비상수 최종 점수는 경계 smoothing·위치 가중치가 만드는 순위를 보여준다. 반올림 없는 부동소수점 계산은 retrieved 단계의 수치상 동점에도 미세한 차이를 만들 수 있다. 최종 AUROC 상승만으로 시각 검출 능력 향상을 주장하지 않는다.

### 검증·재현·한계

- 전체 테스트 **97개 통과**, 독립 점수 재구성·라벨 정렬·해시 검증 통과. [테스트 로그](experiments/stage5/execution/regression_final/20260928T051013Z_72de03af/output.log), [독립 검증](experiments/stage5/independent_verification.json), [통계 검증](experiments/stage5/analysis_verification.json).
- Stage4 전체 **6,837개 파일 SHA256 불변**. R01–R04 기존 추론·평가 재실행 없음. [보호 목록](experiments/stage5/protected_stage4.json).
- 모델 호출 합계 3,276회, model.chat 합산 120.5분, peak allocated 19.55 GiB. 시간은 전처리·로딩·검산을 제외한다.
- 요청한 출력 필드가 모두 나타난 응답: P1 1072/1072, P2 1072/1072, P3 1072/1072. 이는 표현 형식 진단이며 판정값을 바꾸는 기준은 아니다.
- 이미 여러 차례 관찰한 IPAD 재분할 평가이며 독립적인 새 테스트가 아니다. 영상 수가 적고 같은 촬영 세션의 상관이 남을 수 있다.
- P1 대 기존 방법은 질문 의미·문장·형식이 함께 다르다. P2−P1은 정상 설명·해석 지침·길이를 함께 추가한다. P3−P2도 구조와 필드별 길이 배분을 바꾸므로 완전한 단일 요인 분해라고 주장하지 않는다.
- 정상 요약은 같은 VLM이 만든 불완전한 관찰이며 정적인 물체 설명에 치우쳤다. 새 정상 공정 지식을 획득했다고 표현하지 않는다.
- 잘못된 출력은 정상·이상으로 대체하지 않는다. 토큰 수는 디코딩 텍스트를 다시 토큰화한 길이이며 제공자가 노출하지 않은 정확한 종료 사유를 주장하지 않는다.

[실제 판정 사례](experiments/stage5/explanation_examples.md), [원문 응답](experiments/stage5/inference), [프레임별 6조건 점수](experiments/stage5/frame_scores.csv), [모든 지표](experiments/stage5/metrics.json), [상수 대조군](experiments/stage5/constant_control_metrics.json), [실행 명령·stdout](experiments/stage5/execution).

재현(기존 데이터·모델·특징 캐시 필요):

```bash
python scripts/run_vera_stage5.py --phase preflight
python scripts/run_vera_stage5.py --phase evaluation
python scripts/audit_vera_stage5.py
python scripts/analyze_vera_stage5.py
```

## 6단계 — 정상 참조 이미지 비교의 실제 실행

GPT 6 Pro와 Claude Opus 5.5 High의 상호 검토에서 C0(중립 장면 문구), N(동일 장면 정상 참조 4장), X(다른 장면 정상 참조 4장)에 합의한 뒤 실행했다. query 8장과 기존 후처리는 고정했다. 모델 학습·질문 최적화는 수행하지 않았다.

정상 TRAIN 사전 점검은 **60/60 유효**였다. 그러나 평가 호출 78번째(R01/training/27, center 64, C0)에서 설명이 반복되고 응답 재토큰화 길이가 1,024가 되었으며 `Output`이 없었다. **77개 유효 응답과 1개 실패 원문을 보존하고 즉시 중단**했다. N/X 평가·라벨 접근·AUROC/AP 계산은 수행하지 않았으므로 참조 이미지의 성능 효과를 판단할 수 없다. 실제 생성 토큰 수와 종료 사유는 이 실행기에서 노출되지 않았다.

[실제 결과와 재실행 명령](experiments/stage6/results.md) · [사전 합의](experiments/stage6/consultation/consensus.json) · [고정 설정](experiments/stage6/frozen.json) · [실패 독립 검증](experiments/stage6/failure_verification.json) · [실행 로그](experiments/stage6/execution)
