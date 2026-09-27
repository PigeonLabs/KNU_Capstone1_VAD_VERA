# VERA 원형 추론의 IPAD 전이 평가

2026-09-27 승인. 질문 학습 전체나 UCF-Crime/XD-Violence 수치 재현이 아닌, 공개 UCF 질문의 IPAD R01–R04 오프라인 전이 평가입니다. 후속 실험은 결과 검토 뒤 결정하며 자동 게시하지 않습니다.

## 사전 고정 설정

- InternVL2-8B BF16, 동결, batch 1, greedy (`do_sample=False`, beams=1, max_new_tokens=1024), seed 0. 모델 revision은 `ipad/vera.py`에 고정합니다. 공식 learner 프롬프트와 UCF 질문 5개를 유지합니다. 오디오는 입력하지 않습니다.
- 기존 Python 3.13 / PyTorch 2.11 CUDA 12.8을 VERA 별도 환경에서 읽기 전용 공유합니다. Transformers 4.51.3, SentencePiece 0.2.1. FlashAttention 미설치 환경에서는 같은 BF16 모델의 명시적 eager attention을 사용합니다. 이는 공식 실행 경로와 다르므로 비트 단위 재현을 주장하지 않습니다.
- 중심 간격 16프레임. 가정 30 FPS에서 중심 전후 150프레임을 경계 내로 자르고 8프레임을 endpoint-exclusive 균등 추출합니다. 공식 RGB/PIL resize 및 ImageNet 정규화를 유지합니다.
- ImageBind huge FP32, 동결. 논문이 참조하는 LAVAD의 공개 frame-list 영상 변환을 사용합니다. 문맥 창에서 endpoint-inclusive 10프레임을 추출한 뒤 원 변환의 5 temporal clips × 3 spatial crops 및 집계 방식을 유지합니다. VLM의 8프레임 샘플러와 구분합니다. ImageBind/LAVAD 소스 SHA를 고정하고 기본 FPS=30의 프레임 목록 변환을 그대로 보존합니다. 오디오/텍스트 특징은 사용하지 않습니다. 학습 데이터는 정상 사전 점검 이외 사용하지 않습니다.
- 동일 영상 안에서 L2 정규화 특징의 내적으로 cosine similarity를 계산합니다. 자기 자신 포함 top-K, K=max(1,floor(0.1h)), softmax(sim/10). 동점은 원 구간 순서로 처리합니다.
- 대칭 Gaussian kernel 15, sigma1=10, 합 1 정규화, zero padding. 짧은 구간열에서도 출력 길이를 보존합니다. 16프레임씩 확장하되 마지막 부분 구간도 포함합니다.
- 위치 가중치는 논문의 1-based i, c=floor(F/2), sigma2=floor(F/2)를 사용합니다. F=1일 때만 sigma2=1로 정의합니다. 중간 점수 반올림과 별도 min-max 정규화는 하지 않습니다.

## 논문과 코드 차이

| 항목 | 논문 v3 | 공개 코드 | 본 실험 |
|---|---|---|---|
| UCF 검색 비율 | 0.1h | 0.15h | 0.1h |
| 온도 | Eq.4 sim/tau, B.4 tau=10 | sim*10 | sim/10; 부록 sensitivity 표의 다른 tau 값으로 튜닝하지 않음 |
| Gaussian 위치 | 중심으로부터 거리 p | linspace와 음수 floor division으로 비대칭 가능 | 대칭 -7..7 |
| 중간 점수 | 반올림 명시 없음 | 소수점 한 자리 | 반올림 없음 |
| 마지막 구간 | floor(F/d), 나머지 처리 불명확 | range(0,F,16) | 마지막 부분 구간 포함, 실제 frame ID 정렬 |
| 시각 토큰 연결 | 8개 프레임을 구분하는 입력 | VERA batch_chat 경로는 첫 placeholder에 8개 patch 묶음을 삽입 | 모델의 chat API에 num_patches_list=[1]*8로 모든 placeholder를 프레임별 연결 |
| 잘못된 출력 | 별도 규칙 미기재 | '0' 미포함 응답을 1로 처리 | 원문/실패를 기록하고 중단 |

## 데이터·평가

- 원본 영상·프레임 ID를 유지합니다. R02 12/13/14는 기존 strict 정렬 제외를 유지합니다. 새 정렬 불일치는 자동 보정하지 않고 실패로 처리합니다.
- 영상/프레임 SHA256 manifest를 고정합니다. 추론기는 라벨 파일을 읽지 않습니다. 평가기는 전체 추론 완료 후 라벨을 읽어 AUROC와 AP(sklearn average_precision_score)를 계산합니다.
- 초기/검색/smoothing/최종 단계의 장면별, scene macro, pooled 지표 및 모든 정렬된 프레임 점수를 저장합니다. IPAD와 DINOv2의 기존 점수는 같은 프레임 교집합에서 다시 평가합니다.
- 공개 질문에 대한 학습은 없으며 학습 이력은 해당 없음입니다. seed 반복 또는 테스트 성능으로 설정 선택을 하지 않습니다. 생성 설명은 검증된 사실 주석이 아닙니다.
- R01은 15개 중 14개 영상이 300프레임 미만입니다. 고정된 10초 창과 위치 가중치가 산업 공정의 짧은 영상에서 미치는 영향을 결과 해석에 표시합니다.

## 실행·보존

```bash
PYTHONPATH=. HF_HOME="$PWD/cache/vera/hf" cache/vera/venv/bin/python scripts/run_vera.py --phase all
```

구간별 응답과 설명을 즉시 fsync하고 동일 설정 재시작 시 완료 구간을 재사용합니다. 소스/설정 fingerprint가 다르면 기존 실행을 덮어쓰지 않습니다. 실패는 events.jsonl 및 상태 파일에 기록합니다. 특징 캐시는 cache/vera/feature_journals에만 보존합니다. 모델·원 데이터·환경은 게시하지 않습니다.

실험 상태는 running/diagnostic/failed/paused_low_disk/complete로 구분합니다. 디스크 여유 10 GiB 이하에서 중단하며 latch 파일이 있으면 자동 재개하지 않습니다. 사용자 GPU 작업을 종료하지 않습니다. 원형은 미래 프레임·영상 전체 문맥을 사용하므로 처리 속도를 실시간 인과 추론 성능으로 해석하지 않습니다.

초기 호환성 실패(SentencePiece 0.2.2의 null token 거부)는 preflight.log에 보존했으며 0.2.1로 수정했습니다. 기존 환경은 변경하지 않았습니다.

## 추가 구현 가정

LAVAD의 FrameVideo는 이미 샘플된 10프레임 목록도 기본 30 FPS로 해석합니다. 따라서 2초 temporal clip 5개가 같은 짧은 목록을 볼 수 있고 UniformTemporalSubsample(2)는 양 끝 프레임을 고릅니다. 이 공개 변환을 임의로 바꾸지 않았으며 8프레임 VLM 입력과 구별합니다. 이미지 특징 선택/전처리의 대안은 후속 실험으로만 취급합니다.
