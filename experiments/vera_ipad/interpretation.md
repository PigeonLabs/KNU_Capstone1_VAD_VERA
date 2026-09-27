# VERA IPAD 결과 해석

공개 질문 기반 추론 구현과 전체 평가는 완료했지만, 최종 macro AUROC는 52.02%, pooled AUROC는 52.88%로 낮았습니다.
정상 학습 영상 4개에서 실행 가능성을 점검한 뒤, 고정 설정으로 63개 테스트 영상·31,550프레임을 평가했습니다. R02 12·13·14는 기존 정렬 규약에 따라 제외했습니다.

## 초기 판정의 문제

| 장면 | 구간 수 | 이상 판정 구간 | 모든 구간이 정상 판정인 영상 |
|---|---:|---:|---:|
| R01 | 238 | 0 | 15/15 |
| R02 | 487 | 0 | 12/12 |
| R03 | 758 | 6 | 14/17 |
| R04 | 518 | 1 | 18/19 |

전체 2,001개 구간 중 7개만 이상으로 판정했습니다. R01·R02는 초기 점수가 모두 0이어서 검색·평활화·위치 가중치를 적용해도 AUROC가 50%입니다.
초기 macro AUROC 50.21%에서 후처리 후 52.02%가 되었습니다. 현재 결과에서 우선 확인할 문제는 후처리보다 초기 시각 판정의 거의 전부 정상이라는 응답입니다.
UCF에서 학습한 질문이 산업 공정에 맞지 않을 가능성, 8프레임·10초 문맥의 시간 정보 손실, VLM의 미세한 공정 동작 인식 한계는 가능한 설명입니다. 이번 실험만으로 원인을 확정하지는 않습니다.

## 동일 프레임 교집합 비교

아래 모든 방법은 공통 30,353프레임에서 재계산했습니다. 전체 프레임 VERA 지표와 구분합니다. 기존 seed 0 결과를 사용했으며 가장 높은 결과를 골라 설정을 바꾸지 않았습니다.

| 방법 | Macro AUROC (%) | Macro AP (%) |
|---|---:|---:|
| VERA | 52.023 | 44.964 |
| IPAD: negative_psnr_with_phase | 72.526 | 66.003 |
| DINOv2_reconstruction: dino_patch6_with_phase | 69.977 | 62.705 |
| DINOv2_reconstruction: dino_patch12_with_phase | 72.943 | 65.847 |
| DINOv2_reconstruction: dino_cls_with_phase | 75.194 | 68.262 |
| DINOv2_reconstruction: dino_multilevel_with_phase | 72.609 | 65.604 |
| DINOv2_prototype: conditional_nn_with_phase | 75.828 | 69.031 |
| DINOv2_prototype: conditional_soft_with_phase | 74.613 | 67.929 |
| DINOv2_prototype: unconditional_nn_with_phase | 77.483 | 67.433 |
| DINOv2_prototype: unconditional_soft_with_phase | 71.398 | 61.083 |

## 실행 및 재현 범위

VLM 평균 1.867초/구간, ImageBind 평균 0.189초/구간입니다. 최대 할당 GPU 메모리는 각각 18.62GiB, 4.79GiB입니다.
시간은 전처리와 모델 계산을 포함한 구간 처리 시간이며 모델 로딩·manifest 검사·로그 쓰기 등 전체 실행 시간과 다릅니다. 오프라인 미래 문맥을 사용하므로 실시간 인과 추론 성능으로 해석하지 않습니다.
정합성·디스크 보호 테스트 23개와 실제 정상 영상 GPU 진단을 통과했습니다. tokenizer 초기 호환성 실패 및 수정 이력은 preflight 로그에 보존했습니다.
원 논문 학습 전체나 원 벤치마크 점수 재현은 아닙니다. BF16 eager attention, 다중 이미지 토큰 연결, 논문/공식 코드 차이, ImageBind 전처리 가정은 ../../docs/vera_ipad_protocol.md에 명시했습니다.
공개 질문을 임의로 수정하거나 테스트 지표로 설정을 선택하지 않았습니다. 후속 실험은 이번 결과를 검토한 뒤 범위를 정합니다.

근거: metrics.json, comparisons.json, prototype_comparison.json, prediction_diagnostics.json, runtime_summary.json, positive_explanations.json, frame_scores.csv.
