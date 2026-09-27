# 공개 자료와 로컬 보존 자료

현재 단계의 **모든 저장된 분석 결과**를 게시한다. 원본 이미지/영상, 가중치, 특징 벡터 캐시 및 가상환경을 GitHub에 올리지 않는 기존 사용자 지침을 유지한다.

| 자료 | 저장 위치 / 공개 범위 |
|---|---|
| 소스·고정 질문·설정·검증 코드 | ipad/, scripts/, tests/, docs/ |
| 모든 초기 구간 판정·설명·프레임 ID·시간·VRAM | experiments/vera_ipad/inference/ 및 diagnostic/ |
| 전체 프레임 정답·4단계 점수 | experiments/vera_ipad/frame_scores.csv |
| 검색 이웃/가중치·평활화 결과·위치 가중치 | experiments/vera_ipad/postprocessing/ |
| 실행/다운로드/초기 실패 로그·환경 | experiments/vera_ipad/*.log, events.jsonl, environment.txt, gpu.txt |
| 원본 정상/테스트 프레임 | 로컬 IPAD_dataset; 정상/테스트 manifest의 frame SHA256만 공개 |
| 모델 가중치 | 로컬 cache/vera; model_inventory.json의 경로·크기·SHA256만 공개 |
| ImageBind 특징 캐시 | 로컬 cache/vera/feature_journals; inference_inventory.json 및 stage1_publication/local_feature_inventory.json만 공개 |
| 정상 진단의 특징 처리시간·입력 ID·차원 | stage1_publication/diagnostic_feature_summary.json; 특징 벡터 제외 |
| 기존 IPAD/DINOv2 비교 근거 | stage1_reproduction 및 stage2_dinov2의 scores.csv/metrics.json만 복사. 이 폴더는 이전 프로젝트의 기준선 이름이며 VERA 단계 번호가 아님 |
| 원 논문 PDF, 로컬 문서, 환경, 자격증명 | 게시하지 않음. 원 논문은 공식 arXiv 링크 제공 |

experiments/vera_ipad의 1단계 원형 파일을 그대로 보존했다. 내부 README/규약의 ‘자동 게시하지 않음’은 **실행 당시 정책**이고, 현재는 루트 AGENTS.md와 docs/logging_and_publication.md의 2026-09-27 게시 승인이 우선한다. 원형 artifact_inventory.json은 로컬 전체를 기준으로 하므로 제외된 Python bytecode 경로가 포함될 수 있다. 현재 공개된 실제 파일 목록은 docs/publication_manifest.json이다.

새 공개 저장소의 검증에는 원본 이미지·모델 가중치가 필요 없다. scripts/verify_publication.py는 고정 소스, 원형 분석 파일, frame coverage, 전체/장면별/macro 지표 및 공통 프레임 비교를 검산한다. 특징 벡터를 제외했으므로 ImageBind embedding 자체의 재계산은 원본 데이터와 모델을 갖춘 환경에서만 가능하다.
