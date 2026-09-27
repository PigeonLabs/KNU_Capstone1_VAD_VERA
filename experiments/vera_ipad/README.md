# VERA의 IPAD 전이 평가

2026-09-27 승인된 공개 질문 기반 오프라인 추론 실험입니다. 학습 전체 재현 또는 UCF-Crime 논문 수치 재현이 아닙니다. 전체 평가를 완료했습니다. 63개 영상·31,550프레임, 최종 macro AUROC 52.02%, pooled AUROC 52.88%입니다. 상세 해석은 `interpretation.md`, 단계별 지표는 `results.md`를 참고하세요.

- 규약: ../../docs/vera_ipad_protocol.md
- 설정/질문/소스 해시: frozen.json
- 모델 SHA256: model_inventory.json
- 전체 실행 기록: run.log, events.jsonl (실패와 재시작 포함)
- 정상 영상 진단: diagnostic/, diagnostic_complete.json
- 실행 완료 후 지표: metrics.json, comparisons.json, results.md
- 원본 프레임 정렬 결과: frame_scores.csv
- 모델·특징·격리 환경: ../../cache/vera/ (로컬 전용)

```bash
# 기존 IPAD .venv를 보존하는 별도 환경과 공식 소스
.venv/bin/python scripts/bootstrap_vera.py --install
# 약 21GB 모델 파일, 최소 40GiB 여유 요구
cache/vera/venv/bin/python scripts/download_vera_models.py
# 테스트
cache/vera/venv/bin/python -m pytest -q tests/test_vera.py tests/test_disk_guard.py
# 정상 학습 영상 진단 -> 전체 테스트 추론 -> 평가
PYTHONPATH=. HF_HOME="$PWD/cache/vera/hf" cache/vera/venv/bin/python scripts/run_vera.py --phase all
cache/vera/venv/bin/python scripts/analyze_vera.py
```

명령은 프로젝트 루트에서 실행합니다. 재시작은 동일 명령이며 동일 fingerprint의 완료 구간만 재사용합니다. 디스크 일시중단은 사용자 지시 없이 재개하지 않습니다. 자동 GitHub 게시를 하지 않습니다.

초기 SentencePiece 0.2.2 토크나이저 실패는 preflight.log에, 0.2.1 수정 뒤 성공한 GPU 점검은 preflight_02.log에 보존했습니다. 순수 BF16 eager attention을 사용하며 공식 flash attention 실행 경로와의 비트 단위 동일성을 주장하지 않습니다.
