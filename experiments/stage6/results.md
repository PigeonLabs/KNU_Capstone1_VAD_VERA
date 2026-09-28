# 6단계 실제 실행 결과 — 실패 종료

정상 TRAIN 이미지 점검과 GPT 6 Pro·Claude Opus 5.5 High의 실제 상호 검토를 거쳐 실행했다. 협업 기록은 `consultation/`에 있으며 발췌임을 명시했다. 모델 업데이트는 0회다.

- C0: 5단계 P1의 장면 물체 이름만 중립적인 관찰 표현으로 수정.
- N: C0 + 같은 장면 정상 TRAIN의 독립 이미지 4장.
- X: N과 동일한 문구·이미지 수 + 다음 장면의 정상 이미지 4장.
- query: 기존 8프레임, 300프레임 문맥, 중심 간격 16. 참조 선택·프롬프트·후처리·통계 계획을 추론 전에 고정했다.
- InternVL2-8B BF16/eager, greedy, seed 0, 최대 1,024 새 토큰. 기존 모델·데이터·결과 보존.

| 항목 | 실제 결과 |
|---|---|
| 정상 사전 점검 | 60/60 유효 |
| 평가 호출 | 78/3,216 실행 |
| 평가 파싱 | 유효 77 / 실패 1 |
| 실패 위치 | R01/training/27, center=64, C0 |
| 실패 입력 frame ID | 0, 26, 53, 80, 107, 133, 160, 187 |
| 원인 | 설명 반복, Output 필드 누락 |
| 응답 재토큰화 길이 | 1,024; 실제 생성 토큰 수·종료 사유 미노출 |
| N/X 평가 | 미실행 |
| 평가 라벨·AUROC/AP | 접근/계산하지 않음 |
| 전체 회귀 테스트 | 106 passed |
| 기존 Stage4/5 보호 | 10,284개 파일 SHA256 불변 |

판정 실패 시 중단한다는 사전 규칙을 적용했다. 임의 정상 대체·출력 수선·재시도·일부 유효 창만의 성능 보고를 하지 않았다. 이 실행은 정상 참조 이미지의 성능 향상 또는 악화에 관한 결론을 제공하지 않는다. 미실행 성공 경로용 분석 코드는 구현되어 있지만 그 분석 결과는 없다.

실행 명령(저장된 실패 캐시를 덮어쓰거나 자동 재개하지 않는다):

```bash
python scripts/logged_command.py --stage stage6 --step prepare -- python scripts/prepare_vera_stage6.py
python scripts/logged_command.py --stage stage6 --step preflight -- python scripts/run_vera_stage6.py --phase preflight
python scripts/logged_command.py --stage stage6 --step evaluation -- python scripts/run_vera_stage6.py --phase evaluation
python scripts/logged_command.py --stage stage6 --step regression_final -- python -m pytest -q
python scripts/logged_command.py --stage stage6 --step failure_verification -- python scripts/verify_vera_stage6_failure.py
```

검증 근거: [실패 상태](status.json), [실패 응답](parsing_failures.jsonl), [모든 호출](calls.jsonl), [사전 점검](preflight_summary.json), [고정 프로토콜](frozen.json), [독립 검증](failure_verification.json), [실행 로그](execution/).
