# 상세 기록 및 단계별 main 게시 규약

2026-09-27 사용자가 각 단계 종료 후 commit/main push를 명시적으로 승인했다. 후속 실험의 실행 승인과 결과 게시 승인은 구별한다.

## 실험 단계별 필수 기록

| 시점 | 기록 |
|---|---|
| 시작 전 | 목적·가설·승인 상태·주/보조 지표·허용 데이터·제외 영상·분할·seed·고정 설정·명령·모델/소스/환경 버전 및 SHA256 |
| 실행 중 | 명령 시작/종료 UTC·KST, stdout/stderr, exit code, 모든 오류/중단/재시작, 입력 frame ID, 샘플링 인덱스, raw 응답/설명, 구간/프레임 점수, 단계별 처리시간·VRAM |
| 학습 시 | epoch/iteration, 학습 손실 또는 질문 업데이트, validation 결과, 최종 선택 근거 및 선택 시점. 미학습 단계는 해당 없음 |
| 완료 시 | 프레임 정렬/유한값/누락/중복/공통 support 검증, 모든 고정 비교군 결과, 실패 결과 포함, 한국어 README, SHA256 inventory |
| 게시 시 | stage와 상태가 드러나는 commit 메시지, main push 결과, 원격 HEAD SHA 일치 확인. 원격 경합 시 force push 금지 |

새 명령은 프로젝트 루트에서 다음과 같이 실행한다. 출력은 별도 attempt 디렉터리에 저장하여 이전 실행을 덮어쓰지 않는다.

```bash
python scripts/logged_command.py --stage stage2 --step diagnostic -- python scripts/run_stage2.py
# 위 명령의 stage2 실행기는 아직 제안 단계이며 실제로 구현/실행되지 않았다.
```

현재까지 제공되지 않았던 과거 command/stdout를 나중에 생성해 실제 원본 로그처럼 표현하지 않는다. 1단계는 기존 raw 로그·events·manifest·응답·score 파일을 원형대로 보존하고, experiments/stage1_publication/stage1_ledger.json에서 사후 정리임을 표시한다. 기존 로그에는 모든 설치 명령의 완전한 terminal transcript와 최초 실행 pytest stdout 원문이 없으며, 당시 기록한 검증 요약과 이번 게시 검증의 실제 stdout을 구별한다.

## 게시 절차

1. 완료/실패/진단 상태와 README를 갱신하고 필요한 분석을 검증한다.
2. 데이터/모델/환경은 로컬에 남기고 텍스트 분석·SHA256만 Git에 포함한다. 비밀값을 명령 인자나 로그에 넣지 않는다.
3. `python scripts/publish_vera_stage.py --stage stage1 --message "1단계: VERA 추론 재현과 IPAD 전체 평가 기록"`을 실행한다.
4. 스크립트는 목적지·main·게시 파일 정책을 점검하고 현재 허용 파일을 커밋한다. fetch 후 원격이 앞서 있으면 중단하여 무관한 작업을 보존한다. push에는 force 옵션을 사용하지 않는다.
5. 게시 영수증은 로컬 runs/publication에 남기고 사용자에게 원격 commit URL을 보고한다. 자기 자신이 속한 commit SHA를 파일에 넣으려는 순환 기록은 만들지 않는다.

정상/실패/진단 상태를 혼동하지 않는다. 디스크 pause는 상태를 보존하고 사용자 지시 없이 재개하지 않는다. 반복 예약 작업은 만들지 않는다.
