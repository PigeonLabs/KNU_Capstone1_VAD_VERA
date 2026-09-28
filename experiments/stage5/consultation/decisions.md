# 실제 모델 협업과 채택 결정

실행 전 사용자가 지정한 열린 탭에서 ChatGPT **6 Pro**(모델 선택 화면)와 Claude **Opus 5.5 High**(모델 메뉴)를 확인하고 설계를 요청했다. 이는 웹 UI 표기이며 제공자의 내부 모델 빌드 ID를 검증했다는 뜻은 아니다. 두 모델은 설계 자문만 했고 실험 추론은 기존 로컬 InternVL2-8B BF16이 담당한다. 아래는 실제 응답의 발췌와 작업자의 결정 요약이며 전체 대화 원문으로 표현하지 않는다.

- GPT 대화: https://chatgpt.com/c/6ab9e50c-86e8-83e8-a617-4b42df385966
- Claude 대화: https://claude.ai/chat/7ce4b602-fe43-4e89-becb-7d6de1d0b7d6
- 제공한 맥락: 공개 Stage4 R04 요약 지표와 구조적 실패, 고정 모델/샘플링/분할, 정상 학습 요약에서 확인한 장면 정보. 추가 evaluation 영상·이미지·새 결과를 제공하지 않았다. 이미 알려진 평가 결과를 근거로 후속 방법을 설계하므로 탐색적 평가임을 명시했다.
- 브라우저 기본 연결 실패 후 사용 가능한 대체 UI 제어로 실제 요청/응답을 확인했다. 이미지·모델 파일·비밀값은 전송하지 않았다.

## GPT 6 Pro 실제 조언 발췌

> “초기 판별력과 후처리 후 순위 개선을 분리”해야 합니다.

> Use independently observable process context to establish applicability; do not require the expected normal state itself to be present before checking whether that expectation is violated.

> An ordinary-looking background or familiar object does not cancel a supported discrepancy elsewhere.

> 추가 GPU 추론 없이 all-zero와 all-one 초기 점수도 동일 후처리에 통과시키세요.

GPT는 산업적 직접 질문, 균형 증거 비교, 엄격한 조건 게이트 대조군의 3조건을 제안했다. 모델이 보지 못한 중심 프레임을 설명하게 하지 말고 실제 입력 8장과 점수 구간의 라벨을 따로 진단하라고 했다. 영상 단위 paired bootstrap 2,000회, undefined 재표집 횟수, 이미 관측한 평가의 한계, malformed 출력 무대체를 권했다. 실행자가 검사한 코드의 실제 순서 initial→retrieved→smoothed→final과 일치했다.

## Claude Opus 5.5 High 실제 조언 발췌

> Matching the scene layout does not by itself mean the clip is normal, and a detail missing from the reference does not by itself mean the clip is anomalous.

> P2−P1이 측정하는 것은 "정상이라는 프레이밍, 추가 서술 내용, 길이"가 섞인 효과입니다.

> If the summary does not mention motion, do not infer that motion is abnormal.

Claude의 첫 제안에는 새 동작 규칙 및 Jaccard lint, 로짓 점수 등이 있었으나, 학습 요약에 stationary/motion 및 재료 명칭 충돌이 있음을 제공한 후 재검토했다. 새 필수 동작 규칙을 추측하지 않는 작은 중첩 실험, 중립 물체 명칭, 짧은 정상·이상 증거 필드를 구체화했다. 첫 답변의 후처리 순서 오류는 코드 근거로 정정해서 전달했고, 최종 실험은 기존 코드 순서를 그대로 사용한다.

## 최종 채택

1. P1: 장면별 정상 학습 근거의 중립적 물체 설명 + 산업 공정 관찰 질문.
2. P2: P1에 Stage4 정상 설명 원문과 불완전성·조건 동어반복 방지 지침 추가.
3. P3: P2의 정보는 그대로 두고 양쪽 시각 근거를 명시적으로 비교. GPT가 제안한 공통 이진 결정 기준을 P1/P2에도 동일하게 둔다.
4. 새 규칙/결함 목록/정확한 타이밍/숨은 기능 생성, 로짓 추출, Jaccard 기준, 새로운 샘플링, 학습·양자화는 도입하지 않았다. 엄격한 조건 게이트 대조군 대신 P1/P2의 정상 설명 추가 효과를 비교하도록 선택했다. 이 선택은 평가 전에 했다.
5. 장면별 내용은 정상 학습 요약과 frozen 설명에서만 취했다. 서로 다른 물체 명칭은 중립 상위어로 정리했고, 정확한 공정 기능·동작을 새로 보장하지 않는다. 사례 파일은 `training_context_examples.json`, 원문 설명·해시는 `../references.json`, 실제 프롬프트는 `../prompts.json`에 보존했다.
6. 20개 정상 학습 창 × 3조건으로 파싱/실행 시간만 확인한다. 상수 출력이라는 이유로 수정하지 않는다. 이후 37평가영상의 모든 조건을 실행하며, 새로운 evaluation labels는 전체 추론 후에만 연다. 평가 결과를 보고 다시 생성하거나 조정하지 않는다.
7. 완전한 공통 평가, 초기 recall/FPR, 네 후처리 단계, 입력/점수 구간 정렬, all-zero/all-one 진단, 영상 단위 신뢰구간을 보고한다. 보조 모델의 조언 자체는 성능 증거가 아니다.

이 문서는 실제 협업의 기록이다. 채택하지 않은 제안은 향후 수행할 실험 목록이나 완료 결과로 등록하지 않는다.
