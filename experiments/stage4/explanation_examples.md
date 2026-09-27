# 4단계 실제 판정과 미탐 예시

R01–R04 평가 완료 후 고른 사후 해석 자료다. 정답은 사례 선택에만 사용했고 프롬프트 변경이나 조건 선택에 사용하지 않았다. 원문 응답은 모델의 설명이며, 물체 이름·상태의 사실성에 대한 사람 검증은 아니다. 전체 선택 규칙, 프레임 정답, 원본 응답 경로와 SHA256은 [JSON](explanation_examples.json)에 보존했다.

## R01

### 세 조건 모두 놓친 이상 구간

영상 `R01/testing/06`, 중심 프레임 160, 점수 부여 구간 [160, 176), 이상 프레임 16/16. 입력 프레임: [10, 37, 65, 93, 121, 149, 177, 205]. 입력 프레임 정답: [1, 1, 1, 1, 1, 1, 1, 1]. 프레임 ID는 0부터 시작한다.

**A: 정상 판정**

> Answers to Prompt Questions: The video shows a workbench with various tools and objects, but there is no person or unusual behavior visible. The scene appears to be a typical workshop environment.
> Output: 0

**B: 정상 판정**

> Answers to Prompt Questions: The video shows a stationary power tool on a workbench, which is consistent with the normal reference for R01. There is no suspicious person or object, and no unusual behavior is observed. The power tool is stationary, and there is no indication of an anomaly.
> Output: 0

**C: 정상 판정**

> Answers to Prompt Questions: The power tool is stationary, and the red power tool is stationary.
> Output: 0

## R02

### 조건별 판정이 다른 첫 구간

영상 `R02/training/28`, 중심 프레임 192, 점수 부여 구간 [192, 208), 이상 프레임 0/16. 입력 프레임: [42, 79, 117, 154, 192, 229, 267, 304]. 입력 프레임 정답: [0, 0, 0, 0, 0, 0, 0, 0]. 프레임 ID는 0부터 시작한다.

**A: 정상 판정**

> Answers to Prompt Questions: The video shows a digital device on a table in a workshop, and a person touches the device. There is no suspicious person or object, and no unusual behavior is observed.
> Output: 0

**B: 이상 판정**

> Answers to Prompt Questions: In the video, there is no suspicious person or object that looks unusual. The only behavior that looks unusual is when a hand touches the digital device on the table. Based on this observation, there is an anomaly in the video.
> Output: 1

**C: 정상 판정**

> Answers to Prompt Questions: 
> 1. No, there is no visible deviation from the digital device remaining stationary on the metal table throughout the video.
> 2. No, there is no visible deviation from the digital device being placed on a metal table in an industrial setting and remaining stationary throughout the video.
> 3. No, there is no visible deviation from the digital device being placed on a metal table with a blue and white striped surface.
> 4. No, there is no visible deviation from the digital device remaining stationary on the metal table throughout the video.
> 5. No, there is no visible deviation from the digital device being placed on a metal table in an industrial setting with a green clamp and a silver metal bar.
> 
> Output: 0

### 세 조건 모두 놓친 이상 구간

영상 `R02/testing/05`, 중심 프레임 176, 점수 부여 구간 [176, 192), 이상 프레임 16/16. 입력 프레임: [26, 63, 101, 138, 176, 213, 251, 288]. 입력 프레임 정답: [1, 1, 1, 1, 1, 1, 1, 1]. 프레임 ID는 0부터 시작한다.

**A: 정상 판정**

> Answers to Prompt Questions: There is no suspicious person or object in the scene, and there is no unusual behavior. The video appears to be a static shot of a device on a table.
> Output: 0

**B: 정상 판정**

> Answers to Prompt Questions: There is no suspicious person or object that looks unusual in this scene. There is no behavior that looks unusual in this scene.
> Output: 0

**C: 정상 판정**

> Answers to Prompt Questions: 
> 1. No, there is no visible deviation from the digital device remaining stationary on the metal table throughout the video.
> 2. No, there is no visible deviation from the digital device being placed on a metal table in an industrial setting and remaining stationary throughout the video.
> 3. No, there is no visible deviation from the digital device being placed on a metal table with a blue and white striped surface.
> 4. No, there is no visible deviation from the digital device remaining stationary on the metal table throughout the video.
> 5. No, there is no visible deviation from the digital device being placed on a metal table in an industrial setting with a green clamp and a silver metal bar.
> 
> Output: 0

## R03

### 조건별 판정이 다른 첫 구간

영상 `R03/training/02`, 중심 프레임 480, 점수 부여 구간 [480, 496), 이상 프레임 0/16. 입력 프레임: [330, 367, 405, 442, 480, 517, 555, 592]. 입력 프레임 정답: [0, 0, 0, 0, 0, 0, 0, 0]. 프레임 ID는 0부터 시작한다.

**A: 이상 판정**

> Answers to Prompt Questions: The video shows a forklift with four white cups on its forks, which is an unusual behavior for a forklift to be carrying such items. There is no person or object that looks suspicious in the scene.
> Output: 1

**B: 정상 판정**

> Answers to Prompt Questions: The yellow forklift remains stationary on the green tabletop throughout the frames. There are no suspicious persons or objects, and no unusual behavior is observed.
> Output: 0

**C: 정상 판정**

> Answers to Prompt Questions: The yellow forklift remains stationary on the green tabletop throughout the frames, and there is no visible deviation from its position.
> Output: 0

### 세 조건 모두 놓친 이상 구간

영상 `R03/testing/01`, 중심 프레임 0, 점수 부여 구간 [0, 16), 이상 프레임 16/16. 입력 프레임: [0, 18, 37, 56, 75, 93, 112, 131]. 입력 프레임 정답: [1, 1, 1, 1, 1, 1, 1, 1]. 프레임 ID는 0부터 시작한다.

**A: 정상 판정**

> Answers to Prompt Questions: The video shows a forklift and a stack of plates on a green mat. There is no suspicious person or object, and no unusual behavior is observed.
> Output: 0

**B: 정상 판정**

> Answers to Prompt Questions: The scene shows a yellow forklift and a stack of plates on a green tabletop. There are no suspicious persons or objects, and no unusual behavior is observed.
> Output: 0

**C: 정상 판정**

> Answers to Prompt Questions: The yellow forklift remains stationary on the green tabletop throughout the frames, and there are no visible deviations from the normal reference.
> Output: 0

## R04

### 조건별 판정이 다른 첫 구간

영상 `R04/testing/15`, 중심 프레임 32, 점수 부여 구간 [32, 48), 이상 프레임 0/16. 입력 프레임: [0, 22, 45, 68, 91, 113, 136, 159]. 입력 프레임 정답: [0, 0, 0, 1, 1, 1, 1, 1]. 프레임 ID는 0부터 시작한다.

**A: 정상 판정**

> Answers to Prompt Questions: There is no suspicious person or object in the scene, and there is no unusual behavior. The video shows a machine cutting a piece of wood, which is a normal activity in a workshop.
> Output: 0

**B: 이상 판정**

> Answers to Prompt Questions: There is no suspicious person or object in the scene. The behavior of the metal blade cutting the paper is normal. However, the presence of the paper being cut by the metal blade is unusual in this context.
> Output: 1

**C: 정상 판정**

> Answers to Prompt Questions: The video shows a metal blade on a metal tabletop with a green surface. There is no visible deviation from the expected pattern.
> Output: 0

### 세 조건 모두 놓친 이상 구간

영상 `R04/testing/02`, 중심 프레임 224, 점수 부여 구간 [224, 240), 이상 프레임 16/16. 입력 프레임: [74, 111, 149, 186, 224, 261, 299, 336]. 입력 프레임 정답: [1, 1, 1, 1, 1, 1, 1, 1]. 프레임 ID는 0부터 시작한다.

**A: 정상 판정**

> Answers to Prompt Questions: The video shows a person using a machine to cut a piece of wood. There is no suspicious person or object, and the behavior appears normal. The person is handling the wood and the machine in a standard manner.
> Output: 0

**B: 정상 판정**

> Answers to Prompt Questions: The person is using a saw to cut a piece of wood, which is a normal activity in a workshop. There is no suspicious person or object, and the behavior appears to be normal.
> Output: 0

**C: 정상 판정**

> Answers to Prompt Questions: The video shows a person using a saw to cut a piece of wood on a workbench. The process appears to be normal and there are no visible deviations from the expected pattern.
> Output: 0
