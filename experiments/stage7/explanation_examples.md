# 7단계 실제 응답 비교

전체 추론과 지표가 고정된 뒤 각 조건에 해당하는 첫 (video ID, center) 사례를 골랐다. 원문 설명은 모델의 주장이며 사람이 검증한 시각적 사실이 아니다. 사례는 프롬프트·참조 선택에 피드백하지 않았다.

## R01

- C0는 음성, N은 양성인 이상 포함 점수 구간: 해당 사례 없음.

- 입력 8장과 점수 구간 모두 정상인데 N이 양성인 구간: 해당 사례 없음.

- N과 X의 판정이 다른 구간: 해당 사례 없음.

### 새 세 조건 모두 음성인 이상 포함 점수 구간

`R01/testing/06`, center=0, 점수 [0,16), 이상 점수 프레임 6/16, 이상 입력 7/8.

**P1 / Output 0** ([원문](../stage5/inference/P1/R01/testing/06/000000.json))

> Observations: The red power tool remains stationary on the tabletop throughout the frames. The green and black device, along with the blue object, are consistently positioned near the power tool. The lighting and object positions do not show significant changes.
> 
> Assessment: The red power tool's stationary position and the consistent placement of other objects suggest a coherent industrial work process. There are no visible discrepancies in object state, relative position, or tool-workpiece relationships.
> 
> Output: 0

**C0 / Output 0** ([원문](inference/C0/R01/testing/06/000000.json))

> Output: 0
> Evidence: The objects and their positions remain consistent across the images, indicating a coherent industrial work process.

**N / Output 0** ([원문](inference/N/R01/testing/06/000000.json))

참조: `R01/testing/14:28`, `R01/training/10:88`, `R01/training/20:142`, `R01/training/31:203`

> Output: 0
> Evidence: The objects and their positions appear consistent across the frames, with no visible changes or disruptions in the industrial work process.

**X / Output 0** ([원문](inference/X/R01/testing/06/000000.json))

참조: `R02/training/03:68`, `R02/training/13:221`, `R02/training/18:361`, `R02/training/27:514`

> Output: 0
> Evidence: The objects and their positions appear consistent across the images, with no visible changes or disruptions in the industrial work process.

## R02

- C0는 음성, N은 양성인 이상 포함 점수 구간: 해당 사례 없음.

- 입력 8장과 점수 구간 모두 정상인데 N이 양성인 구간: 해당 사례 없음.

- N과 X의 판정이 다른 구간: 해당 사례 없음.

### 새 세 조건 모두 음성인 이상 포함 점수 구간

`R02/testing/05`, center=0, 점수 [0,16), 이상 점수 프레임 5/16, 이상 입력 7/8.

**P1 / Output 0** ([원문](../stage5/inference/P1/R02/testing/05/000000.json))

> Observations: The digital device is placed on the tabletop in all frames. The green clamp and metal bar are consistently positioned in the background. The lighting remains consistent throughout the frames.
> 
> Assessment: There are no visible discrepancies in the object state, relative position, tool-workpiece relationships, or changes across the supplied images. The scene appears coherent and consistent with an industrial work process.
> 
> Output: 0

**C0 / Output 0** ([원문](inference/C0/R02/testing/05/000000.json))

> Output: 0
> Evidence: The dial device is consistently placed on the white platform, and the surrounding objects and their positions remain unchanged across the images.

**N / Output 0** ([원문](inference/N/R02/testing/05/000000.json))

참조: `R02/training/03:68`, `R02/training/13:221`, `R02/training/18:361`, `R02/training/27:514`

> Output: 0
> Evidence: The device remains stationary on the platform throughout the frames, and the surrounding objects and their positions appear consistent with a coherent industrial work process.

**X / Output 0** ([원문](inference/X/R02/testing/05/000000.json))

참조: `R03/training/03:90`, `R03/training/08:265`, `R03/training/12:465`, `R03/training/17:598`

> Output: 0
> Evidence: The images show a consistent setup with a white platform, crossed supports, and a dial device, indicating a coherent industrial work process.

## R03

- C0는 음성, N은 양성인 이상 포함 점수 구간: 해당 사례 없음.

- 입력 8장과 점수 구간 모두 정상인데 N이 양성인 구간: 해당 사례 없음.

- N과 X의 판정이 다른 구간: 해당 사례 없음.

### 새 세 조건 모두 음성인 이상 포함 점수 구간

`R03/testing/01`, center=0, 점수 [0,16), 이상 점수 프레임 16/16, 이상 입력 8/8.

**P1 / Output 0** ([원문](../stage5/inference/P1/R03/testing/01/000000.json))

> Observations: The yellow forklift-like object remains stationary in the same position across all frames. The black and white support with white cylindrical objects is also stationary and undisturbed. The green tabletop with grid patterns shows no signs of movement or change.
> 
> Assessment: There are no visible discrepancies in the object state, relative position, or tool-workpiece relationships. The scene appears consistent with a coherent industrial work process.
> 
> Output: 0

**C0 / Output 0** ([원문](inference/C0/R03/testing/01/000000.json))

> Output: 0
> Evidence: The forklift and the stack of white cylindrical objects remain in the same positions and state across all frames, indicating a coherent industrial work process.

**N / Output 0** ([원문](inference/N/R03/testing/01/000000.json))

참조: `R03/training/03:90`, `R03/training/08:265`, `R03/training/12:465`, `R03/training/17:598`

> Output: 0
> Evidence: The forklift and the stack of white cylindrical objects remain in consistent positions and states across all frames, indicating a coherent industrial work process.

**X / Output 0** ([원문](inference/X/R03/testing/01/000000.json))

참조: `R04/training/04:50`, `R04/training/09:156`, `R04/training/14:201`, `R04/training/22:331`

> Output: 0
> Evidence: The yellow-and-black forklift-like object and the white cylindrical objects remain in consistent positions and states across all frames, indicating a coherent industrial work process.

## R04

- C0는 음성, N은 양성인 이상 포함 점수 구간: 해당 사례 없음.

- 입력 8장과 점수 구간 모두 정상인데 N이 양성인 구간: 해당 사례 없음.

- N과 X의 판정이 다른 구간: 해당 사례 없음.

### 새 세 조건 모두 음성인 이상 포함 점수 구간

`R04/testing/02`, center=64, 점수 [64,80), 이상 점수 프레임 13/16, 이상 입력 5/8.

**P1 / Output 0** ([원문](../stage5/inference/P1/R04/testing/02/000064.json))

> Observations: The brown material is being fed into the blade in a consistent manner across all frames. The clamp is holding the metal bar steady, and the tool-workpiece relationship remains unchanged. The lighting and object positions are consistent throughout the images.
> 
> Assessment: There are no visible discrepancies in the object state, relative position, tool-workpiece relationships, or changes across the supplied images. The process appears to be coherent and normal.
> 
> Output: 0

**C0 / Output 0** ([원문](inference/C0/R04/testing/02/000064.json))

> Output: 0
> Evidence: The images show a consistent industrial work process with the blade-like piece, brown sheet material, and green and metal components in their expected positions and states. There are no visible discrepancies in the tool-workpiece relationships or changes across the images.

**N / Output 0** ([원문](inference/N/R04/testing/02/000064.json))

참조: `R04/training/04:50`, `R04/training/09:156`, `R04/training/14:201`, `R04/training/22:331`

> Output: 0
> Evidence: The images show a consistent industrial work process with a blade-like piece cutting a brown sheet material. The tool-workpiece relationships and object states remain coherent across the images.

**X / Output 0** ([원문](inference/X/R04/testing/02/000064.json))

참조: `R01/testing/14:28`, `R01/training/10:88`, `R01/training/20:142`, `R01/training/31:203`

> Output: 0
> Evidence: The images show a consistent industrial work process with a blade-like piece, brown sheet material, and green and metal components. There are no visible discrepancies in object state, relative position, tool-workpiece relationships, or changes across the supplied images.

