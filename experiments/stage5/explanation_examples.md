# 5단계 실제 판정 사례

모든 추론·지표가 고정된 뒤 결정적으로 선택한 사례다. 프롬프트 수정이나 선택에 사용하지 않았다. 점수 구간 라벨과 실제 입력 프레임 라벨은 다를 수 있다. 출력 설명은 모델의 주장이지 사람이 검증한 사실이 아니다.

## R01

- 기준 A는 음성이고 P3는 양성인 이상 포함 점수 구간: 해당 사례 없음.

### P3가 양성인 정상 점수 구간

`R01/testing/09`, 중심 0, 점수 [0,16), 점수 구간 이상 0/16프레임, 입력 이상 0/8프레임. 입력 ID `[0, 18, 37, 56, 75, 93, 112, 131]`.

**P1 / Output 0** ([원문](inference/P1/R01/testing/09/000000.json))

> Observations: The red power tool remains stationary on the tabletop throughout the frames. The green and black device is consistently positioned near the power tool. The lighting and object positions do not show any significant changes.
> 
> Assessment: The red power tool's stationary position and the consistent placement of the green and black device suggest a coherent industrial work process. There are no visible discrepancies in the object state, relative position, or tool-workpiece relationships.
> 
> Output: 0

**P2 / Output 0** ([원문](inference/P2/R01/testing/09/000000.json))

> Observations: The red power tool remains stationary throughout the images. The green and black device is consistently positioned on the tabletop. The blue object and silver tool are also stationary. There are no visible changes in object state, relative position, or tool-workpiece relationships.
> 
> Assessment: The objects and their positions are consistent across all frames, indicating no process deviation. The power tool's stationary state is expected given the context.
> 
> Output: 0

**P3 / Output 1** ([원문](inference/P3/R01/testing/09/000000.json))

> Observed changes: The red power tool is moved from the left side to the center of the tabletop.
> Evidence consistent: The power tool is stationary, and the green and black device remains in place.
> Evidence conflicting: The power tool's movement suggests a deviation from a coherent industrial work process.
> Uncertainty: None observed.
> Output: 1

### 새 세 조건이 모두 음성인 이상 포함 점수 구간

`R01/testing/06`, 중심 0, 점수 [0,16), 점수 구간 이상 6/16프레임, 입력 이상 7/8프레임. 입력 ID `[0, 18, 37, 56, 75, 93, 112, 131]`.

**P1 / Output 0** ([원문](inference/P1/R01/testing/06/000000.json))

> Observations: The red power tool remains stationary on the tabletop throughout the frames. The green and black device, along with the blue object, are consistently positioned near the power tool. The lighting and object positions do not show significant changes.
> 
> Assessment: The red power tool's stationary position and the consistent placement of other objects suggest a coherent industrial work process. There are no visible discrepancies in object state, relative position, or tool-workpiece relationships.
> 
> Output: 0

**P2 / Output 0** ([원문](inference/P2/R01/testing/06/000000.json))

> Observations: The red power tool remains stationary on the tabletop throughout the images. The green and black device, along with other tools, are consistently positioned on the cutting mat. There are no visible changes in the relative positions of the objects.
> 
> Assessment: The objects are consistently positioned and there are no visible changes in their state or relationships. The power tool is stationary, and no motion or process deviation is evident.
> 
> Output: 0

**P3 / Output 0** ([원문](inference/P3/R01/testing/06/000000.json))

> Observed changes: The red power tool remains stationary throughout the images.
> Evidence consistent: The power tool is stationary, and no other objects or their positions change.
> Evidence conflicting: None.
> Uncertainty: None.
> Output: 0

## R02

### 기준 A는 음성이고 P3는 양성인 이상 포함 점수 구간

`R02/testing/05`, 중심 192, 점수 [192,208), 점수 구간 이상 16/16프레임, 입력 이상 8/8프레임. 입력 ID `[42, 79, 117, 154, 192, 229, 267, 304]`.

**P1 / Output 0** ([원문](inference/P1/R02/testing/05/000192.json))

> Observations: The digital device remains stationary on the tabletop throughout the frames. The green clamp and metal bar are consistently positioned in the background. There are no visible changes in the object state, relative position, or tool-workpiece relationships.
> 
> Assessment: The scene appears consistent with a coherent industrial work process, as there are no deviations in the object state, relative position, or tool-workpiece relationships. The digital device's stationary position and the unchanged positions of the green clamp and metal bar support this assessment.
> 
> Output: 0

**P2 / Output 0** ([원문](inference/P2/R02/testing/05/000192.json))

> Observations: In Frame 8, a hand is seen touching the digital device on the metal table. The device remains stationary, but the interaction with it introduces a new element of human interaction not present in previous frames.
> 
> Assessment: The presence of a hand touching the digital device does not indicate a deviation from a coherent industrial work process, as it could be a normal part of the process, such as checking or adjusting the device.
> 
> Output: 0

**P3 / Output 1** ([원문](inference/P3/R02/testing/05/000192.json))

> Observed changes: A hand appears and touches the digital device in Frame 8.
> Evidence consistent: The digital device remains stationary on the metal table throughout the video.
> Evidence conflicting: The hand touching the device in Frame 8 is a deviation from a coherent industrial work process.
> Uncertainty: None observed.
> Output: 1

### P3가 양성인 정상 점수 구간

`R02/testing/02`, 중심 208, 점수 [208,224), 점수 구간 이상 0/16프레임, 입력 이상 0/8프레임. 입력 ID `[58, 95, 133, 170, 208, 245, 283, 320]`.

**P1 / Output 1** ([원문](inference/P1/R02/testing/02/000208.json))

> Observations: In Frame 8, a hand is seen interacting with the digital device on the tabletop. The device appears to be turned off or in a different state than in previous frames. The clamp and metal bar remain in their original positions.
> 
> Assessment: The interaction with the digital device suggests a change in its state, which could indicate a deviation from a coherent industrial work process. The clamp and metal bar are in their expected positions, and no other significant changes are visible.
> 
> Output: 1

**P2 / Output 1** ([원문](inference/P2/R02/testing/02/000208.json))

> Observations: In Frame 8, a hand is seen interacting with the digital device on the table. The device appears to be turned on or adjusted.
> Assessment: The interaction with the digital device suggests a process deviation, as the device should remain stationary throughout the video according to normal training observations.
> Output: 1

**P3 / Output 1** ([원문](inference/P3/R02/testing/02/000208.json))

> Observed changes: A hand is seen interacting with the digital device in Frame 8.
> Evidence consistent: The digital device remains stationary on the metal table throughout the video.
> Evidence conflicting: The hand's interaction with the digital device in Frame 8 suggests a deviation from a coherent industrial work process.
> Uncertainty: None observed.
> Output: 1

### 새 세 조건이 모두 음성인 이상 포함 점수 구간

`R02/testing/05`, 중심 0, 점수 [0,16), 점수 구간 이상 5/16프레임, 입력 이상 7/8프레임. 입력 ID `[0, 18, 37, 56, 75, 93, 112, 131]`.

**P1 / Output 0** ([원문](inference/P1/R02/testing/05/000000.json))

> Observations: The digital device is placed on the tabletop in all frames. The green clamp and metal bar are consistently positioned in the background. The lighting remains consistent throughout the frames.
> 
> Assessment: There are no visible discrepancies in the object state, relative position, tool-workpiece relationships, or changes across the supplied images. The scene appears coherent and consistent with an industrial work process.
> 
> Output: 0

**P2 / Output 0** ([원문](inference/P2/R02/testing/05/000000.json))

> Observations: The digital device remains stationary on the metal table throughout the video. The green clamp and silver metal bar are visible in the background, but their positions and states do not change. The lighting and background remain consistent.
> 
> Assessment: There are no visible discrepancies in the object state, relative position, tool-workpiece relationships, or changes across the supplied images. The scene appears to be a coherent industrial work process.
> 
> Output: 0

**P3 / Output 0** ([원문](inference/P3/R02/testing/05/000000.json))

> Observed changes: The digital device is placed on the metal table in an industrial setting with a green clamp and a silver metal bar.
> Evidence consistent: The digital device remains stationary on the metal table throughout the video.
> Evidence conflicting: None observed.
> Uncertainty: None observed.
> Output: 0

## R03

- 기준 A는 음성이고 P3는 양성인 이상 포함 점수 구간: 해당 사례 없음.

- P3가 양성인 정상 점수 구간: 해당 사례 없음.

### 새 세 조건이 모두 음성인 이상 포함 점수 구간

`R03/testing/01`, 중심 0, 점수 [0,16), 점수 구간 이상 16/16프레임, 입력 이상 8/8프레임. 입력 ID `[0, 18, 37, 56, 75, 93, 112, 131]`.

**P1 / Output 0** ([원문](inference/P1/R03/testing/01/000000.json))

> Observations: The yellow forklift-like object remains stationary in the same position across all frames. The black and white support with white cylindrical objects is also stationary and undisturbed. The green tabletop with grid patterns shows no signs of movement or change.
> 
> Assessment: There are no visible discrepancies in the object state, relative position, or tool-workpiece relationships. The scene appears consistent with a coherent industrial work process.
> 
> Output: 0

**P2 / Output 0** ([원문](inference/P2/R03/testing/01/000000.json))

> Observations: The yellow forklift remains stationary on the green tabletop throughout the frames. The black and white support with white cylindrical objects is consistently positioned to the left of the forklift.
> 
> Assessment: There are no visible discrepancies in object state, relative position, or tool-workpiece relationships. The scene appears consistent with a coherent industrial work process.
> 
> Output: 0

**P3 / Output 0** ([원문](inference/P3/R03/testing/01/000000.json))

> Observed changes: The yellow forklift remains stationary on the green tabletop throughout the frames.
> Evidence consistent: The yellow forklift remains stationary, and the objects on the tabletop do not show any significant changes in position or state.
> Evidence conflicting: No significant changes or discrepancies are observed in the objects or their positions.
> Uncertainty: None observed.
> Output: 0

## R04

### 기준 A는 음성이고 P3는 양성인 이상 포함 점수 구간

`R04/testing/02`, 중심 320, 점수 [320,336), 점수 구간 이상 16/16프레임, 입력 이상 7/8프레임. 입력 ID `[170, 198, 227, 255, 284, 312, 341, 369]`.

**P1 / Output 0** ([원문](inference/P1/R04/testing/02/000320.json))

> Observations: The blade is consistently cutting the brown material in a straight line. The clamp holds the material steady, and the metal bar is positioned correctly. The lighting remains consistent across all frames.
> 
> Assessment: The blade's consistent cutting action, the steady clamp, and the correct positioning of the metal bar indicate a coherent industrial work process. There are no visible discrepancies in object state, relative position, or tool-workpiece relationships.
> 
> Output: 0

**P2 / Output 0** ([원문](inference/P2/R04/testing/02/000320.json))

> Observations: The metal blade is stationary in the center of the metal tabletop with a green surface. The brown material is being cut by the blade in a consistent manner across the images. The clamp is holding the brown material in place.
> 
> Assessment: The process appears to be consistent and coherent, with no visible discrepancies in object state, relative position, or tool-workpiece relationships. The cutting process is stable and predictable.
> 
> Output: 0

**P3 / Output 1** ([원문](inference/P3/R04/testing/02/000320.json))

> Observed changes: The blade is moving back and forth across the brown material on the tabletop.
> Evidence consistent: The blade is stationary in the center of the tabletop with a green surface.
> Evidence conflicting: The blade is moving across the brown material, indicating a cutting process.
> Uncertainty: None observed.
> Output: 1

### P3가 양성인 정상 점수 구간

`R04/testing/02`, 중심 0, 점수 [0,16), 점수 구간 이상 0/16프레임, 입력 이상 4/8프레임. 입력 ID `[0, 18, 37, 56, 75, 93, 112, 131]`.

**P1 / Output 0** ([원문](inference/P1/R04/testing/02/000000.json))

> Observations: The blade is moving back and forth across the metal bar in a consistent manner. The clamp is holding the metal bar steady. The brown material is being cut by the blade. The lighting remains consistent across the frames.
> 
> Assessment: The blade's motion and the cutting process appear consistent and normal. The clamp is holding the metal bar steady, and the brown material is being cut as expected. There are no visible discrepancies in the object state, relative position, or tool-workpiece relationships.
> 
> Output: 0

**P2 / Output 0** ([원문](inference/P2/R04/testing/02/000000.json))

> Observations: In Frame1, a metal blade is stationary on a green tabletop. In Frame2, the blade is still stationary. In Frame3, the blade is still stationary. In Frame4, the blade is still stationary. In Frame5, the blade is still stationary. In Frame6, the blade is still stationary. In Frame7, the blade is still stationary. In Frame8, the blade is still stationary.
> 
> Assessment: The blade remains stationary throughout the entire sequence of images, which is consistent with a coherent industrial work process. There are no visible discrepancies in object state, relative position, tool-workpiece relationships, or changes across the supplied images.
> 
> Output: 0

**P3 / Output 1** ([원문](inference/P3/R04/testing/02/000000.json))

> Observed changes: The metal blade is moving and cutting a piece of brown material in the center of the metal tabletop with a green surface.
> Evidence consistent: The metal blade is stationary in the center of the metal tabletop with a green surface.
> Evidence conflicting: The metal blade is moving and cutting a piece of brown material.
> Uncertainty: None observed.
> Output: 1

### 새 세 조건이 모두 음성인 이상 포함 점수 구간

`R04/testing/02`, 중심 64, 점수 [64,80), 점수 구간 이상 13/16프레임, 입력 이상 5/8프레임. 입력 ID `[0, 26, 53, 80, 107, 133, 160, 187]`.

**P1 / Output 0** ([원문](inference/P1/R04/testing/02/000064.json))

> Observations: The brown material is being fed into the blade in a consistent manner across all frames. The clamp is holding the metal bar steady, and the tool-workpiece relationship remains unchanged. The lighting and object positions are consistent throughout the images.
> 
> Assessment: There are no visible discrepancies in the object state, relative position, tool-workpiece relationships, or changes across the supplied images. The process appears to be coherent and normal.
> 
> Output: 0

**P2 / Output 0** ([원문](inference/P2/R04/testing/02/000064.json))

> Observations: In Frame1, a metal blade is stationary on a green tabletop. In subsequent frames, a brown material is being fed into the blade, indicating a cutting process. The tool-workpiece relationship remains consistent, with the blade cutting the brown material.
> 
> Assessment: The observed process of feeding the brown material into the blade for cutting is coherent with an industrial work process. There are no visible discrepancies in object state, relative position, or tool-workpiece relationships.
> 
> Output: 0

**P3 / Output 0** ([원문](inference/P3/R04/testing/02/000064.json))

> Observed changes: The brown material is being fed into the metal blade in the center of the metal tabletop.
> Evidence consistent: The brown material is being processed by the metal blade, indicating a coherent industrial work process.
> Evidence conflicting: None observed.
> Uncertainty: None observed.
> Output: 0

