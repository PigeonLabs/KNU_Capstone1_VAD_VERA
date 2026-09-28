# 8단계 점수와 분리된 설명

영상별 최고 평균 점수 구간 1개씩, 정상 학습 참조 4프레임과 query 8프레임으로 생성했습니다. 256토큰·repetition_penalty 1.1, 총 37회. **아래 문장은 VLM 생성문이며 참조와의 차이가 실제 이상이라는 검증은 아닙니다.** 정상 영상의 위상 차이나 가시성 차이도 설명에 포함될 수 있습니다.

| 영상 | 중심 | 생성 원문 |
|---|---:|---|
| R01/testing/06 | 64 | The most noticeable difference between QUERY and REFERENCE is the presence of a red object in Image 5-12, which does not appear to be present or visible in Images 1-4. |
| R01/testing/08 | 0 | The most noticeable difference between QUERY and REFERENCE is the presence of a hand in Image 5-12, which is not present in Images 1-4. |
| R01/testing/09 | 224 | The most noticeable difference between QUERY and REFERENCE is the presence of a hand in Image 5-12, which is not present in Images 1-4. |
| R01/testing/10 | 144 | The most noticeable difference between QUERY and REFERENCE is the addition of a black object with green buttons in Image 5-12, which was not present in Images 1-4. |
| R01/training/03 | 0 | The most noticeable difference between QUERY and REFERENCE is the addition of a black object with red details in Image 5-12, which was not present in Images 1-4. |
| R01/training/04 | 0 | The most noticeable difference between QUERY and REFERENCE is the presence of a black object with red buttons in Image 5-12, which was not present in Images 1-4. |
| R01/training/05 | 0 | The most noticeable difference between QUERY and REFERENCE is the presence of a black object with red wheels in Image 5-12, which does not appear in Images 1-4. |
| R01/training/22 | 0 | The most noticeable difference between QUERY and REFERENCE is the addition of a red power drill in Image 5-12, which was not present in Images 1-4. |
| R01/training/25 | 96 | The most noticeable difference between QUERY and REFERENCE is the addition of a black object in Image 5-12, which was not present in Images 1-4. |
| R01/training/27 | 0 | The most noticeable difference between QUERY and REFERENCE is the addition of a green object with wires in Image 5-12, which was not present in Images 1-4. |
| R01/training/28 | 0 | The most noticeable difference between QUERY and REFERENCE is the addition of a green object in Image 5-12, which was not present in Images 1-4. |
| R02/testing/02 | 16 | The most noticeable difference between QUERY and REFERENCE is the presence of a hand pressing down on an object in Image 5-12, which is not present in Images 1-4. |
| R02/testing/05 | 16 | The most noticeable difference between QUERY and REFERENCE is the addition of a digital display on the device in Image 5-12. |
| R02/testing/07 | 304 | The most noticeable difference between QUERY and REFERENCE is the presence of a hand interacting with an object on Image 9, which is not present in any of the images labeled as REFERENCE. |
| R02/training/05 | 304 | The most noticeable difference between QUERY and REFERENCE is the presence of a hand interacting with an object on Image 9, which is not present in any of the images labeled as REFERENCE. |
| R02/training/09 | 304 | The most noticeable difference between QUERY and REFERENCE is the presence of a digital display on the device in Image 5-12, which is not present in Image 1-4. |
| R02/training/10 | 16 | The most noticeable difference between QUERY and REFERENCE is the presence of a hand interacting with an object on Image 5-12, which is not present in Images 1-4. |
| R02/training/23 | 16 | The most noticeable difference between QUERY and REFERENCE is the presence of a hand interacting with an object on the table in Image 5-12, which does not appear to be present or visible in Images 1-4. |
| R02/training/26 | 0 | The most noticeable difference between QUERY and REFERENCE is the presence of a hand interacting with an object on Image 5-12, which is not present in Images 1-4. |
| R02/training/28 | 0 | The most noticeable difference between QUERY and REFERENCE is the presence of a hand interacting with an object on Image 5-12, which does not appear in Images 1-4. |
| R03/testing/01 | 688 | The most noticeable difference between QUERY and REFERENCE is the addition of a white object with red text in Image 5-12, which was not present in Images 1-4. |
| R03/testing/08 | 688 | The most noticeable difference between QUERY and REFERENCE is the addition of a white object on top of the pallet in Image 5-12. |
| R03/testing/11 | 688 | The most noticeable difference between QUERY and REFERENCE is the addition of a white object on top of the forklift in Image 5-12. |
| R03/training/02 | 320 | The most noticeable difference between QUERY and REFERENCE is the presence of a white object on top of the forklift in Image 5-12, which does not appear to be there in Images 1-4. |
| R03/training/16 | 416 | The most noticeable difference between QUERY and REFERENCE is the addition of a white object in front of the forklift on the floor. |
| R03/training/20 | 672 | The most noticeable difference between QUERY and REFERENCE is the addition of a white object on top of the pallet in Image 5-12. |
| R03/training/21 | 704 | The most noticeable difference between QUERY and REFERENCE is the addition of a white cup on top of the pallet in Image 5-12. |
| R03/training/22 | 688 | The most noticeable difference between QUERY and REFERENCE is the addition of a white cup on top of the pallet in Image 5-12. |
| R04/testing/02 | 224 | The most noticeable difference between QUERY and REFERENCE is the presence of a person operating machinery in Image 5-12, which is not present in Images 1-4. |
| R04/testing/03 | 112 | The most noticeable difference between QUERY and REFERENCE is the presence of a brown paper sheet in Image 5-12, which does not appear to be present in Images 1-4. |
| R04/testing/09 | 288 | The most noticeable difference between QUERY and REFERENCE is the presence of a red tool in Image 5-12, which is not present in Images 1-4. |
| R04/testing/15 | 96 | The most noticeable difference between QUERY and REFERENCE is the presence of a red tool in Image 5-12, which is not present in Images 1-4. |
| R04/training/02 | 400 | The most noticeable difference between QUERY and REFERENCE is the presence of a red clamp in Image 5-12, which is not present in Images 1-4. |
| R04/training/17 | 272 | The most noticeable difference between QUERY and REFERENCE is the presence of a red tool in Image 5-12, which is not present in Images 1-4. |
| R04/training/18 | 224 | The most noticeable difference between QUERY and REFERENCE is the presence of a person in Image 5-12, which is not present in Images 1-4. |
| R04/training/20 | 368 | The most noticeable difference between QUERY and REFERENCE is the presence of a person in the background, which is not present in the REFERENCE images. |
| R04/training/25 | 336 | The most noticeable difference between QUERY and REFERENCE is the presence of a red tool in Image 5-12, which is not present in Images 1-4. |
