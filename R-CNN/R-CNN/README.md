# R-CNN

컴퓨터 비전에서의 문제들은 크게 4가지로 분류할 수 있다.
1. Classification
2. Object Detection
3. Image Segmentation
4. Visual relationship

이중에서 R-CNN은 Object Detection 모델이다.

우선 Object detection에는 1-stage detector, 2-stage detector가 있다.

## 1-stage detector와 2-stage detector란?

![`1stage`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Frd2Ho%2FbtqBcxO6C0m%2FMCINIrwGAnzMjevTDOqKJ0%2Fimg.png)   
1-stage detector vs 2-stage detector

직선을 기준으로 위가 2-stage detector들이고 아래가 1-stage detector들이다.
___
### 2-stage detector

Regional Proposal과 classification이 순차적으로 이뤄진다.

**Regional Proposal이란?**
기존에는 이미지에서 object detection을 위해 sliding window방식을 이용했었다.
Sliding window 방식은 이미지에서 모든 영역을 다양한 크기의 window로 탐색하는 것이다.
!['sliding'](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FwYMa4%2FbtqA6pruEvn%2FJJGkGhvMK2yIw1pVzKNGtk%2Fimg.png)   
Sliding window

이런 비효율성을 개선하기 위해 **물체가 있을만한** 영역을 빠르게 찾아내는 알고리즘이 region proposal이다.
대표적으로 selective search, Edge boxes들이 있다.
(*selective search: 비슷한 질감, 색, 강도를 갖는 인접 픽셀로 구성된 다양한 크기의 window를 생성한다.)
즉, region proposal은 object의 위치를 찾는 localization문제이다.

따라서 2-stage detector에서 regional proposal과 classification이 순차적으로 이루어진다는 것은
classification과 localization문제를 순차적으로 해결한다는 것이다.

## R-CNN

R-CNN은 2-stage detector모델이다.
