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
### 2-