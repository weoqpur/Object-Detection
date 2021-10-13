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

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fsh68S%2FbtqBcxuQWbw%2FoO78Y4XgO0j0q2fR0mytBk%2Fimg.png)   

R-CNN은 Image classification을 수행하는 CNN과 localization을 위한 regional proposal알고리즘을 연결한 모델이다.

**R-CNN 프로세스**
1. Image를 입력받는다.
2. Selective search알고리즘에 의해 regional proposal output 약 2000개를 추출한다.
   추출한 regional proposal output을 모두 동일 input size로 만들어주기 위해 warp해준다.
   (왜 동일 input size로 만들어주냐? Convolution Layer에서 마지막 FC layer의 input size가 
   고정이므로 Convolution Layer에 대한 output size도 동일해야한다.)
    
3. 2000개의 warped image를 각각 CNN 모델에 넣는다.
4. 각각의 Convolution 결과에 대해 classification을 진행하여 결과를 얻는다.
   
### 1. Region Proposal (영역 찾기)
R-CNN은 Region Proposal을 할때 Selective search를 사용한다.
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMMlO6%2FbtqA7pEJsfi%2F4fLKHSxIkKJ8tEaFvKQ651%2Fimg.png)   
Selective Search

1. 색상, 질감, 영역크기 등 을 이용해 non-object-based segmentation을 수행한다.
이 작업을 통해 좌측 제일 하단 그림과 같이 많은 small segmented areas들을 얻을 수 있다.
   
2. Bottom-up 방식으로 small segmented areas들을 합쳐서 더 큰 segmented areas들을 만든다.
3. (2)작업을 반복하여 최종적으로 2000개의 region proposal을 생성한다.

Selective search 알고리즘에 의해 2000개의 region proposal이 생성되면 이들을 모두 CNN에 넣기 전에
같은 사이즈로 warp시켜야 한다. (CNN output size를 동일하게 만들기 위해)

### 2. CNN
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcVwCdl%2FbtqA9BLoE49%2FTL94t2Kdy745q9pBCYZlq0%2Fimg.png)   

Warp작업을 통해 region proposal 모두 224x224 크기로 되면 CNN 모델에 넣는다.
여기서 CNN은 AlexNet의 구조를 거의 그대로 가져다 썼다.
최종적으로 CNN을 거쳐 각각의 region proposal로부터 4096-dimentional feature vector를 뽑아내고,
이를 통해 고정길이의 Feature Vector를 만들어낸다.

### 3. SVM

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHTaEr%2FbtqA9BxS2bV%2FkQJvYDyBDzpKY9pwVjegW1%2Fimg.png)   

CNN모델로부터 feature가 추출되면 Linear SVM을 통해 classification을 진행한다.
위에서 설명했듯 Classifier로 softmax보다 SVM이 더 좋은 성능을 보였기 때문에 SVM을 채택했다.
SVM은 CNN으로부터 추출된 각각의 feature vector들의 점수를 class별로 매기고, 객체인지 아닌지,
객체라면 어떤 객체인지 등을 판별하는 역할을 하는 Classifier이다.

### 3-1. Bounding Box Regression

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FLbP8c%2FbtqBaAZLZKc%2F1wxNUB5zD7XikkSoFRKtgK%2Fimg.png)   
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtIrL9%2FbtqBfsHBlpd%2FLlKUlXZGZrZBlR3ToWxkXK%2Fimg.png)   

Selective search로 만든 bounding box는 정확하지 않기 때문에 물체를 정확히 감싸도록 조정해주는
bounding box regression(선형회귀 모델)이 존재한다.
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbZHHte%2FbtqBaBxDVWC%2FAVMf11jZOEsiSpoaK148h0%2Fimg.png)   
bounding box regression 수식


