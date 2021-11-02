# Mask R-CNN

Mask R-CNN은 Faster R-CNN과 다르게 Image segmentation을 수행하기 위해 고안된 모델이다.   
구조는 Faster R-CNN과 크게 다르지 않고 mask branch와 FPN, 그리고 RoI align이 추가된 것 말고는 없다.

1. Fast R-CNN의 classification, localization(bounding box regression) branch에 새롭게 mask branch가 추가됐다.
2. RPN 전에 FPN(feature pyramid network)가 추가됐다.
3. Image segmentation의 masking을 위해 RoI align이 RoI Pooling을 대신하게 됐다.

위 3가지를 설명하기에 앞서 Mask R-CNN의 구조를 살펴보도록 하자.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc0pdEg%2FbtqBL8vzmxg%2F1zkQAmbSKShCvdqXx8jXkk%2Fimg.png)   

N x N 사이즈의 인풋 이미지가 주어졌을때 Mask R-CNN의 process는 다음과 같다.

### process

1. 800~1024 사이즈로 이미지를 resize해준다. (using bilinear interpolation)
2. Backbone network의 인풋으로 들어가기 위해 1024 x 1024의 인풋사이즈로 맞춰준다. (using padding)
3. ResNet-101을 통해 각 layer(stage)에서 feature map (c1, c2, c3, c4, c5)를 생성한다.
4. FPN을 통해 이전에 생성된 feature map에서 P2, P3, P4, P5, P6 feature map을 생성한다.
5. 최종 생성된 feature map에 각각 RPN을 적용하여 classification, bbox regression output값을 도출한다.
6. output으로 얻은 bbox regression값을 원래 이미지로 projection시켜서 anchor box를 생성한다.
7. Non-max-suppression을 통해 생성된 anchor box 중 score가 가장 높은 anchor box를 제외하고 모두 삭제한다.
8. 각각 크기가 서로다른 anchor box들을 RoI align을 통해 size를 맞춰준다.
9. Fast R-CNN에서의 classification, bbox regression branch와 더불어 mask branch에 anchor box값을 통과시킨다.