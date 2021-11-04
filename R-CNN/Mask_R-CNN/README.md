# Mask R-CNN

Mask R-CNN은 Faster R-CNN과 다르게 Image segmentation을 수행하기 위해 고안된 모델이다.   
구조는 Faster R-CNN과 크게 다르지 않고 mask branch와 FPN, 그리고 RoI align이 추가된 것 말고는 없다.

1. Fast R-CNN의 classification, localization(bounding box regression) branch에 새롭게 mask branch가 추가됐다.
2. RPN 전에 FPN(feature pyramid network)가 추가됐다.
3. Image segmentation의 masking을 위해 RoI align이 RoI Pooling을 대신하게 됐다.

위 3가지를 설명하기에 앞서 Mask R-CNN의 구조를 살펴보도록 하자.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc0pdEg%2FbtqBL8vzmxg%2F1zkQAmbSKShCvdqXx8jXkk%2Fimg.png)   

N x N 사이즈의 인풋 이미지가 주어졌을때 Mask R-CNN의 process는 다음과 같다.

## process

1. 800~1024 사이즈로 이미지를 resize해준다. (using bilinear interpolation)
2. Backbone network의 인풋으로 들어가기 위해 1024 x 1024의 인풋사이즈로 맞춰준다. (using padding)
3. ResNet-101을 통해 각 layer(stage)에서 feature map (c1, c2, c3, c4, c5)를 생성한다.
4. FPN을 통해 이전에 생성된 feature map에서 P2, P3, P4, P5, P6 feature map을 생성한다.
5. 최종 생성된 feature map에 각각 RPN을 적용하여 classification, bbox regression output값을 도출한다.
6. output으로 얻은 bbox regression값을 원래 이미지로 projection시켜서 anchor box를 생성한다.
7. Non-max-suppression을 통해 생성된 anchor box 중 score가 가장 높은 anchor box를 제외하고 모두 삭제한다.
8. 각각 크기가 서로다른 anchor box들을 RoI align을 통해 size를 맞춰준다.
9. Fast R-CNN에서의 classification, bbox regression branch와 더불어 mask branch에 anchor box값을 통과시킨다.

## Resize input image

Mask R-CNN에서는 backbone으로 ResNet-101을 사용하는데 ResNet 네트워크에서는 이미지 input size가 800~1024일때 성능이 좋다고 알려져있다.   
따라서 이미지를 위 size로 맞춰주는데 이때 bilinear interpolation을 사용하여 resize해준다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbsf2P5%2FbtqBPYeHm3Z%2FaMt9hUpAVr57ZCPYrxv4B0%2Fimg.png)   
bilinear interpolation  

bilinear interpolation은 여러 interpolation기법 중 하나로 동작과정은 다음과 같다.
2 x 2의 이미지를 위 그림과 같이 4 x 4로 Upsampling을 한다면 2 x 2에 있던 pixel value가 각각
P1, P2, P3, P4로 대응된다. 이때 총 16개 중 4개의 pixel만 값이 대응되고 나머지 12개는 값이 아직 채워지지 않았는데
이를 bilinear interpolation으로 값을 채워주는 것이다. 계산하는 방법은 아래와 같다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcPSvrn%2FbtqBQnekM6R%2F0AbGEOE0zdw7AtjU1FckA0%2Fimg.png)   

이렇게 기존의 image를 800~1024사이로 resize해준 후 네트워크의 input size인 1024 x 1024로 맞춰주기 위해 나머지
값들은 zero padding으로 값을 채워준다.

## Backbone-ResNet101

Mask R-CNN에서는 Backbone으로 ResNet모델을 다음과 같다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc9budm%2FbtqBSOa9F71%2FTf2pCxuju04Ke6wmovWyaK%2Fimg.png)   

[ResNet](https://github.com/weoqpur/classification/tree/main/ResNet) <- 여기에 ResNet을 정리해두었다.

## Feature Pyramid Network

이전의 Faster R-CNN에서는 backbone의 결과로 나온 1개의 feature map에서 RoI를 생성하고 classification 및 bbox regression을 진행했다.
해당 feature map은 backbone 모델에서 최종 layer에서의 output인데 이렇게 layer를 통과할수록 아주 중요한 feature만 남게되고 중간중간의 feature들은
모두 잃어버리고 만다. 그리고 최종 layer에서 다양한 크기의 object를 검출해야하므로 여러 scale값으로 anchor를 생성하므로 비효율적이다.

따라서 이를 극복하기 위한 방법이 FPN이다.   
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbkIepJ%2FbtqBOC4iYQb%2F7WxQWcEOXMQqhC6AkNfWA0%2Fimg.png)   
FPN

FPN에서는 위 그림과 같이 마지막 layer의 feature map에서 점점 이전의 중간 feature map들을 더하면서 이전 정보까지 유지할 수 있도록 한다.
이렇게 함으로써 더 이상 여러 scale값으로 anchor를 생성할 필요가 없게됐고 모두 동일한 scale의 anchor를 생성한다. 따라서 작은 feature map에서는 큰
anchor를 생성하여 큰 object를, 큰 feature map에서는 다소 작은 anchor를 생성하여 작은 object를 detect할 수 있도록 설계되었다.

마지막 layer에서의 feature map에서 이전 feature map을 더하는 것은 Upsampling을 통해 이루어진다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdp0vqm%2FbtqBNNFfeKy%2FSnYsbgYs0TmgH1CiQtjjnK%2Fimg.png)   
FPN detail

위 그림과 같이 먼저 2배로 upsampling을 한 후 이전 layer의 feature map을 1x1 Fully convolution 연산을 통해 filter개수를 똑같이 맞춰준 후
더함으로써 새로운 feature map을 생성한다.

결과적으로 ResNet을 통해 C1, C2, C3, C4, C5 feature map을 생성하고 C1은 사용하지 않고 C5부터 F5, F4, F3, F2를 생성한다. 이때 5F에서
maxpooling을 통해 F6을 추가로 생성한다. 최종적으로 F2, F3, F4, F5, F6 이렇게 5개의 feature map이 생성되는데 이때 F2~F5의 경우 RPN에 보내기전에
3x3 convolution 연산을 거친 후 보낸다. F2~F5는 upsampling과 이전 feature map을 더함으로써 feature data가 조금 망가졌을 수 있기에 3x3 연산을
한 번 더해주는 것이다.
반면, F6은 F5에서 max pooling을 한 결과이므로 3x3연산을 하지않고 RPN에 그대로 전달된다.