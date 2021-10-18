# Fast R-CNN

Fast R-CNN은 이전 R-CNN의 한계점을 극복하고자 나왔다.
한계점
1. RoI (Region of Interest) 마다 CNN연산을 함으로써 속도저하
2. multi-stage pipelines으로써 모델을 한 번에 학습시키지 못 함
   
다음과 같은 한계점들이 있다.

Fast R-CNN에서는 다음 두 가지를 통해 위 한계점들을 극복했다.
1. RoI pooling
2. CNN 특징 추출부터 classification, bounding box regression까지 하나의 모델에서 학습

## Fast R-CNN process

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcC15WF%2FbtqA57Lvbgm%2FZX3VwTFw89kc2Gbx2SKuD0%2Fimg.png)
Fast R-CNN의 수행과정은 다음과 같다.

   1-1. R-CNN에서 마찬가지로 Selective Search를 통해 RoI를 찾는다.   
   1-2. 전체 이미지를 CNN에 통과시켜 feature map을 추출한다.   
   2. Selective Search로 찾았었던 RoI를 feature map크기에 맞춰서 projection시킨다.   
   3. projection시킨 RoI에 대해 RoI Pooling을 진행하여 고정된 크기의 feature vector를 얻는다.   
   4. feature vector는 FC layer를 통과한 뒤, 구 브랜치로 나뉘게 된다.   
   5-1. 하나는 softmax를 통과하여 RoI에 대해 object classification을 한다.  
   5-2. bounding box regression을 통해 selective search로 찾은 box의 위치를 조정한다.

### 핵심 아이디어
Fast R-CNN의 핵심 아이디어는 RoI Pooling이다.
R-CNN에서 CNN output이 FC layer에 input으로 들어가야 했기에 CNN input을 FC layer의 input과 동일한 사이즈로 맞춰야 했습니다.
그러나 Fast R-CNN은 **RoI Pooling**을 사용하여 FC layer전에 사이즈를 조정하여 넣어주기 때문에 CNN input의 사이즈가 고정되지 않습니다.

여기서 Spatial Pyramid Pooling(SPP)이 제안된다.

## Spatial Pyramid Pooling(SPP)

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbLaqYY%2FbtqA6EhMolU%2FWhKtYSETGVYyeKgZYUUpZ0%2Fimg.png)   
SPP에서는 먼저 이미지를 CNN에 통과시켜 feature map을 추출한다.
그리고 미리 정해진 4x4, 2x2, 1x1 영역의 피라미드로 feature map을 나눠준다. 피라미드 한칸을 bin이라 한다.
bin내에서 max pooling을 적용하여 각 bin마다 하나의 값을 추출하고,
최종적으로 피라미드 크기만큼 max값을 추출하여 3개의 피라미드의 결과를 쭉 이어붙여 고정된 크기의 vector를 만든다.

그렇게 만든 vector값이 FC layer의 input으로 들어간다.

따라서 CNN을 통과한 feature map에서 2000개의 region proposal을 만들고 region proposal마다 SPPNet에 집어넣어 고정된 크기의 feature vector를 얻어낸다.
이 작업을 통해 모든 2000개의 region proposal마다 해야했던 CNN연산이 1번으로 줄었다.

