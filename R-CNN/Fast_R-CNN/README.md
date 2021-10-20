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

## RoI Pooling

Fast R-CNN에서 이 SPP가 적용되는 것을 보면 다음과 같다.
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FIiNzk%2FbtqA8iSURGO%2F8F29HIsdwxAd6kMUnuKuu1%2Fimg.png)   
실제로 Fast R-CNN에서는 1개의 피라미드를 적용시킨 SPP로 구성되어있다.
또한 피라미드의 사이즈는 7x7이다. Fast R-CNN에서 적용된 1개의 피라미드 SPP로 고정된
크기의 feature vector를 만드는 과정을 **RoI Pooling**이라 한다.

### RoI Pooling

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FoHUnL%2FbtqBc5dG1ix%2F9EWJiCVhnHoTFZQTtCckYK%2Fimg.png)   
Fast R-CNN에서 먼저 입력 이미지를 CNN에 통과시켜 feature map을 추출한다.
그 후 이전에 미리 Selective search로 만들어놨던 RoI(=Region proposal)을 feature map에 projection시킨다.
위 그림의 가장 좌측 그림이 feature map이고 그 안에 *h*x*w* 크기의 검은색 box가 투영된 RoI이다.

1. 미리 설정한 *H*x*W* 크기로 만들어주기 위해서 (*h*/*H*) * (*w*/*H*) 크기만큼 grid를 RoI위에 만든다.
2. RoI를 grid크기로 split시킨 뒤 max pooling을 적용시켜 결국 각 grid 칸마다 하나의 값을 추출한다.

위 작업을 통해 feature map에 투영했던 *H*x*W*크기의 RoI는 *H*x*W*크기의 고정된 feature vector로 변환된다.

이렇게 RoI pooling을 이용함으로써
원래 이미지를 CNN에 통과시켜도 FC layer의 input에 맞출 수 있게 되었다.

## end-to-end : Trainable
다음은 R-CNN의 두번째 문제였던 multi-stage pipeline으로 인해 3가지 모델을 따로 학습해야 했던 문제이다.
R-CNN에서는 CNN을 통과한 후 각각 서로다른 모델인 SVM(classification), bounding box regression(localization)
안으로 들어가 forward됐기 때문에 연산이 공유되지 않았다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcC15WF%2FbtqA57Lvbgm%2FZX3VwTFw89kc2Gbx2SKuD0%2Fimg.png)   

그러나 위 그림을 다시보면 RoI Pooling을 추가함으로써 이제 RoI영역을 CNN을 거친후의 feature map에 투영시킬 수 있었다.
따라서 동일 data가 각자 softmax(classification), box regressor(localization)으로 들어가기에 연산을 공유한다.
이는 이제 모델이 end-to-end로 한 번에 학습시킬 수 있다는 뜻이다.

### Loss function

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcUFclv%2FbtqA57Y1CWZ%2FlAVBX4FyK0dW47IhBfJNC1%2Fimg.png)   

이제 Fast R-CNN의 Loss function은 위와 같이 classification과 localization loss를 합친 function으로써 한 번의 학습으로 둘 다 학습시킬 수 있다.
