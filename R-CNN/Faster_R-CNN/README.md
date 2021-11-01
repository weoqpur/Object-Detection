# Faster R-CNN

이 모델은 Fast R-CNN의 병목 현상을 해결하고자 만들어졌다.   
어떤 병목이냐 region proposal인 Selective search알고리즘을 CNN외부에서 연산하므로 RoI 생성단계가
병목이다. 

따라서 Faster R-CNN에서는 detection에서 쓰인 conv feature를 RPN에서도 공유해서
RoI생성역시 CNN level에서 수행하여 속도를 향상시킨다.


## Faster R-CNN

Selective search가 느린 이유는 cpu에서 돌기 때문이다.   
따라서 Region proposal 생성하는 네트워크도 gpu에 넣기 위해서 Conv layer에서 생성하도록 하자는게
아이디어 이다.

Faster R-CNN은 한마디로 RPN + Fast R-CNN이라할 수 있다.   
Faster R-CNN은 Fast R-CNN구조에서 conv feature map과 RoI Pooling사이에 RoI를 생성하는
Region Proposal Network가 추가된 구조이다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdhq4iV%2FbtqBaAFDl4d%2FIZdxlDX5mkPMdnoKy2f2k0%2Fimg.png)   

그리고 Faster R-CNN에서는 RPN 네트워크에서 사용할 CNN과 Fast R-CNN에서 classification, bbox regression을 위해 사용한 CNN 네트워크를 공유하자는
개념에서 나왔다.

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb3fKvm%2FbtqA7qcGUyK%2FRtIY6qVkJ6yerNqBUnV0h1%2Fimg.png)   

결국 위 그림에사와 같이 CNN을 통과하여 생성된 conv feature map이 RPN에 의해 RoI를 생성한다.
주의해야할 것이 생성된 RoI는 feature map에서의 RoI가 아닌 original image에서의 RoI이다.
(그래서 코드 상에서도 anchor box의 scale은 original image 크기에 맞춰서 (128, 256, 512)와 같이 생성하고
이 anchor box와 network의 output 값 사이의 loss를 optimize하도록 훈련시킨다.)

따라서 original image위에서 생성된 RoI는 아래 그림과 같이 conv feature map의 크기에 맞게 rescaling된다.
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FewrNhQ%2FbtqByFOw4xg%2FELJ9xbK9EKR3OJFDL7j6E0%2Fimg.png)   

이렇게 feature map에 RoI가 투영되고 나면 FC layer에 의해 classification과 bbox regression이 수행된다.
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FzJoAZ%2FbtqBBU4w395%2FBMWWphbMKuo4HbzFjIM0T0%2Fimg.png)   
위 그림에서 보다시피 마지막에 FC layer를 사용하기에 input size를 맞춰주기 위해 RoI pooling을 사용한다.
RoI pooling을 사용하니까 RoI들의 size가 달라도 되는 것 처럼 original image의 input size도 달라도 된다.
그러나 구현할때 코드를 보면 original image의 size는 같은 크기로 맞춰주는데 그 이유는

"vgg의 경우 244x224, resNet의 경우 min : 600, max : 1024 등 으로 맞춰줄때 성능이 가장 좋기 때문이다."

original image를 resize할때 손실되는 data가 존재 하듯이
feature map을 RoI pooling에서 max pooling을 통해 resize할때 손실되는 data도 존재한다.
따라서 이때 손실되는 data와 input image를 resize할때 손실되는 data 사이의 trade off가 각각 
vgg의 경우 244x244이고, ResNet의 경우 600~1024이기에 input size를 고정시킨다.

**"따라서 요즘에는 FC layer보다 GAP(Global average pooling)을 사용하는 추세이다"**

GAP를 사용하면 input size와 관계없이 1 value로 average pooling하기에 filter의 개수만 정해져있으면 되기 때문이다.
따라서 input size를 고정할 필요가 없기에 RoI pooling으로 인해 손실되는 data도 없고 original image의 size역시 
고정시킬 필요가 없는 장점이 있다.

### RPN(Region proposal network)

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcBZOij%2FbtqBgEtQ5CC%2Fsi04v7TSFdRndJyckCsxwK%2Fimg.png)   

RPN의 input 값은 이전 CNN 모델에서 뽑아낸 feature map이다.
Region proposal을 생성하기 위해 feature map위에 nxn window를 sliding window 시킨다.
이때, object의 크기와 비율이 어떻게 될지모르므로 k개의 anchor box를 미리 정의해놓는다.
이 anchor box가 bounding box가 될 수 있는 것이고 미리 가능할만한 box모양 k개를 정의해놓는것이다.
여기서는 가로세로길이 3종류 x 비율 3종류 = 9개의 anchor box박스를 이용한다.

이 단계에서 9개의 anchor box를 이용하여 classification과 bbox regression을 먼저 구한다. (for 학습)
먼저, CNN에서 뽑아낸 feature map에 대해 3x3 conv filter 256개를 연산하여 depth를 256으로 만든다.
그 후 1x1 conv 두개를 이용해서 각각 classification과 bbox regression을 계산한다.

이때 network를 가볍게 하기 위해 binary classification으로 bbox에 물체가 있나 없나만 판단한다.
무슨 물체인지 판단하는 것은 마지막 classification 단계에서 한다.

RPN단계에서 classification과 bbox regression을 하는 이유는 결국 학습을 위함이다.
위 단계로 부터 positive / negative examples들을 뽑아내는데 다음 기준을 따른다.   
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FIEkcY%2FbtqBcwbZTpn%2FvSU5RjT6EBjvUkp2mtVpfk%2Fimg.png)   
IoU가 0.7보다 크거나, 한 지점에서 모든 anchor box중 가장 IoU가 큰 anchor box는 positive example로 만든다.
IoU가 0.3보다 작으면 object가 아닌 background를 뜻 하므로 negative example로 만들고
이 사이에 있는 값은 애매한 값이므로 학습에 사용하지 않는다.

### Non-Maximum Suppression

Faster R-CNN에 대한 학습이 완료된 후 RPN모델을 예측시키면 한 객체당 여러 proposal값이 나올 것이다.
이 문제를 해결하기 위해 NMS알고리즘을 사용하여 proposal의 개수를 줄인다. NMS알고리즘은 다음과 같다.
1. box들의 score(confidence)를 기준으로 정렬한다.
2. score가 가장 높은 box부터 시작해서 다른 모든 box들과 IoU를 계산해서 0.7이상이면 같은 객체를 detect한 box라고
생각할 수 있기 때문에 해당 box는 지운다.
   
3. 최종적으로 각 object별로 score가 가장 높은 box 하나씩만 남게 된다.   

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FlAtRG%2FbtqBGgAw0Dd%2FxakhVprkQJKjnztAGjJRl1%2Fimg.png)   
NMS 전

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbFQwuR%2FbtqBG0jD3nh%2FbhkdKFOk0PbkmWAh9qiDQ1%2Fimg.png)   
NMS 후


