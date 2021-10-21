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

