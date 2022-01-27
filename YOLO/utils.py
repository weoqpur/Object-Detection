import os
import torch
import torch.nn as nn
import numpy as np
from loss import intersection_over_union
from collections import Counter

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save(ckpt_dir, model, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))

def load(ckpt_dir, model, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return model, optim, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit(), f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    model.load_state_dict(dict_model['model'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return model, optim, epoch

# 가중치 초기화
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format='corners'):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = list()

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes if box[0] != chosen_box[0] or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms



def convert_cellboxes(predictions, S=7):
    """
    yolo가 만든 bounding box를 셀비가 아닌 전체 이미지의 비로 변환
    """
    predictions = predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 13)
    bboxes1 = predictions[..., 4:8]
    bboxes2 = predictions[..., 9:13]
    scores = torch.cat(
        (predictions[..., 3].unsqueeze(0), predictions[..., 8].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = 1 / S * best_boxes[..., 2:4]
    converted_cellboxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = predictions[..., :3].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 3], predictions[..., 8]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_cellboxes), dim=-1)

    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = list()

    for ex_idx in range(out.shape[0]):
        bboxes = list()

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

# yolo 훈련의 시각화 bounding box 만들기
def get_bboxes(loader, model, iou_threshold, threshold, pred_format='cells', box_format='midpoint', decive='cpu'):
    all_pred_boxes = []
    all_true_boxes = []

    model.evel()
    train_idx = 0

    for batch_idx, data in enumerate(loader):
        x = data['image'].to(decive)
        labels = data['label'].to(decive)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

# 평균 정확도 계산
def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format='midpoint'):
    """

    :param pred_boxes: [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    각 bounding box와 함께 모든 bounding box를 포함하는 목록
    :param true_boxes: label bounding boxes
    :param iou_threshold: 예측된 bounding box의 임계값
    :param box_format: box를 어떻게 format할 것인가
    :return flaot: 특정 IoU 임계값이 지정된 모든 클래스의 mAP 값
    """

    average_precisions = list()

    epsilon = 1e-6

    for c in range(3):
        detections = list()
        ground_truths = list()

        # 모든 예측과 대상을 살펴보고 현재 클래스 c에 속하는 예측만 추가합니다.
        for detections in pred_boxes:
            if detections[1] == c:
                detections.append(detections)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # 각 교육 예제에 대한 bounding box의 양을 찾습니다. 여기서 카운터는 각 교육 예제에
        # 대해 많은 ground truth bounding box를 얻습니다.
        # 만약 img 0이 3을 가지고 img 1이 5를 가지면
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # 만약 class가 없다면 건너뛴다.
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # detection과 동일한 class의 index를 가진 ground truth만 뺀다.
            ground_truths_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truths_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truths_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # ground_truths detection 한 번만 실행
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive 및 이 bounding box를 표시에 추가
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # IoU가 낮으면 detection이 잘 이루어진게 아님
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # 수치 적분
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)