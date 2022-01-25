import numpy as np
import torch
import torch.nn as nn

coord_weight, noobj_weight = 5, 0.5

class YoloLoss(nn.Module):
    def __call__(self, predict, target):

        # get xywh, cfd, class
        target_cellbox = target[..., :4]
        target_obj = target[..., 4]
        target_cls = target[..., 5:]

        pred_cellbox1 = predict[..., :4]
        pred_obj1 = predict[..., 4]
        pred_cellbox2 = predict[..., 5:9]
        pred_obj2 = predict[..., 9]
        pred_cls = predict[..., 10:]

        target_corner_bbox = convert_cellbox_to_corner_bbox(target_cellbox, target_obj)
        pred_corner_bbox1 = convert_cellbox_to_corner_bbox(pred_cellbox1)
        pred_corner_bbox2 = convert_cellbox_to_corner_bbox(pred_cellbox2)

        iou_box1 = intersection_over_union(pred_corner_bbox1, target_corner_bbox)
        iou_box2 = intersection_over_union(pred_corner_bbox2, target_corner_bbox)

        ious = torch.stack([iou_box1, iou_box2], axis=-1)
        best_iou = torch.argmax(ious, axis=-1).type(torch.FloatTensor)

        xy_loss = compute_xy_loss(target_cellbox[..., :2], pred_cellbox1[..., :2], pred_cellbox2[..., :2], target_obj,
                                  best_iou)

        wh_loss = compute_wh_loss(target_cellbox[..., 2:], pred_cellbox1[..., 2:], pred_cellbox2[..., 2:], target_obj,
                                  best_iou)

        obj_loss = compute_obj_loss(target_obj, pred_obj1, pred_obj2, best_iou)

        no_obj_loss = compute_no_obj_loss(target_obj, pred_obj1, pred_obj2)

        cls_loss = compute_class_dist_loss(target_cls, pred_cls, target_obj)

        yolo_loss = (coord_weight * (xy_loss + wh_loss) + obj_loss + noobj_weight * no_obj_loss + cls_loss)

        return yolo_loss


def intersection_over_union(box_1, box_2):
    """
    box value: (x1, y1, x2, y2)
    :param box_1: 모델이 만든 bounding box (batch, 4)
    :param box_2: 원본 bounding box (batch, 4)
    :return: IOU
    """

    # 교차점 좌표 찾기
    x1 = torch.maximum(box_1[..., 0], box_2[..., 0])
    y1 = torch.maximum(box_1[..., 1], box_2[..., 1])
    x2 = torch.maximum(box_1[..., 2], box_2[..., 3])
    y2 = torch.maximum(box_1[..., 3], box_2[..., 3])

    # 범위 산출
    inter_area = torch.maximum(0.0, x2 - x1) + torch.maximum(0.0, y2 - y1)
    box_1_area = torch.abs((box_1[..., 2] - box_1[..., 0]) *
                           (box_1[..., 3] - box_1[..., 1]))
    box_2_area = torch.abs((box_2[..., 2] - box_1[..., 0]) *
                           (box_2[..., 3] - box_1[..., 1]))

    return inter_area / (box_1_area + box_2_area - inter_area)


def convert_cellbox_to_xywh(cellbox, mask=None):
    """
    (x_offset, y_offset, w, h) to (x_center, y_center, w, h)
    :param cellbox: tensor box (batch, grid, grid, 4)
    :param mask: 어떤 cell이 obj를 가지는지 결정하는 tensor
    :return bbox: Tensor box (batch, grid, grid, 4)
    """

    x_offset, y_offset = cellbox[..., 0], cellbox[..., 1]
    w_h = cellbox[..., 2:]

    num_w_cells = x_offset.shape[-1]
    num_h_cells = x_offset.shape[-2]

    # x_offset to x_center
    w_cell_indices = np.array(range(num_w_cells))
    w_cell_indices = np.broadcast_to(w_cell_indices, x_offset.shape[-2:])

    # y_offset to y_center
    h_cell_indices = np.array(range(num_h_cells))
    h_cell_indices = np.repeat(h_cell_indices, 7, 0).reshape(x_offset.shape[-2:])

    x_center = (x_offset + w_cell_indices) / num_w_cells
    y_center = (y_offset + h_cell_indices) / num_h_cells

    if mask is not None:
        x_center *= mask
        y_center *= mask

    x_y = torch.stack([x_center, y_center], axis=-1)

    bbox = torch.concat([x_y, w_h], axis=-1)

    return bbox

def convert_cellbox_to_corner_bbox(cellbox, mask=None):
    """
    (x_offset, y_offset, w, h) to (x_center, y_center, w, h)
    :param cellbox: tensor box (batch, grid, grid, 4)
    :param mask: 어떤 cell이 obj를 가지는지 결정하는 tensor
    :return corner_bbox: Tensor box (batch, grid, grid, 4)
    """

    bbox = convert_cellbox_to_xywh(cellbox, mask)
    x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]

    x_min = x - (w / 2)
    y_min = y - (h / 2)
    x_max = x + (w / 2)
    y_max = y + (h / 2)

    corner_box = torch.stack([x_min, y_min, x_max, y_max], axis=-1)

    return corner_box

def compute_xy_loss(target_xy, box1_xy, box2_xy, mask, best_iou):
    """
    :param target_xy: Target xy offset
    :param box1_xy: Prediction xy offset from box 1
    :param box2_xy: Prediction xy from box 2
    :param mask: 어떤 cell이 obj를 포함하는지 결정하는 tensor
    :param best_iou: 어떤 bounding box가 predictor인지 결정하는 tensor
    :return xy_loss: xy_loss
    """
    sse_xy_1 = torch.sum(torch.square(target_xy - box1_xy), -1)
    sse_xy_2 = torch.sum(torch.square(target_xy - box2_xy), -1)

    xy_predictor_1 = sse_xy_1 * mask * (1 - best_iou)
    xy_predictor_2 = sse_xy_2 * mask * best_iou
    xy_predictor = xy_predictor_1 + xy_predictor_2

    xy_loss = torch.sum(torch.sum(xy_predictor, [1, 2]))

    return xy_loss

def compute_wh_loss(target_wh, box1_wh, box2_wh, mask, best_iou):
    """
    :param target_wh: Target wh offset
    :param box1_wh: prediction box1의 wh offset
    :param box2_wh: prediction box2의 wh offset
    :param mask: 어떤 cell이 obj를 포함하는지 결정하는 tensor
    :param best_iou: 어떤 bounding box가 predictor인지 결정하는 tensor
    :return wh_loss:
    """

    target_wh = torch.sqrt(target_wh)
    box1_wh, box2_wh = torch.sqrt(torch.abs(box1_wh)), torch.sqrt(torch.sqrt(box2_wh))

    sse_wh_1 = torch.sum(torch.square(target_wh - box1_wh), -1)
    sse_wh_2 = torch.sum(torch.square(target_wh - box2_wh), -1)

    wh_predictor_1 = sse_wh_1 * mask * (1 - best_iou)
    wh_predictor_2 = sse_wh_2 * mask * best_iou
    wh_predictor = wh_predictor_1 + wh_predictor_2

    wh_loss = torch.mean(torch.sum(wh_predictor, [1, 2]))

    return wh_loss

def compute_obj_loss(target_obj, box1_obj, box2_obj, best_iou):
    """
    :param target_obj: Target obj (셀에 obj가 포함되어 있는 경우:1 else: 0)
    :param box1_obj: prediction box1의 obj
    :param box2_obj: prediction box2의 obj
    :param best_iou: 어떤 bounding box가 predictor인지 결정하는 tensor
    :return obj_loss:
    """

    pred_obj_1 = box1_obj * target_obj * (1 - best_iou)
    pred_obj_2 = box2_obj * target_obj * best_iou
    pred_obj = pred_obj_1 + pred_obj_2

    sqrt_err_obj = torch.square(target_obj - pred_obj)

    obj_loss = torch.mean(torch.sum(sqrt_err_obj, [1, 2]))

    return obj_loss

def compute_no_obj_loss(target_obj, box1_obj, box2_obj):
    """
    :param target_obj: Target obj (셀에 obj가 포함되어 있는 경우:1 else: 0)
    :param box1_obj: prediction box1의 obj
    :param box2_obj: prediction box2의 obj
    :return no_obj_loss:
    """

    target_no_obj_mask = 1 - target_obj

    pred_no_obj_1 = box1_obj * target_no_obj_mask
    pred_no_obj_2 = box2_obj * target_no_obj_mask

    sqr_err_no_obj_1 = torch.square((target_obj * target_no_obj_mask) - pred_no_obj_1)
    sqr_err_no_obj_2 = torch.square((target_obj * target_no_obj_mask) - pred_no_obj_2)
    sqr_err_no_obj = sqr_err_no_obj_1 + sqr_err_no_obj_2

    no_obj_loss = torch.mean(torch.sum(sqr_err_no_obj, [1, 2]))

    return no_obj_loss

def compute_class_dist_loss(target_cls, pred_cls, mask):
    """
    :param target_cls: Target class distribution
    :param pred_cls: class prediction
    :param mask: 어떤 cell이 obj를 포함하는지 결정하는 tensor
    :return cls_loss:
    """

    sse_cls = torch.sum(torch.square(target_cls - pred_cls), -1)
    sse_cls = sse_cls * mask

    cls_loss = torch.mean(torch.sum(sse_cls, [1, 2]))

    return cls_loss