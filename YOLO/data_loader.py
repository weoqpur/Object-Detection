import os
import numpy as np
import pandas as pd
from xml.etree import ElementTree

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from PIL import Image
import tqdm

class_names = ['apple', 'banana', 'orange']

class Dataset(Dataset):
    def __init__(self, file_dir):
        """
        :param file_dir: dataset 경로
        """

        super(Dataset, self).__init__()
        self.img_files = [os.path.join(file_dir, img_file) for img_file in sorted(os.listdir(file_dir))
                          if img_file[-4:] == '.jpg']
        self.annot_files = [img_file[:-4] + '.xml' for img_file in self.img_files]

    def __len__(self):
        return len(self.img_files)


def convert_to_xywh(bboxes):
    """
    :param bboxes: xmin, ymin, xmax, ymax를 포함한 list
    :return: x_center, y_center, box_width, box_height
    """

    boxes = list()
    for box in bboxes:
        xmin, ymin, xmax, ymax = box

        box_width = xmax - xmin
        box_height = ymax - ymin

        x_center = xmin + (box_width // 2)
        y_center = ymin + (box_height // 2)

        boxes.append((x_center, y_center, box_width, box_height))

    return boxes

def extract_annotation_file(file_name):
    tree = ElementTree.parse(file_name)

    root = tree.getroot()
    boxes = list()
    classes = list()

    for box in root.findall('.//object'):
        cls = class_names.index(box.find('name').text)
        xmin = int(box.find('bndbox/xmin').text)
        ymin = int(box.find('bndbox/ymin').text)
        xmax = int(box.find('bndbox/xmax').text)
        ymax = int(box.find('bndbox/ymax').text)
        coors = (xmin, ymin, xmax, ymax)
        boxes.append(coors)
        classes.append(cls)

    boxes = convert_to_xywh(boxes)

    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)

    if width == 0 or height == 0:
        img = Image.open(file_name[:-4] + '.jpg')
        width, height = img.size[0], img.size[1]

    return boxes, classes, width, height

def convert_bboxes_to_tensor(bboxes, classes, img_width, img_height, grid_size=7):
    num_classes = len(class_names)
    target = np.zeros(shape=(grid_size, grid_size, 5 + num_classes), dtype=np.float32)

    for idx, bbox in enumerate(bboxes):
        x_center, y_center, width, height = bbox

        # grid안에 있는 각 셀 크기 계산
        cell_w, cell_h = img_width / grid_size, img_height / grid_size

        # bounding box의 셀 i, j 설정
        i, j = int(y_center / cell_h), int(x_center / cell_w)

        # 셀에서 x_center와 y_center 계산
        x, y = (x_center / cell_w) - j, (y_center / cell_h) - i

        # bounding box의 width, height 정규화
        w_norm, h_norm = width / img_width, height / img_height

        # x, y, w, h 설정
        target[i, j, :4] += (x, y, w_norm, h_norm)
        target[i, j, :4] = 1.
        # class dist 설정
        target[i, j, 5 + classes[idx]] = 1.

    return target
