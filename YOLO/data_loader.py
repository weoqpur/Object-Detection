import os
import numpy as np
import pandas as pd
from xml.etree import ElementTree

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
from skimage.transform import resize

class_names = ['apple', 'banana', 'orange']

class Dataset(Dataset):
    def __init__(self, file_dir, transform=None):
        """
        :param file_dir: dataset 경로
        :param transform: transform 여부
        """

        super(Dataset, self).__init__()
        img_files = [img_file for img_file in sorted(os.listdir(file_dir)) if img_file[-4:] == '.jpg']
        annot_files = [img_file[:-4] + '.xml' for img_file in img_files]
        images = pd.Series(img_files, name='images')
        annots = pd.Series(annot_files, name='annots')
        df = pd.concat([images, annots], axis=1)
        self.df = pd.DataFrame(df)
        self.file_dir = file_dir
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label_path = os.path.join(self.file_dir, self.df.iloc[index, 1])
        boxes = torch.tensor(extract_annotation_file(label_path))

        image_path = os.path.join(self.file_dir, self.df.iloc[index, 0])
        image = Image.open(image_path)
        image = image.convert("RGB")

        '''if self.transform:
            image, boxes = self.transfrom(image, boxes)'''

        image = transforms.Resize((448, 448))(image)
        image = transforms.ToTensor()(image)

        label_matrix = torch.zeros((7, 7, 13))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(7 * y), int(7 * x)
            x_cell, y_cell = 7 * x - j, 7 * y - i

            width_cell, height_cell = width * 7, height * 7

            if label_matrix[i, j, 3] == 0:
                # obj가 존재하도록 설정
                label_matrix[i, j, 3] = 1

                # box coordinates
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_matrix[i, j, 4:8] = box_coordinates

                # class_label의 one hot encoding 설정
                label_matrix[i, j, class_label] = 1

        data = {'image': image, 'label': label_matrix}

        return data





def extract_annotation_file(file_name):
    tree = ElementTree.parse(file_name)

    root = tree.getroot()
    boxes = list()

    img = Image.open(file_name[:-4] + '.jpg')
    width, height = img.size[0], img.size[1]

    for box in root.findall('.//object'):
        cls = class_names.index(box.find('name').text)
        xmin = int(box.find('bndbox/xmin').text)
        ymin = int(box.find('bndbox/ymin').text)
        xmax = int(box.find('bndbox/xmax').text)
        ymax = int(box.find('bndbox/ymax').text)

        x_center = ((xmax - xmin) / 2) / width
        y_center = ((ymax - ymin) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        boxes.append([cls, x_center, y_center, box_width, box_height])

    return boxes

def convert_bboxes_to_tensor(bboxes, classes, img_width, img_height, grid_size=7):
    num_classes = len(class_names)
    target = torch.zeros(shape=(grid_size, grid_size, 5 + num_classes), dtype=np.float32)

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
        target[i, j, :4] += torch.tensor(x, y, w_norm, h_norm)
        target[i, j, :4] = 1.
        # class dist 설정
        target[i, j, 5 + classes[idx]] = 1.

    return target

class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):

        key = list(data.keys())[0]

        h, w = data[key].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1], self.shape[2]))

        return data

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes