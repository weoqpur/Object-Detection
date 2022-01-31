import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
from data_loader import Dataset
from model import Detect
from utils import cellboxes_to_boxes, non_max_suppression

model_dir = './epochs/model_epoch19.pth'
model = Detect().eval()
model.load_state_dict(torch.load(model_dir, map_location='cpu')['model'])

image = Image.open(image_dir).convert('RGB')
transform = Compose([Resize((448, 448)), ToTensor()])
image = transform(image).unsqueeze(0)

for idx in range(8):
    bboxes = cellboxes_to_boxes(model(image))
    bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format='midpoint')

    img = np.array(image[idx].permute(1,2,0).to('cpu'))
    height, width, _ = img.shape

    # Create fiqure
    fig, ax = plt.subplots(1)
    # 이미지 출력
    ax.imshow(img)

    # 정사각형 patch 생성
    for box in bboxes:
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        # Add patch
        ax.add_patch(rect)

    plt.show()










