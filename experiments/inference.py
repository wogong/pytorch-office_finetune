import torch
import sys
from PIL import Image
sys.path.append('../')
from models.alexnet import alexnet
import cv2
import numpy as np

model = alexnet(pretrained=True)
model.eval()
image_jpg = Image.open('test.jpg')
img = cv2.imread('test.jpg')
img = img.astype(np.float32)
mean_color = [104.0069879317889, 116.66876761696767, 122.6789143406786]
img -= np.array(mean_color)
img = torch.from_numpy(img)
img = img.transpose(0, 1).transpose(0, 2).contiguous()

img = img.unsqueeze(0)

preds = model(img)
print('predicted class is: {}'.format(preds.argmax()))
