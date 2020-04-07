from dataset import affectnet_loader
from torchvision.transforms.functional import to_pil_image
import torch
import cv2
import numpy as np
import math
from PIL import Image

loader = affectnet_loader()
i = 0
for data in loader:
    img, target, landmark = data
    for i in range(8):
        pixel = to_pil_image(img[i])
        pixel.save('s%d.png' % i)
        label = target[i].detach().numpy()
        print(label)
        mark = landmark[i].detach().numpy()
        pixel = cv2.imread('s%d.png' % i)
        for j in range(mark.shape[0] // 2):
            cv2.circle(pixel, (math.floor(mark[2 * j]), math.floor(mark[2 * j + 1])), 1, (0, 255, 0), -1)
        cv2.imwrite('l%d.png' % i, pixel)
    i += 1
    if i >= 1:
        break
