import numpy as np
import cv2
from torchvision.transforms import RandomApply, ColorJitter, ToTensor, ToPILImage, Normalize, RandomGrayscale, Compose
import numbers
import warnings
import random
import math
import torch
from PIL import Image
from torchvision.transforms.functional import crop, hflip


class RandomGaussianBlur(object):
    def __init__(self, p=0.5, min=0.1, max=2.0, kernel_size=9):
        self.min = min
        self.max = max
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        sample = Image.fromarray(sample, 'RGB')
        return sample


def erase(img, i, j, h, w, v, inplace=False):
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.clone()

    img[:, i:i + h, j:j + w] = v
    return img


class RandomErasing(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                v = value
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
            img = erase(img, x, y, h, w, v, self.inplace)
            return img
        return img


class RandomResizedCrop(object):
    def __init__(self, size=224, range=(0.9, 1.0)):
        self.range = range
        self.size = size

    def __call__(self, img, landmark):
        h, w, c = np.array(img).shape
        px = random.uniform(self.range[0], self.range[1])
        py = random.uniform(self.range[0], self.range[1])
        new_h, new_w = np.round(py * h), np.round(px * w)
        x = np.floor(random.uniform(0, w - new_w))
        y = np.floor(random.uniform(0, w - new_h))
        new_img = crop(img, y, x, new_h, new_w)
        new_img = new_img.resize((self.size, self.size))
        new_landmark = np.zeros_like(landmark)
        for i in range(landmark.shape[0]):
            m = landmark[i]
            new_landmark[i] = (m - x) * self.size / new_w if i % 2 == 0 else (m - y) * self.size / new_h
        new_landmark = torch.tensor(new_landmark)
        return new_img, new_landmark


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, landmark):
        h, w, c = np.array(img).shape
        if random.uniform(0, 1) < self.p:
            new_img = hflip(img)
            new_landmark = np.zeros_like(landmark)
            for i in range(landmark.shape[0]):
                m = landmark[i]
                new_landmark[i] = w - m if i % 2 == 0 else m
            new_landmark = torch.tensor(new_landmark)
            return new_img, new_landmark
        else:
            return img, landmark


class Augment(object):
    def __init__(self):
        self.random_resized_crop = RandomResizedCrop(range=(0.85,1.0))
        self.random_horizontal_flip = RandomHorizontalFlip(p=1.0)
        self.other_transform = Compose([
            RandomApply([ColorJitter(0.8, 0.8, 0.2)], p=0.3),
            RandomGrayscale(p=0.3),
            RandomGaussianBlur(p=0.3),
            ToTensor()
        ])

    def __call__(self, img, landmark):
        img = ToPILImage()(img)
        img, landmark = self.random_resized_crop(img, landmark)
        img, landmark = self.random_horizontal_flip(img, landmark)
        img = self.other_transform(img)
        return img, landmark


def transform_augment():
    return Augment()
