import PIL.Image
import cv2
import numpy as np
import random
from torchvision import transforms
import PIL
import glob
import os
import torch
from data.AugImgTorch2 import PerlinTorch
from data.randomized_quantization import RandomizedQuantizationAugModule


def patch_shuffle(img, n):
    noise = img.copy()
    w = np.size(noise, 1)
    h = np.size(noise, 0)
    patch_w = w // n
    patch_h = h // n
    patch_list = []
    sy = 0
    for i in range(n):
        sx = 0
        for j in range(n):
            patch_list.append(noise[sy: sy + patch_h, sx: sx + patch_w, :].copy())
            sx += patch_w
        sy += patch_h
    random.shuffle(patch_list)
    sy = 0
    for i in range(n):
        sx = 0
        for j in range(n):
            noise[sy: sy + patch_h, sx: sx + patch_w, :] = patch_list[i * n + j]
            sx += patch_w
        sy += patch_h
    return noise

def random_padding(img, wsize=16, hsize=16):
    pad_w = random.randint(0, wsize)
    pad_h = random.randint(0, hsize)
    src_w, src_h = np.size(img, 1), np.size(img, 0)
    left_x = src_w // 4
    top_y = src_h // 4
    right_x = src_w - src_w // 4
    bottom_y = src_h - src_h // 4
    x = random.randint(left_x, right_x - pad_w)
    y = random.randint(top_y, bottom_y - pad_h)
    img[y: y + pad_h, x: x + pad_w, :] = 0  
    return img


def gaussian_noise(img):
    noise_scale=0.3
    noise_std=1.0
    img_ = img.astype(float) / 255.0
    noise = np.random.normal(0, noise_std, img.shape)
    img_ = img_ * (1.0 - noise_scale) + noise * noise_scale
    img_ = np.clip(img_, 0, 1.0) * 255.0
    img = img_.astype(np.uint8)
    # cv2.imwrite('D:/data/ocr_training_data/img_aug_hsl.bmp', img)
    return img


def color_jitter(img):
    jitter_trans = transforms.ColorJitter(0, 0.4, 1.0, 0.5)
    pil_img = PIL.Image.fromarray(img)
    aug_img = jitter_trans(pil_img)
    aug_img = np.array(aug_img)
    return aug_img


def gray_jitter(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def rand_quantize(img):
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    bin = random.randint(2, 16)
    rq = RandomizedQuantizationAugModule(bin, transforms_like=False)
    y = rq(x)
    y = y.numpy().squeeze().transpose(1, 2, 0).astype(np.uint8)
    return y


def rescale(img):
    ra = random.randint(0, 100)
    h, w = np.size(img, 0), np.size(img, 1)
    if ra < 50:
        aug_img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    else:
        aug_img = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    aug_img = cv2.resize(aug_img, (w, h), interpolation=cv2.INTER_AREA)
    return aug_img


def sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5) 
    magnitude = cv2.magnitude(sobelx, sobely)
    min = np.min(magnitude)
    max = np.max(magnitude)
    magnitude = (magnitude - min) / (max - min) * 255.0
    magnitude = magnitude.astype(np.uint8)
    return magnitude


def PerlinMerge(img, noise_img):
    PL_tool = PerlinTorch(thr0=220, thr1=250, p0=1, p2=6, sigma=1.5, base0=5, base1=7)
    h = np.size(img, 0)
    w = np.size(img, 1)
    f_tensor, _ = PL_tool.gen_mask(1, h, w)
    f_tensor = f_tensor.squeeze().unsqueeze(-1) # h*w*1
    img = img.astype(float) / 255.0
    noise_img = noise_img.astype(float) / 255.0
    img = torch.from_numpy(img) #h*w*3
    noise_img = torch.from_numpy(noise_img)
    aug_img = img * (1.0 - f_tensor) + noise_img * f_tensor
    aug_img = torch.clamp(aug_img, 0, 1.0) * 255.0
    aug_img = aug_img.numpy().astype(np.uint8)
    return aug_img


class AugOptor:
    def __init__(self, opt_list=None):
        if opt_list is None:
            self.opt_list = ['patch_shuffle', 'gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'PerlinMerge', 'rand_quantize', 'gray_jitter', 'random_padding']
        else:
            self.opt_list = opt_list

    def get_opt_list(self):
        return ['patch_shuffle', 'gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'PerlinMerge', 'rand_quantize', 'gray_jitter', 'random_padding']
 
    def opt_img(self, img, noise_img=None):
        opt_num = len(self.opt_list)
        opt_idx = random.randint(0, opt_num-1)
        if self.opt_list[opt_idx] == 'patch_shuffle':
            aug_img = patch_shuffle(img, 4)
        elif self.opt_list[opt_idx] == 'gaussian_noise':
            aug_img = gaussian_noise(img)
        elif self.opt_list[opt_idx] == 'color_jitter':
            aug_img = color_jitter(img)
        elif self.opt_list[opt_idx] == 'rescale':
            aug_img = rescale(img)
        elif self.opt_list[opt_idx] == 'sobel':
            aug_img = sobel(img)
        elif self.opt_list[opt_idx] == 'PerlinMerge':
            aug_img = PerlinMerge(img, noise_img)
        elif self.opt_list[opt_idx] == 'rand_quantize':
            aug_img = rand_quantize(img)
        elif self.opt_list[opt_idx] == 'gray_jitter':
            aug_img = gray_jitter(img)
        elif self.opt_list[opt_idx] == 'random_padding':
            aug_img = random_padding(img)
        else:
            aug_img = None
        return aug_img


class AugOptorPreb:
    def __init__(self, opt_list=None, preb_list=None):
        if opt_list is None:
            self.opt_list = ['patch_shuffle', 'gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'PerlinMerge', 'rand_quantize', 'gray_jitter', 'random_padding']
            self.preb_list = []
            for i in range(len(self.opt_list)):
                self.preb_list.append(1.0)
        else:
            self.opt_list = opt_list
            self.preb_list = preb_list
        sum = 0
        for i in range(len(self.preb_list)):
            sum += self.preb_list[i]
        for i in range(len(self.preb_list)):
            self.preb_list[i] /= sum
        for i in range(len(self.preb_list)):
            if i > 0:
                self.preb_list[i] += self.preb_list[i-1]

    def get_opt_list(self):
        return ['patch_shuffle', 'gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'PerlinMerge', 'rand_quantize', 'gray_jitter', 'random_padding']
 
    def opt_img(self, img, noise_img=None):
        opt_num = len(self.opt_list)
        ra = random.random()
        for i in range(opt_num):
            if ra < self.preb_list[i]:
                opt_idx = i
                break
        if self.opt_list[opt_idx] == 'patch_shuffle':
            aug_img = patch_shuffle(img, 4)
        elif self.opt_list[opt_idx] == 'gaussian_noise':
            aug_img = gaussian_noise(img)
        elif self.opt_list[opt_idx] == 'color_jitter':
            aug_img = color_jitter(img)
        elif self.opt_list[opt_idx] == 'rescale':
            aug_img = rescale(img)
        elif self.opt_list[opt_idx] == 'sobel':
            aug_img = sobel(img)
        elif self.opt_list[opt_idx] == 'PerlinMerge':
            aug_img = PerlinMerge(img, noise_img)
        elif self.opt_list[opt_idx] == 'rand_quantize':
            aug_img = rand_quantize(img)
        elif self.opt_list[opt_idx] == 'gray_jitter':
            aug_img = gray_jitter(img)
        elif self.opt_list[opt_idx] == 'random_padding':
            aug_img = random_padding(img, 32, 32)
        else:
            aug_img = None
        return aug_img
    

if __name__ == '__main__':
    img = cv2.imread('/home/primary/data1/mvTec/data/cable/train/good/000.png', cv2.IMREAD_COLOR)
    img= cv2.resize(img, (256, 256))
    cv2.imwrite('/home/primary/data1/mvTec/runs/10/src.png', img)
    opt1 = ['patch_shuffle', 'gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'rand_quantize']
    opt1_preb = [1, 1, 1, 1, 1, 1]
    augoptor = AugOptorPreb(opt1, opt1_preb)
    for i in range(50):
        aug = augoptor.opt_img(img)
        cv2.imwrite('/home/primary/data1/mvTec/runs/10/{}.png'.format(i), aug)