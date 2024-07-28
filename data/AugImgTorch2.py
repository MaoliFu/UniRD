import os
import cv2
import numpy as np
import random
from torch import Tensor, nn
import torch
from kornia.filters import gaussian_blur2d
import torch.nn.functional as F


class PerlinTorch():
    def __init__(self, thr0=160, thr1=240, p0=1, p2=3, sigma=1.5, base0=5, base1=8):
        super(PerlinTorch, self).__init__()
        # self.sz = 64
        self.thr0 = thr0
        self.thr1 = thr1
        self.p0 = p0
        self.p2 = p2
        self.sigma = sigma
        self.base0 = base0
        self.base1 = base1

    def gen_mask(self, b, h, w, p=-1):
        mask_frd_list = []
        mask_bgd_list = []
        torch.manual_seed(random.randint(0, 10000000))
        for i in range(b):
            ra = random.randint(self.base0, self.base1)
            base_sz = 2 ** ra
            if p == -1:
                p = random.randint(self.p0, self.p2)
            pp = torch.randperm(256)
            lin_tensor = torch.linspace(0, p, base_sz + 1)
            lin_tensor = lin_tensor[:-1]
            yt, xt = torch.meshgrid(lin_tensor, lin_tensor, indexing='ij')
            mask_tensor = self.perlin_torch(xt, yt, pp)
            # mask = mask + 0.5
            max_v = torch.max(mask_tensor)
            min_v = torch.min(mask_tensor)
            diff = max_v - min_v
            mask_tensor = (mask_tensor - min_v) / diff # * 255.0
            '''mask = mask*255.0
            mask = np.clip(mask, 0, 255)'''
            # mask_tensor = mask_tensor.type(torch.uint8)
            # thr = random.randint(thr0, thr1)
            thr = torch.randint(self.thr0, self.thr1, (1,)).item() / 255.0
            mask_bin = (mask_tensor > thr).float()
            mask_bin = gaussian_blur2d(mask_bin.unsqueeze(0).unsqueeze(1), (3, 3), (self.sigma, self.sigma))
            # mask_bgd = (mask_bin == 0).astype(np.uint8) * 255
            # mask_bin = cv2.GaussianBlur(mask_bin, (3, 3), 1.5)
            # mask_bgd = (mask_bin == 0).astype(np.uint8) * 255
            if h != base_sz or w != base_sz:
                mask_bin = F.interpolate(mask_bin, size=(w, h), mode='bilinear', align_corners=False)
                # mask_bgd = F.interpolate(mask_bgd, size=(w, h), mode='bilinear', align_corners=False)
            mask_bgd = (mask_bin == 0).float()
            mask_frd_list.append(mask_bin)
            mask_bgd_list.append(mask_bgd)
        mask_bin = torch.cat(mask_frd_list, dim=0)
        mask_bgd = torch.cat(mask_bgd_list, dim=0)

        return mask_bin, mask_bgd

    def perlin_torch(self, x, y, p):
        # permutation table
        p = torch.stack([p, p]).flatten()

        # coordinates of the top-left
        xi = x.to(torch.int64)
        yi = y.to(torch.int64)
        # internal coordinates
        xf = x - xi.to(torch.float32)
        yf = y - yi.to(torch.float32)
        # fade factors
        u = self.fade(xf)
        v = self.fade(yf)
        # noise components
        pxi = p[xi]
        pxi_plus_one = p[xi + 1]
        pxiyi = p[pxi + yi]
        pxiyi_plus_one = p[pxi + yi + 1]
        pxi_plus_oneyi = p[pxi_plus_one + yi]
        pxi_plus_oneyi_plus_one = p[pxi_plus_one + yi + 1]
        # add blended results from 4 corners of each grid cell
        n1 = self.gradient_tensor(pxiyi, xf, yf)
        n2 = self.gradient_tensor(pxiyi_plus_one, xf, yf - 1)
        ix1 = self.lerp(n1, n2, v)
        n1 = self.gradient_tensor(pxi_plus_oneyi, xf - 1, yf)
        n2 = self.gradient_tensor(pxi_plus_oneyi_plus_one, xf - 1, yf - 1)
        ix2 = self.lerp(n1, n2, v)
        return self.lerp(ix1, ix2, u)

    def lerp(self, a, b, x):
        "linear interpolation"
        return a + x * (b - a)

    def fade(self, t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def gradient_tensor(self, h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y


if __name__ == '__main__':
    for i in range(100):
        # mask_frd, mask_bgd = get_mask(128, p2=5, thr0=210)
        PLmask = PerlinTorch(thr0=220, thr1=250, p0=1, p2=6, sigma=1.5, base0=5, base1=7) #.cuda()
        f_tensor, b_tensor = PLmask.gen_mask(1, 256, 256)
        b = f_tensor.size(0)
        f_tensor = f_tensor * 255.0
        b_tensor = b_tensor * 255.0

        f_tensor = f_tensor.squeeze()
        b_tensor = b_tensor.squeeze()

        f_tensor = f_tensor.numpy().astype(np.uint8)
        b_tensor = b_tensor.numpy().astype(np.uint8)
        for j in range(b):
            cv2.imwrite('/home/primary/data1/mvTec/log2/test/{}_{}_frd.bmp'.format(i, j), f_tensor)
        # cv2.imwrite('H:/Anomaly detection data set/mvtec_anomaly_detection.tar/log/all/3/t/{}_bgd.bmp'.format(i),
        #             b_tensor)
