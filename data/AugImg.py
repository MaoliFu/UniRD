import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import random


def perlin(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    pxi = p[xi]
    pxi_plus_one = p[xi + 1]
    pxiyi = p[pxi + yi]
    pxiyi_plus_one = p[pxi + yi + 1]
    pxi_plus_oneyi = p[pxi_plus_one + yi]
    pxi_plus_oneyi_plus_one = p[pxi_plus_one + yi + 1]
    n00 = gradient(pxiyi, xf, yf)
    n01 = gradient(pxiyi_plus_one, xf, yf - 1)
    n11 = gradient(pxi_plus_oneyi_plus_one, xf - 1, yf - 1)
    n10 = gradient(pxi_plus_oneyi, xf - 1, yf)

    '''n001 = gradient(p[p[xi] + yi], xf, yf)
    n011 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n111 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n101 = gradient(p[p[xi + 1] + yi], xf - 1, yf)'''

    '''n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)'''
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here


def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)


def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


def get_mask(sz, thr0=160, thr1=240, p0=1, p2=3):
    p = random.randint(p0, p2)
    lin = np.linspace(0, p, sz, endpoint=False)
    x, y = np.meshgrid(lin, lin)
    mask = perlin(x, y, random.randint(0, 10000000))
    # mask = mask + 0.5
    max_v = np.max(mask)
    min_v = np.min(mask)
    diff = max_v - min_v
    mask = (mask - min_v) / diff * 255.0
    '''mask = mask*255.0
    mask = np.clip(mask, 0, 255)'''
    mask = mask.astype(np.uint8)
    thr = random.randint(thr0, thr1)
    _, mask_bin = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)
    mask_bin = cv2.GaussianBlur(mask_bin, (3, 3), 1.5)
    mask_bgd = (mask_bin == 0).astype(np.uint8) * 255
    # mask_frd = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    # mask_bgd = cv2.cvtColor(mask_bgd, cv2.COLOR_GRAY2BGR)
    return mask_bin, mask_bgd


def get_mask2(sz, thr0=160, thr1=240, p0=1, p2=3):
    p = random.randint(p0, p2)
    lin = np.linspace(0, p, sz, endpoint=False)

    x, y = np.meshgrid(lin, lin)
    mask = perlin(x, y, random.randint(0, 10000000))
    # mask = mask + 0.5
    max_v = np.max(mask)
    min_v = np.min(mask)
    diff = max_v - min_v
    mask = (mask - min_v) / diff * 255.0
    '''mask = mask*255.0
    mask = np.clip(mask, 0, 255)'''
    mask = mask.astype(np.uint8)
    thr = random.randint(thr0, thr1)
    _, mask_bin = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)
    mask_bin = cv2.GaussianBlur(mask_bin, (3, 3), 1.5)
    mask_bgd = (mask_bin == 0).astype(np.uint8) * 255
    # mask_frd = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    # mask_bgd = cv2.cvtColor(mask_bgd, cv2.COLOR_GRAY2BGR)
    return mask_bin, mask_bgd


def get_light_param(new128=-1):
    pt = np.zeros([3, 2])
    pt[0][0] = pt[0][1] = 0.0
    pt[2][0] = pt[2][1] = 255.0
    pt[1][0] = 128.0
    if new128 == -1:
        pt[1][1] = random.uniform(80, 176)
    else:
        pt[1][1] = new128
    # pt[1][1] = 170

    A = np.zeros([3, 3])
    b = np.zeros([3, 1])

    for i in range(3):
        A[i][0] = pt[i][0] ** 2
        A[i][1] = pt[i][0]
        A[i][2] = 1.
        b[i] = pt[i][1]

    A_t = A.T
    A_t_A = np.dot(A_t, A)
    A_t_B = np.dot(A_t, b)

    A_t_A_ = np.linalg.inv(A_t_A)

    param = A_t_A_.dot(A_t_B)

    pGray = np.zeros(256)
    for i in range(256):
        pGray[i] = i * i * param[0] + i * param[1] + param[2]
    pGray = pGray.astype(np.uint8)
    return pGray


def aug_light_3ch(img):
    b, g, r = cv2.split(img)
    pGray_0_127 = get_light_param()
    dst_b = cv2.LUT(b, pGray_0_127)
    dst_g = cv2.LUT(g, pGray_0_127)
    dst_r = cv2.LUT(r, pGray_0_127)
    dst = cv2.merge([dst_b, dst_g, dst_r])
    return dst


def aug_light_3ch_left_right(img, left, right):
    b, g, r = cv2.split(img)
    s = random.uniform(left, right)
    pGray_0_127 = get_light_param(s)
    dst_b = cv2.LUT(b, pGray_0_127)
    dst_g = cv2.LUT(g, pGray_0_127)
    dst_r = cv2.LUT(r, pGray_0_127)
    dst = cv2.merge([dst_b, dst_g, dst_r])
    return dst


def aug_light_3ch_2(img, model):
    b, g, r = cv2.split(img)
    bm, gm, rm = cv2.split(model)

    pGray_0_127 = get_light_param()

    dst_b = cv2.LUT(b, pGray_0_127)
    dst_g = cv2.LUT(g, pGray_0_127)
    dst_r = cv2.LUT(r, pGray_0_127)
    dst = cv2.merge([dst_b, dst_g, dst_r])

    dst_b = cv2.LUT(bm, pGray_0_127)
    dst_g = cv2.LUT(gm, pGray_0_127)
    dst_r = cv2.LUT(rm, pGray_0_127)
    dst_model = cv2.merge([dst_b, dst_g, dst_r])
    return dst, dst_model


def aug_light_1ch(img):
    pGray_0_127 = get_light_param()
    dst = cv2.LUT(img, pGray_0_127)
    return dst


def aug_color_1ch(img, off_scale=25.0):
    m = np.mean(img)
    off = random.uniform(-1.0, 1.0) * off_scale
    off = min(255.0, max(0.0, m + off))
    m = off / m
    dst = img * m
    dst = np.clip(dst, 0, 255.0)
    dst = dst.astype(np.uint8)
    return dst


def aug_color(img, off_scale=25.0, off_thr=5.0):
    m = cv2.mean(img)
    if m[0] == 0 or m[1] == 0 or m[2] == 0:
        print('err: {} {}'.format(np.size(img, 0), np.size(img, 1)))
        return img
    # off_scale = 10.0
    off_b = random.uniform(-1.0, 1.0) * off_scale
    off_g = random.uniform(-1.0, 1.0) * off_scale
    off_r = random.uniform(-1.0, 1.0) * off_scale
    while abs(off_b) < off_thr and abs(off_g) < off_thr and abs(off_r) < off_thr:
        off_b = random.uniform(-1.0, 1.0) * off_scale
        off_g = random.uniform(-1.0, 1.0) * off_scale
        off_r = random.uniform(-1.0, 1.0) * off_scale

    off_b = min(255.0, max(0.0, m[0] + off_b))
    off_g = min(255.0, max(0.0, m[1] + off_g))
    off_r = min(255.0, max(0.0, m[2] + off_r))

    mean1 = list(m)
    mean1[0] = off_b / m[0]
    mean1[1] = off_g / m[1]
    mean1[2] = off_r / m[2]
    # cv2.imwrite('H:/All_data/TCJ/log_j2/src.bmp', img)
    # img = cv2.multiply(img, mean1)

    dst_0 = img[:, :, 0] * mean1[0]
    dst_1 = img[:, :, 1] * mean1[1]
    dst_2 = img[:, :, 2] * mean1[2]
    # cv2.imwrite('H:/All_data/TCJ/log_j2/dst.bmp', img)
    dst_img = cv2.merge([dst_0, dst_1, dst_2])
    dst_img = np.clip(dst_img, 0, 255.0)
    dst_img = dst_img.astype(np.uint8)
    return dst_img


def aug_color2(img, model, off_scale=25.0, off_thr=5.0):
    m = cv2.mean(img)
    mm = cv2.mean(model)
    if m[0] == 0 or m[1] == 0 or m[2] == 0:
        print('err: {} {}'.format(np.size(img, 0), np.size(img, 1)))
        return img, model
    # off_scale = 10.0
    off_b = random.uniform(-1.0, 1.0) * off_scale
    off_g = random.uniform(-1.0, 1.0) * off_scale
    off_r = random.uniform(-1.0, 1.0) * off_scale
    while abs(off_b) < off_thr and abs(off_g) < off_thr and abs(off_r) < off_thr:
        off_b = random.uniform(-1.0, 1.0) * off_scale
        off_g = random.uniform(-1.0, 1.0) * off_scale
        off_r = random.uniform(-1.0, 1.0) * off_scale

    off_b = min(255.0, max(0.0, m[0] + off_b))
    off_g = min(255.0, max(0.0, m[1] + off_g))
    off_r = min(255.0, max(0.0, m[2] + off_r))

    off_b_m = min(255.0, max(0.0, mm[0] + off_b))
    off_g_m = min(255.0, max(0.0, mm[1] + off_g))
    off_r_m = min(255.0, max(0.0, mm[2] + off_r))

    mean1 = list(m)
    mean1[0] = off_b / m[0]
    mean1[1] = off_g / m[1]
    mean1[2] = off_r / m[2]

    mean2 = list(mm)
    mean2[0] = off_b_m / mm[0]
    mean2[1] = off_g_m / mm[1]
    mean2[2] = off_r_m / mm[2]
    # cv2.imwrite('H:/All_data/TCJ/log_j2/src.bmp', img)
    # img = cv2.multiply(img, mean1)

    dst_0 = img[:, :, 0] * mean1[0]
    dst_1 = img[:, :, 1] * mean1[1]
    dst_2 = img[:, :, 2] * mean1[2]
    # cv2.imwrite('H:/All_data/TCJ/log_j2/dst.bmp', img)
    dst_img = cv2.merge([dst_0, dst_1, dst_2])
    dst_img = np.clip(dst_img, 0, 255.0)
    dst_img = dst_img.astype(np.uint8)

    dst_0 = model[:, :, 0] * mean2[0]
    dst_1 = model[:, :, 1] * mean2[1]
    dst_2 = model[:, :, 2] * mean2[2]
    # cv2.imwrite('H:/All_data/TCJ/log_j2/dst.bmp', img)
    dst_model = cv2.merge([dst_0, dst_1, dst_2])
    dst_model = np.clip(dst_model, 0, 255.0)
    dst_model = dst_model.astype(np.uint8)

    return dst_img, dst_model


def add_gaussian_noise(img):
    img_ = img.astype(float)
    noise = np.random.normal(0, 2.5, img.shape)
    img_ = img_ + noise
    img_ = np.clip(img_, 0, 255)
    img = img_.astype(np.uint8)
    # cv2.imwrite('D:/data/ocr_training_data/img_aug_hsl.bmp', img)
    return img


def aug_light_3ch_rigid(img, min128=70, max128=200, thr=10):
    b, g, r = cv2.split(img)
    while True:
        new128 = random.randint(min128, max128)
        if abs(new128-128) > thr:
            break

    pGray_0_127 = get_light_param(new128)
    dst_b = cv2.LUT(b, pGray_0_127)
    dst_g = cv2.LUT(g, pGray_0_127)
    dst_r = cv2.LUT(r, pGray_0_127)
    dst = cv2.merge([dst_b, dst_g, dst_r])
    return dst


def add_gaussian_noise2(img, noise_scale=2.5, noise_std=2.5):
    img_ = img.astype(float) / 255.0
    noise = np.random.normal(0, noise_std, img.shape)
    img_ = img_ * (1.0 - noise_scale) + noise * noise_scale
    img_ = np.clip(img_, 0, 1.0) * 255.0
    img = img_.astype(np.uint8)
    # cv2.imwrite('D:/data/ocr_training_data/img_aug_hsl.bmp', img)
    return img


def get_homo(cols, rows):
    step_c = cols * 0.1
    step_r = rows * 0.1

    x0 = random.uniform(-step_c, step_c)
    y0 = random.uniform(-step_r, step_r)

    x1 = random.uniform(-step_c + cols, step_c + cols)
    y1 = random.uniform(-step_r, step_r)

    x2 = random.uniform(-step_c + cols, step_c + cols)
    y2 = random.uniform(-step_r + rows, step_r + rows)

    x3 = random.uniform(-step_c, step_c)
    y3 = random.uniform(-step_r + rows, step_r + rows)

    src_pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    dst_pts = np.array([[0.0, 0.0], [cols - 1.0, 0.0], [cols - 1.0, rows - 1.0], [0.0, rows - 1.0]])

    H, mask = cv2.findHomography(dst_pts, src_pts)
    return H


def random_crop(x, w, h, clip_width=10):
    t = random.randint(1, clip_width)
    l = random.randint(1, clip_width)
    b = random.randint(1, clip_width)
    r = random.randint(1, clip_width)
    y = x[t: -b, l: -r, ...]
    if np.size(y, 0) == 0 or np.size(y, 1) == 0:
        return x
    y = cv2.resize(y, (w, h))
    return y


def random_crop4(src, clip_range=64):
    h, w = np.size(src, 0), np.size(src, 1)
    width_crop = random.randint(0, clip_range)
    height_crop = random.randint(0, clip_range)
    new_width = w - width_crop
    new_height = h - height_crop
    x = random.randint(0, width_crop)
    y = random.randint(0, height_crop)
    patch = src[y: y+new_height, x: x+new_width, ...]
    patch = cv2.resize(patch, (w, h))
    return patch


def random_crop2(x, clip_width=96):
    ra = random.randint(0, 100)
    t = random.randint(1, clip_width)
    if ra < 25:
        x[-t:, ...] = 0
        mark = 0
    elif ra < 50:
        x[:t, ...] = 0
        mark = 1
    elif ra < 75:
        x[:, -t:, ...] = 0
        mark = 2
    else:
        x[:, :t, ...] = 0
        mark = 3

    return x, mark, t


def random_crop3(x, y, clip_width=96):
    ra = random.randint(0, 100)
    t = random.randint(1, clip_width)
    if ra < 25:
        x[-t:, ...] = 0
        y[-t:, ...] = 0
    elif ra < 50:
        x[:t, ...] = 0
        y[:t, ...] = 0
    elif ra < 75:
        x[:, -t:, ...] = 0
        y[:, -t:, ...] = 0
    else:
        x[:, :t, ...] = 0
        y[:, :t, ...] = 0

    return x, y


'''def random_corp4(src, clip_width=32):
    h, w = np.size(src, 0), np.size(src, 1)
    cur_clip_width = random.randint(4, clip_width)
    x = random.randint(0, w - cur_clip_width)
    y = random.randint(0, h - cur_clip_width)
    crop_x = src.copy()
    crop_x[y: y + cur_clip_width, x: x + cur_clip_width] = 0
    return crop_x'''


def rotate_img(img_list):
    w = np.size(img_list[0], 1)
    h = np.size(img_list[0], 0)
    theta = random.randint(1, 359)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1)
    for i in range(len(img_list)):
        img_list[i] = cv2.warpAffine(img_list[i], M, (w, h))
    return img_list


def rotate_img_rigid(img_list):
    w = np.size(img_list[0], 1)
    h = np.size(img_list[0], 0)
    theta = random.randint(30, 330)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1)
    for i in range(len(img_list)):
        img_list[i] = cv2.warpAffine(img_list[i], M, (w, h))
    return img_list


def trans_img(img, offset, offsetThr):
    cur_ox = random.randint(-offset, offset)
    cur_oy = random.randint(-offset, offset)
    if offsetThr < offset:
        while -offsetThr < cur_ox < offsetThr:
            cur_ox = random.randint(-offset, offset)
        while -offsetThr < cur_oy < offsetThr:
            cur_oy = random.randint(-offset, offset)

    big_img2 = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_CONSTANT)
    #判断img是否为3通道图像
    if len(img.shape) == 1:
        img = big_img2[offset + cur_oy: offset + cur_oy + np.size(img, 0),
              offset + cur_ox: offset + cur_ox + np.size(img, 1)]
    else:
        img = big_img2[offset + cur_oy: offset + cur_oy + np.size(img, 0),
              offset + cur_ox: offset + cur_ox + np.size(img, 1), :]
    return img


def flip_img(img):
    ra = random.randint(0, 100)
    if ra < 25:
        img = cv2.flip(img, 1)
    elif ra < 50:
        img = cv2.flip(img, 0)
    elif ra < 75:
        img = cv2.flip(img, -1)
    return img


def flip_img_list(img_list):
    ra = random.randint(0, 100)
    for i in range(len(img_list)):
        if ra < 25:
            img_list[i] = cv2.flip(img_list[i], 1)
        elif ra < 50:
            img_list[i] = cv2.flip(img_list[i], 0)
        elif ra < 75:
            img_list[i] = cv2.flip(img_list[i], -1)
    return img_list


def flip_img_rigid(img):
    ra = random.randint(0, 75)
    if ra < 25:
        img = cv2.flip(img, 1)
    elif ra < 50:
        img = cv2.flip(img, 0)
    else:
        img = cv2.flip(img, -1)
    return img


def rotate_img90_180_270(img):
    ra = random.randint(0, 100)
    if ra < 25:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif ra < 50:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif ra < 75:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def rotate_img90_180_270_list(img_list):
    ra = random.randint(0, 100)
    for i in range(len(img_list)):
        if ra < 25:
            img_list[i] = cv2.rotate(img_list[i], cv2.ROTATE_90_CLOCKWISE)
        elif ra < 50:
            img_list[i] = cv2.rotate(img_list[i], cv2.ROTATE_180)
        elif ra < 75:
            img_list[i] = cv2.rotate(img_list[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_list


def channel_shuffle(src):
    ra = random.randint(0, 120)
    if ra < 20:
        b, g, r = cv2.split(src)
        src = cv2.merge([r, g, b])
    elif ra < 40:
        b, g, r = cv2.split(src)
        src = cv2.merge([r, b, g])
    elif ra < 60:
        b, g, r = cv2.split(src)
        src = cv2.merge([b, r, g])
    elif ra < 80:
        b, g, r = cv2.split(src)
        src = cv2.merge([g, b, r])
    elif ra < 100:
        b, g, r = cv2.split(src)
        src = cv2.merge([g, r, b])
    return src


def channel_shuffle_rigid(src):
    ra = random.randint(0, 100)
    if ra < 20:
        b, g, r = cv2.split(src)
        src = cv2.merge([r, g, b])
    elif ra < 40:
        b, g, r = cv2.split(src)
        src = cv2.merge([r, b, g])
    elif ra < 60:
        b, g, r = cv2.split(src)
        src = cv2.merge([b, r, g])
    elif ra < 80:
        b, g, r = cv2.split(src)
        src = cv2.merge([g, b, r])
    elif ra <= 100:
        b, g, r = cv2.split(src)
        src = cv2.merge([g, r, b])
    return src


def channel_shuffle_2(src, model):
    ra = random.randint(0, 120)
    if ra < 20:
        b, g, r = cv2.split(src)
        src = cv2.merge([r, g, b])

        b, g, r = cv2.split(model)
        model = cv2.merge([r, g, b])
    elif ra < 40:
        b, g, r = cv2.split(src)
        src = cv2.merge([r, b, g])

        b, g, r = cv2.split(model)
        model = cv2.merge([r, b, g])
    elif ra < 60:
        b, g, r = cv2.split(src)
        src = cv2.merge([b, r, g])

        b, g, r = cv2.split(model)
        model = cv2.merge([b, r, g])
    elif ra < 80:
        b, g, r = cv2.split(src)
        src = cv2.merge([g, b, r])

        b, g, r = cv2.split(model)
        model = cv2.merge([g, b, r])
    elif ra < 100:
        b, g, r = cv2.split(src)
        src = cv2.merge([g, r, b])

        b, g, r = cv2.split(model)
        model = cv2.merge([g, r, b])
    return src, model


def aug_neg_seg(patch, src, p_left, p_top, idx):
    src_width = np.size(src, 1)
    src_height = np.size(src, 0)
    pw = np.size(patch, 1)
    ph = np.size(patch, 0)

    img_norm_param = 1.0 / 255.0

    neg_0 = patch.astype(np.float32)
    if pw == 16:
        mask_frd, mask_bgd = get_mask(pw)
    elif pw == 32:
        mask_frd, mask_bgd = get_mask(pw, 180, 240)
    elif pw > 32:
        mask_frd, mask_bgd = get_mask(pw, 200, 240)
    # cv2.imwrite('H:/CD_WY/log_loc/mask/{}_2.png'.format(i), bin_img)
    mask_bgd = mask_bgd.astype(np.float32) * img_norm_param
    mask_frd = mask_frd.astype(np.float32) * img_norm_param

    s = random.randint(1, 100)
    if s < 10:
        beta = random.uniform(0.5, 1.5)
        while abs(beta - 1.0) < 0.1:
            beta = random.uniform(0.5, 1.5)
        noise = (1.0 - mask_bgd) * beta
        neg = neg_0 * mask_bgd + noise * neg_0
        neg_mask = (1.0 - mask_bgd)
    elif s < 50:
        left3 = random.randint(0, src_width - pw)
        top3 = random.randint(0, src_height - ph)
        while abs(left3 - p_left) < 4 and abs(top3 - p_top) < 4:
            left3 = random.randint(0, src_width - pw)
            top3 = random.randint(0, src_height - ph)
        noise = src[top3: top3 + ph, left3: left3 + pw, :].astype(np.float32)
        neg = neg_0 * (1.0 - mask_frd) + noise * mask_frd
        neg_mask = mask_frd
    elif s < 60:
        noise = np.ones((ph, pw, 3), dtype=np.float32) * random.randint(180, 220)
        neg = neg_0 * (1.0 - mask_frd) + noise * mask_frd
        neg_mask = mask_frd
    elif s < 70:
        noise = np.ones((ph, pw, 3), dtype=np.float32) * random.randint(10, 50)
        neg = neg_0 * (1.0 - mask_frd) + noise * mask_frd
        neg_mask = mask_frd
    else:
        noise = aug_color(patch, off_scale=25.0).astype(np.float32)
        neg = neg_0 * (1.0 - mask_frd) + noise * mask_frd
        neg_mask = mask_frd

    neg = np.clip(neg, 0.0, 255.0).astype(np.uint8)
    neg_mask = (neg_mask*255.0).astype(np.uint8)
    _, neg_mask = cv2.threshold(neg_mask, 128, 1, 0)

    diff = np.abs(neg - patch)
    diff_s = np.sum(diff)
    if pw < 30:
        thr_diff = 5000
    else:
        thr_diff = 8000
    if diff_s < thr_diff:
        neg_mask[:] = 0
        return patch, neg_mask

    return neg, neg_mask


def scale_img(img):
    ra = random.randint(0, 100)
    h, w = np.size(img, 0), np.size(img, 1)
    if ra < 10:
        aug_img = cv2.resize(img, (w // 2, h // 2))
    elif ra < 50:
        aug_img = cv2.resize(img, (w // 4, h // 4))
    elif ra < 90:
        aug_img = cv2.resize(img, (w // 8, h // 8))
    else:
        aug_img = cv2.resize(img, (w // 16, h // 16))
    aug_img = cv2.resize(aug_img, (w, h))
    return aug_img

if __name__ == '__main__':
    img = cv2.imread('E:\\AD_data_set\\log\\visa\\all_TestNet1\\rlt\\44.png', cv2.IMREAD_COLOR)
    for i in range(10):
        img2 = 255 - img
        cv2.imwrite('E:\\AD_data_set\\log\\visa\\all_TestNet1\\{}.png'.format(i), img2)
