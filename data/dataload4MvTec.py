import os.path

import cv2
import numpy as np
import random
import torch
import cv2
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from data.Augmentor import AugOptorPreb

# import AugImg
import glob
from numpy.random import shuffle
# from data.noise import Simplex_CLASS
from torchvision import transforms

class LoadImg():
    def __init__(self, file_path, class_list, flag, norm_sz_width=128, norm_sz_height=128):
        self.flag = flag
        self.image_list = []
        self.image_class = []
        self.test_lable_list = []
        self.test_gt_list = []
        self.patch_width = norm_sz_width
        self.patch_height = norm_sz_height
        self.idx = 0

        image_catch = os.path.join(file_path, 'image_catch_{}_{}_{}_BGR.npy'.format(self.flag, self.patch_width,
                                                                                    self.patch_height))
        class_catch = os.path.join(file_path, 'class_catch_{}_{}_{}_BGR.npy'.format(self.flag, self.patch_width,
                                                                                    self.patch_height))
        gt_catch = os.path.join(file_path,
                                'gt_catch_{}_{}_{}_BGR.npy'.format(self.flag, self.patch_width, self.patch_height))
        label_catch = os.path.join(file_path, 'label_catch_{}_{}_{}_BGR.npy'.format(self.flag, self.patch_width,
                                                                                    self.patch_height))
        print(image_catch, class_catch)
        if os.path.exists(image_catch) and os.path.exists(class_catch):
            self.image_list = np.load(image_catch, allow_pickle=True)
            self.image_class = np.load(class_catch, allow_pickle=True)
            if self.flag == 'test':
                self.test_gt_list = np.load(gt_catch, allow_pickle=True)
                self.test_lable_list = np.load(label_catch, allow_pickle=True)
        else:
            print('error: no image catch file, please check the path: {}'.format(image_catch))
            if self.flag == 'train':
                for i in range(len(class_list)):
                    img_path = os.path.join(file_path, class_list[i], 'train/good')
                    img_list_png = glob.glob(os.path.join(img_path, '*.png'))
                    img_list_jpg = glob.glob(os.path.join(img_path, '*.jpg'))
                    img_list_bmp = glob.glob(os.path.join(img_path, '*.bmp'))
                    img_list = img_list_png + img_list_jpg + img_list_bmp
                    # load image
                    for j in range(len(img_list)):
                        img = cv2.imdecode(np.fromfile(img_list[j], dtype=np.uint8), cv2.IMREAD_COLOR)
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.patch_width, self.patch_height))

                        self.image_list.append(img)
                        self.image_class.append(class_list[i])

                        if j % 1000 == 0:
                            print('{}, load {}/{} images'.format(class_list[i], j, len(img_list)))
            else:
                test_blob_zeros = 0
                for i in range(len(class_list)):
                    root_path = os.path.join(file_path, class_list[i], 'test')
                    # 递归遍历root_path下所有bmp\png\jpg文件,如果有嵌套文件夹，也会遍历
                    file_list = []
                    for root, dirs, files in os.walk(root_path):
                        for file in files:
                            file_list.append(os.path.join(root, file))
                    for j in range(len(file_list)):
                        if file_list[j].find('.png') == -1 and file_list[j].find('.jpg') == -1 and file_list[j].find(
                                '.bmp') == -1:
                            continue
                        img = cv2.imdecode(np.fromfile(file_list[j], dtype=np.uint8), cv2.IMREAD_COLOR)
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.patch_width, self.patch_height))

                        blob_path = file_list[j].replace('test', 'ground_truth')
                        blob_path = os.path.join(blob_path[:-4] + '_mask.png')
                        if not os.path.exists(blob_path):
                            blob_path = file_list[j].replace('test', 'ground_truth')
                            blob_path = os.path.join(blob_path[:-4] + '.png')
                        if file_list[j].find('good') != -1:
                            blob = np.zeros((self.patch_height, self.patch_width), dtype=np.uint8)
                            self.test_lable_list.append(0)
                        else:
                            blob = cv2.imdecode(np.fromfile(blob_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                            blob = cv2.resize(blob, (self.patch_width, self.patch_height),
                                              interpolation=cv2.INTER_NEAREST)
                            blob = np.where(blob > 0, 1, 0).astype(np.uint8)
                            if blob.max() == 0:
                                # print('blob max is 0, path: {}'.format(blob_path))
                                test_blob_zeros += 1
                                self.test_lable_list.append(0)
                            else:
                                self.test_lable_list.append(1)
                        self.image_list.append(img)
                        self.image_class.append(class_list[i])
                        self.test_gt_list.append(blob)
                        if j % 1000 == 0:
                            print('{}, load {}/{} images'.format(class_list[i], j, len(file_list)))
                print('test_blob_zeros: {}'.format(test_blob_zeros))
            np.save(image_catch, self.image_list)
            np.save(class_catch, self.image_class)
            np.save(gt_catch, self.test_gt_list)
            np.save(label_catch, self.test_lable_list)

        class_num = []
        for i in range(len(class_list)):
            class_num.append(0)
        for i in range(len(self.image_class)):
            for j in range(len(class_list)):
                if self.image_class[i] == class_list[j]:
                    class_num[j] += 1
                    break

        for i in range(len(class_list)):
            print('flag: {} patch size: {} {}: {}'.format(self.flag, self.patch_width, class_list[i], class_num[i]))

    def get_img_list(self, class_name):
        train_list = []
        if class_name[0] == 'all':
            for i in range(len(self.image_class)):
                train_list.append(i)
            return train_list
        else:
            for i in range(len(class_name)):
                count = 0    
                for j in range(len(self.image_class)):
                    if self.image_class[j] == class_name[i]:
                        train_list.append(j)
                        count += 1
                print('mvtec class_name: {}, flag: {}, num: {}'.format(class_name[i], self.flag, count))
            return train_list


class mvTecADLoaderTrain(Dataset):
    def __init__(self, data_set, class_name):
        self.data_set = data_set
        self.image_idx = self.data_set.get_img_list(class_name)
        print('{} {}: num: {}'.format('train', class_name, len(self.image_idx)))
        self.idx = 0
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
                        # transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.data_set.patch_width),
                        transforms.Normalize(mean=mean_train, std=std_train)])
        
    def __len__(self):
        return len(self.image_idx)

    def __getitem__(self, idx):
        cur_img_idx = self.image_idx[idx]
        Patch_img = self.data_set.image_list[cur_img_idx].copy()
        Patch_img = cv2.cvtColor(Patch_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        Patch_img = self.transform(Patch_img)   
        return Patch_img
        

class mvTecADLoaderTrainAug(Dataset):
    def __init__(self, data_set, class_name):
        self.data_set = data_set
        self.image_idx = self.data_set.get_img_list(class_name)
        print('{} {}: num: {}'.format('train', class_name, len(self.image_idx)))
        self.idx = 0
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
                        # transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.data_set.patch_width),
                        transforms.Normalize(mean=mean_train, std=std_train)])
        
        opt1 = ['patch_shuffle', 'gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'rand_quantize']
        opt1_preb = [1, 1, 1, 1, 1, 1]
        # opt1 = ['gaussian_noise', 'PerlinMerge']
        # opt1_preb = [1, 1]
        self.augoptor = AugOptorPreb(opt1, opt1_preb)
        self.label_map = {}
        label_num = 0
        for i in range(len(self.data_set.image_class)):
            if self.data_set.image_class[i] not in self.label_map:
                self.label_map[self.data_set.image_class[i]] = label_num
                label_num+=1
        print(self.label_map)
        
        
    def __len__(self):
        return len(self.image_idx)

    def __getitem__(self, idx):
        cur_img_idx = self.image_idx[idx]
        Patch_img = self.data_set.image_list[cur_img_idx].copy()
        class_name = self.data_set.image_class[cur_img_idx]
        if self.data_set.flag == 'train':
            ra = random.randint(0, 100)
            if ra < 90:
                '''noise_img_idx = random.randint(0, len(self.image_idx) - 1)
                noise_img_idx = self.image_idx[noise_img_idx]
                noise_img = self.data_set.image_list[noise_img_idx].copy()'''
                Aug_img = self.augoptor.opt_img(Patch_img, None)  # Aug_img
            else:
                Aug_img = Patch_img.copy()
            Patch_img = cv2.cvtColor(Patch_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            Aug_img = cv2.cvtColor(Aug_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            Patch_img = self.transform(Patch_img)
            Aug_img = self.transform(Aug_img)
            # obj_label = self.label_map[class_name]
            return Patch_img, Aug_img
        else:
            Patch_img = cv2.cvtColor(Patch_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            Patch_img = self.transform(Patch_img)

            gt = self.data_set.test_gt_list[cur_img_idx].copy()
            gt = torch.from_numpy(gt).unsqueeze(0)
            label = torch.tensor(self.data_set.test_lable_list[cur_img_idx])

            return Patch_img, gt, label