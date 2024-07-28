import os.path

import cv2
import numpy as np
import random
import torch
import cv2
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms
import pandas as pd

# import AugImg
import data.AugImg as AugImg
import glob
import data.Augmentor as Augmentor


class LoadImg():
    def __init__(self, file_path, norm_sz_width=128, norm_sz_height=128):
        self.file_path = file_path
        self.norm_sz_width = norm_sz_width
        self.norm_sz_height = norm_sz_height
        split_file = os.path.join(file_path, 'split_csv/1cls.csv')
        split_df = pd.read_csv(split_file)
        # 构造一个字典，key是类别，value是对应的图片路径
        self.train_image_dict = {}
        self.test_image_dict = {}
        # 依次读入split_df中的每一行
        for i in range(len(split_df)):
            # 获取类别
            cur_class = split_df.iloc[i]['object']
            flag = split_df.iloc[i]['split']
            label = split_df.iloc[i]['label']
            # 获取图片路径
            cur_path = split_df.iloc[i]['image']
            mask_path = split_df.iloc[i]['mask']
            # 如果当前类别不在字典中，则新建一个key
            if cur_class not in self.train_image_dict:
                self.train_image_dict[cur_class] = []
                self.test_image_dict[cur_class] = []
            # 将当前图片路径,label,mask路径加入到字典中
            if flag == 'train':
                self.train_image_dict[cur_class].append((cur_path, label, mask_path, cur_class))
            else:
                self.test_image_dict[cur_class].append((cur_path, label, mask_path, cur_class))

        #遍历字典，打印每个类别的训练集和测试集的数量
        for key in self.train_image_dict:
            print(key, 'train', len(self.train_image_dict[key]))
            print(key, 'test', len(self.test_image_dict[key]))

    def get_img(self, value, norm_width=None, norm_height=None):
        img_path = os.path.join(self.file_path, value[0])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if value[1] == 'normal':
            label = 0
            mask = np.zeros((img.shape[0], img.shape[1]))
        else:
            label = 1
            mask_path = os.path.join(self.file_path, value[2])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if norm_width is not None and norm_height is not None:
            img = cv2.resize(img, (norm_width, norm_height))
            mask = cv2.resize(mask, (norm_width, norm_height), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (self.norm_sz_width, self.norm_sz_height))
            mask = cv2.resize(mask, (self.norm_sz_width, self.norm_sz_height), interpolation=cv2.INTER_NEAREST)
        # mask中非0的像素值都设置为1
        mask[mask != 0] = 1
        return img, label, mask, value[3]


class visaDLoaderTest(Dataset):
    def __init__(self, data_set, class_name):
        self.data_set = data_set
        self.image_value = []
        if class_name == 'all':
            for key in self.data_set.test_image_dict:
                for i in range(len(self.data_set.test_image_dict[key])):
                    self.image_value.append(self.data_set.test_image_dict[key][i])
        else:
            for i in range(len(self.data_set.test_image_dict[class_name])):
                self.image_value.append(self.data_set.test_image_dict[class_name][i])
        print('test {}: num: {}'.format(class_name, len(self.image_value)))
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
                        # transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.data_set.norm_sz_width),
                        transforms.Normalize(mean=mean_train, std=std_train)])
    
        class_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                  'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                  'pcb4', 'pipe_fryum']
        self.class_map = {}
        for i in range(len(class_list)):
            self.class_map[class_list[i]] = i

    def __len__(self):
        return len(self.image_value)

    def __getitem__(self, idx):
        cur_value = self.image_value[idx]
        Patch_img, label, mask, cls_name = self.data_set.get_img(cur_value)

        Patch_img = cv2.cvtColor(Patch_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        Patch_img = self.transform(Patch_img)  
        # Patch_img = torch.from_numpy(Patch_img).permute(2, 0, 1)

        mask = torch.from_numpy(mask).unsqueeze(0)
        return Patch_img, mask, label


class visaDLoaderTrainNew2(Dataset):
    def __init__(self, data_set, class_name, scale):
        self.data_set = data_set
        self.image_value = []
        if class_name == 'all':
            for key in self.data_set.train_image_dict:
                for i in range(len(self.data_set.train_image_dict[key])):
                    self.image_value.append(self.data_set.train_image_dict[key][i])
        else:
            for i in range(len(self.data_set.train_image_dict[class_name])):
                self.image_value.append(self.data_set.train_image_dict[class_name][i])
        print('Train {}: num: {}'.format(class_name, len(self.image_value)))
        self.idx = 0
        self.scale = scale

        opt1 = ['patch_shuffle', 'gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'rand_quantize']
        opt1_preb = [1, 1, 1, 1, 1, 1]
        opt2 = ['gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'rand_quantize']
        opt2_preb = [1, 1, 1, 1, 1]
        '''opt1 = ['gaussian_noise', 'PerlinMerge']
        opt1_preb = [1, 1]
        opt2 = ['gaussian_noise', 'PerlinMerge']
        opt2_preb = [1, 1]'''
        self.augoptor = Augmentor.AugOptorPreb(opt1, opt1_preb)
        self.augoptor2 = Augmentor.AugOptorPreb(opt2, opt2_preb)
        # self.augoptor = Augmentor.AugOptor()
        # self.aug_perlin = Augmentor.AugOptor(['PerlinMerge'])

        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
                        # transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.data_set.norm_sz_width),
                        transforms.Normalize(mean=mean_train, std=std_train)])
        class_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                  'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                  'pcb4', 'pipe_fryum']
        self.class_map = {}
        for i in range(len(class_list)):
            self.class_map[class_list[i]] = i

    def __len__(self):
        return int(self.scale * len(self.image_value))

    def __getitem__(self, idx):
        c = idx // len(self.image_value)
        cur_idx = idx - c * len(self.image_value)
        cur_value = self.image_value[cur_idx]
        Patch_img, _, _, cls_name = self.data_set.get_img(cur_value)

        # Aug_img = self.augoptor.opt_img(Patch_img)  
        ra = random.randint(0, 100)
        if ra < 90:
            # noise_img_idx = random.randint(0, len(self.image_value) - 1)  
            # noise_img, _, _, _ = self.data_set.get_img(self.image_value[noise_img_idx]) 
            if cls_name == 'capsules' or cls_name == 'macaroni2':
                Aug_img = self.augoptor2.opt_img(Patch_img, None) 
            else:
                Aug_img = self.augoptor.opt_img(Patch_img, None) 
            # local_noise_img = self.aug_perlin.opt_img(Patch_img, noise_img)
        else:
            Aug_img = Patch_img.copy()
        '''cv2.imwrite('/home/primary/data1/Visa/log2/rlt/{}.png'.format(self.idx), Patch_img)
        cv2.imwrite('/home/primary/data1/Visa/log2/rlt/{}_aug.png'.format(self.idx), Aug_img)
        self.idx += 1'''
        # local_noise_img = cv2.cvtColor(local_noise_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        Patch_img = cv2.cvtColor(Patch_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        Aug_img = cv2.cvtColor(Aug_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # local_noise_img = self.transform(local_noise_img)
        Patch_img = self.transform(Patch_img)  
        Aug_img = self.transform(Aug_img) 
        # Patch_img = torch.from_numpy(Patch_img).permute(2, 0, 1)
        # Aug_img = torch.from_numpy(Aug_img).permute(2, 0, 1)
        # cls_id = self.class_map[cls_name]
        return Patch_img, Aug_img
    


    