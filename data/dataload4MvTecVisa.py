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

import data.dataload4visa as dl_vi
import data.dataload4MvTec as dl_mv
from data.dataset import MVTecDataset
from data.dataset import get_data_transforms

class LoadImg():
    def __init__(self, file_path_mv, file_path_vi, norm_sz_width=128, norm_sz_height=128):
        self.img_w = norm_sz_width
        self.img_h = norm_sz_height
        self.vi_class_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                              'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        self.mv_class_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid','hazelnut', 
                              'leather', 'metal_nut', 'pill', 'screw','tile', 'toothbrush', 'transistor', 'zipper', 'wood']
        self.vi_Patch_list = dl_vi.LoadImg(file_path=file_path_vi, norm_sz_width=norm_sz_width, norm_sz_height=norm_sz_height)
        self.mv_train_Patch_list = dl_mv.LoadImg(file_path=file_path_mv, class_list=self.mv_class_list, flag='train',
                                            norm_sz_width=norm_sz_width, norm_sz_height=norm_sz_height)
        
        vi_test_loader_list = []
        for i in range(len(self.vi_class_list)):
            ds_test = dl_vi.visaDLoaderTest(data_set=self.vi_Patch_list, class_name=self.vi_class_list[i])
            testLoader = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)
            vi_test_loader_list.append(testLoader)
        
        mv_test_loader_list = []
        data_transform, gt_transform = get_data_transforms(norm_sz_width, norm_sz_height)
        for cls_name in self.mv_class_list:
            test_path = file_path_mv + cls_name
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
            mv_test_loader_list.append(test_dataloader)
        
        self.test_loader_list = vi_test_loader_list + mv_test_loader_list
        self.test_class_list = self.vi_class_list + self.mv_class_list


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
                        transforms.CenterCrop(self.data_set.img_w),
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


class mvviDLoaderTrainNew2(Dataset):
    def __init__(self, data_set, scale):
        self.vi_mv_data_set = data_set

        self.vi_image_value = []
        for key in self.vi_mv_data_set.vi_Patch_list.train_image_dict:
            for i in range(len(self.vi_mv_data_set.vi_Patch_list.train_image_dict[key])):
                self.vi_image_value.append(self.vi_mv_data_set.vi_Patch_list.train_image_dict[key][i])
        print('Train vi num: {}'.format(len(self.vi_image_value)))


        self.mv_image_idx = self.vi_mv_data_set.mv_train_Patch_list.get_img_list(['all'])
        print('Train mv num: {}'.format(len(self.mv_image_idx)))

        self.idx = 0
        self.scale = scale

        self.sum_num = len(self.mv_image_idx) + len(self.vi_image_value)

        opt1 = ['patch_shuffle', 'gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'rand_quantize']
        opt1_preb = [1, 1, 1, 1, 1, 1]
        # opt2 = ['gaussian_noise', 'color_jitter', 'rescale', 'sobel', 'rand_quantize']
        # opt2_preb = [1, 1, 1, 1, 1]
        '''opt1 = ['gaussian_noise', 'PerlinMerge']
        opt1_preb = [1, 1]
        opt2 = ['gaussian_noise', 'PerlinMerge']
        opt2_preb = [1, 1]'''
        self.augoptor = Augmentor.AugOptorPreb(opt1, opt1_preb)
        # self.augoptor2 = Augmentor.AugOptorPreb(opt2, opt2_preb)
        # self.augoptor = Augmentor.AugOptor()
        # self.aug_perlin = Augmentor.AugOptor(['PerlinMerge'])

        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
                        # transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.vi_mv_data_set.img_w),
                        transforms.Normalize(mean=mean_train, std=std_train)])

    def __len__(self):
        return int(self.scale * self.sum_num)
    
    def get_image(self, cur_idx):
        if cur_idx >= len(self.vi_image_value): # mv image
            cur_img_idx = self.mv_image_idx[cur_idx - len(self.vi_image_value)]
            Patch_img = self.vi_mv_data_set.mv_train_Patch_list.image_list[cur_img_idx].copy()
        else: # vi image
            cur_value = self.vi_image_value[cur_idx]
            Patch_img, _, _, _ = self.vi_mv_data_set.vi_Patch_list.get_img(cur_value)
        return Patch_img

    def __getitem__(self, idx):
        c = idx // self.sum_num
        cur_idx = idx - c * self.sum_num

        Patch_img = self.get_image(cur_idx)

        # Aug_img = self.augoptor.opt_img(Patch_img)  
        ra = random.randint(0, 100)
        if ra < 90:
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
        return Patch_img, Aug_img
    
