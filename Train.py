# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from data.dataset import get_data_transforms

import numpy as np

from net_model.network import UniRD

from utils.test import evaluation
from torch.nn import functional as F
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(image_size, batch_size, data_set_name, backbone='resnet18'):
    setup_seed(1024)
    # data_set_name = 'Mvtec'
    if data_set_name == 'VisA':
        from data import dataload4visa as dl
    elif data_set_name == 'MVTec':
        from data import dataload4MvTec as dl
        from data.dataset import MVTecDataset
    elif data_set_name == 'MvtecVisa':
        from data import dataload4MvTecVisa as dl

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _class_ = data_set_name
    print(_class_)
    epochs = 300
    # batch_size = 256

    if data_set_name == 'VisA':
        class_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                      'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                      'pcb4', 'pipe_fryum']
        log_path = 'log/VisA/'
        data_path = 'dataset/VisA/'
    elif data_set_name == 'MVTec':
        class_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                      'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                      'tile', 'toothbrush', 'transistor', 'zipper', 'wood']
        log_path = 'log/MVTec/'
        data_path = 'dataset/MVTec/'
    elif data_set_name == 'MVTecVisA':
        log_path = 'log/Fuse/'
        mv_data_path = 'dataset/MVTec/'
        vi_data_path = 'dataset/VisA/'

    ckp_path = log_path + _class_ + '_{}_{}.pth'.format(backbone, image_size)
    best_ckp_path = log_path + _class_ + '_{}_{}_best.pth'.format(backbone, image_size)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    loss_path = log_path + _class_ + '_{}_{}.csv'.format(backbone, image_size)
    if not os.path.exists(loss_path):
        fr = open(loss_path, 'w')
        fr.write(' , epoch, pix_auc, img_auc, au_pro, f1, mAP\n')
        fr.close()

    if data_set_name == 'VisA':
        Patch_list = dl.LoadImg(file_path=data_path, norm_sz_width=image_size, norm_sz_height=image_size)
        trainning_cls = class_list
        ds_train = dl.visaDLoaderTrainNew2(data_set=Patch_list, class_name='all', scale=1.0)
        test_loader_list = []
        for i in range(len(trainning_cls)):
            ds_test = dl.visaDLoaderTest(data_set=Patch_list, class_name=trainning_cls[i])
            testLoader = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)
            test_loader_list.append(testLoader)
    elif data_set_name == 'MVTec':
        data_transform, gt_transform = get_data_transforms(image_size, image_size)
        Patch_list_train = dl.LoadImg(file_path=data_path, class_list=class_list, flag='train',
                                      norm_sz_width=image_size, norm_sz_height=image_size)

        ds_train = dl.mvTecADLoaderTrainAug(data_set=Patch_list_train, class_name=['all'])

        test_loader_list = []
        for cls_name in class_list:
            test_path = data_path + cls_name
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
            test_loader_list.append(test_dataloader)
    elif data_set_name == 'MVTecVisA':
        data_set = dl.LoadImg(file_path_mv=mv_data_path, file_path_vi=vi_data_path, norm_sz_width=image_size, norm_sz_height=image_size)
        ds_train = dl.mvviDLoaderTrainNew2(data_set=data_set, scale=1.0)

        test_loader_list = data_set.test_loader_list
        class_list = data_set.test_class_list

    train_dataloader = torch.utils.data.DataLoader(ds_train, pin_memory=True, batch_size=batch_size, shuffle=True,
                                                   num_workers=8)

    if backbone == 'resnet18':
        model = UniRD(backbone='resnet18').to(device)
    elif backbone == 'wide_resnet50':
        model = UniRD(backbone='wide_resnet50').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    schuduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    globel_step = 0
    mean_auroc_px = 0
    mean_auroc_sp = 0
    test_step = 20

    for epoch in range(epochs):
        model.train()
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        loss_list = []
        batch_idx = 0

        for img, aug in train_dataloader:
            img = img.to(device)
            aug = aug.to(device)
            src_list, de_aug_list, de_src_list, src_proj_list = model(img, aug)
            student_loss = 0
            teacher_loss = 0
            student_loss_l1 = 0
            teacher_loss_l1 = 0
            for i in range(len(src_proj_list)):
                student_loss += (1.0 - F.cosine_similarity(src_list[i].detach(), de_aug_list[i])).mean()
                teacher_loss += (1.0 - F.cosine_similarity(de_src_list[i].detach(), src_proj_list[i])).mean()
                student_loss_l1 += F.l1_loss(src_list[i].detach(), de_aug_list[i])
                teacher_loss_l1 += F.l1_loss(de_src_list[i].detach(), src_proj_list[i])
            student_loss = student_loss / len(src_list)
            teacher_loss = teacher_loss / len(src_proj_list)
            student_loss_l1 = student_loss_l1 / len(src_list)
            teacher_loss_l1 = teacher_loss_l1 / len(src_proj_list)
            loss = teacher_loss + student_loss + 0.1 * (student_loss_l1 + teacher_loss_l1)
            # writer.add_scalar('dt_loss', loss, global_step=globel_step)
            globel_step += 1
            if batch_idx % 10 == 0:
                print(
                    'epoch: {} [{}/{}], student_loss:{:.4f}, teacher_loss: {}, student_loss_l1: {}, teacher_loss_l1:{}, imgsize: {}x{}'.format(
                        epoch, batch_idx, len(train_dataloader), student_loss.item(), teacher_loss.item(),
                        student_loss_l1.item(), teacher_loss_l1.item(),
                        img.size(2), img.size(3)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # if batch_idx % 20 == 0:
            #    print('epoch [{}/{}], dataloader: {}/{} loss:{:.4f} loss:{:.4f}'.format(epoch + 1, epochs, batch_idx, len(train_dataloader), loss_src.item(), loss_aug.item()))
            batch_idx += 1
        # writer.add_scalar('loss', np.mean(loss_list), global_step=epoch)
        schuduler.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % test_step == 0:
            torch.save(model.state_dict(), ckp_path)
            test_step = 10
            fr = open(loss_path, 'a')
            auroc_px_list = []
            auroc_sp_list = []
            cur_mean_auroc_px = 0
            cur_mean_auroc_sp = 0
            cur_mean_f1 = 0
            cur_mean_AP = 0
            for i in range(len(class_list)):
                auroc_px, auroc_sp, f1 = evaluation(model, test_loader_list[i], device)
                auroc_px_list.append(auroc_px)
                auroc_sp_list.append(auroc_sp)
                cur_mean_auroc_px += auroc_px
                cur_mean_auroc_sp += auroc_sp
                cur_mean_f1 += f1
                print('cls: {} Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, f1{:.3}'.format(class_list[i], auroc_px, auroc_sp, f1))
                fr.write(
                    '{}, {}, {}, {}, {}, {}, {}\n'.format(class_list[i], epoch, auroc_px, auroc_sp, 0.0, f1, 0.0))
            cur_mean_auroc_px /= len(class_list)
            cur_mean_auroc_sp /= len(class_list)
            cur_mean_f1 /= len(class_list)
            cur_mean_AP /= len(class_list)
            if cur_mean_auroc_px + cur_mean_auroc_sp > mean_auroc_px + mean_auroc_sp:
                mean_auroc_px = cur_mean_auroc_px
                mean_auroc_sp = cur_mean_auroc_sp
                # fr.write('++++++epoch: {} mean Pixel Auroc:{:.3f}, mean Sample Auroc{:.3f}, this is best one\n'.format(epoch, mean_auroc_px, mean_auroc_sp))
                fr.write(
                    '{}, {}, {}, {}, {}, {}, {}\n'.format('best_mean', epoch, mean_auroc_px, mean_auroc_sp, 0.0,
                                                          cur_mean_f1, 0.0))
                torch.save(model.state_dict(), best_ckp_path)
            else:
                fr.write(
                    '{}, {}, {}, {}, {}, {}, {}\n'.format('', epoch, cur_mean_auroc_px, cur_mean_auroc_sp, 0.0,
                                                          cur_mean_f1, 0.0))
            fr.close()


if __name__ == '__main__':
    # train(image_size=256, batch_size=64, data_set_name='MVTec', backbone='wide_resnet50')
    # train(image_size=256, batch_size=256, data_set_name='MVTec', backbone='resnet18')

    # train(image_size=256, batch_size=64, data_set_name='VisA', backbone='wide_resnet50')
    train(image_size=256, batch_size=64, data_set_name='MVTec', backbone='wide_resnet50')

