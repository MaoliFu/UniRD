
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from data.dataset import get_data_transforms
from net_model.network import UniRD
from utils.test import evaluation_array
from utils.eval_helper import performances


def eval_fun(backbone, data_set_name, image_size):
    model_name = backbone
    if data_set_name == 'VisA':
        from data import dataload4visa as dl
    elif data_set_name == 'MVTec':
        from data import dataload4MvTec as dl
        from data.dataset import MVTecDataset
    elif data_set_name == 'MVTecVisA':
        from data import dataload4MvTecVisa as dl

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _class_ = data_set_name
    print(_class_)

    if data_set_name == 'VisA':
        class_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                      'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                      'pcb4', 'pipe_fryum']
        log_path = 'log/VisA/'
        # data_path = '/home/primary/data1/Visa/data'
        data_path = 'dataset/VisA/'
    elif data_set_name == 'MVTec':
        class_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                      'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                      'tile', 'toothbrush', 'transistor', 'zipper', 'wood']
        log_path = 'log/MVTec/'
        # data_path = '/home/primary/data1/mvTec/data/'
        data_path = 'dataset/MVTec/'
    elif data_set_name == 'MVTecVisA':
        log_path = 'log/Fuse/'
        mv_data_path = 'dataset/MVTec/'
        vi_data_path = 'dataset/VisA/'
        '''mv_data_path = '/home/primary/data1/mvTec/data/'
        vi_data_path = '/home/primary/data1/Visa/data'''

    # ckp_path = log_path + _class_ + '_{}_{}_{}_{}.pth'.format(model_name, drop_ration, image_size, train_idx)
    best_ckp_path = log_path + _class_ + '_{}_{}_best.pth'.format(backbone, image_size)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    loss_path = log_path + _class_ + '_{}_{}_eval.csv'.format(model_name, image_size)
    if not os.path.exists(loss_path):
        fr = open(loss_path, 'w')
        fr.write(' , max_auc, pixel_auc, pro_auc, appx_auc, apsp_auc, f1px_auc, f1sp_auc\n')
        fr.close()
    metric_list = ['max_auc', 'pixel_auc', 'pro_auc', 'appx_auc', 'apsp_auc', 'f1px_auc', 'f1sp_auc']

    if data_set_name == 'VisA':
        Patch_list = dl.LoadImg(file_path=data_path, norm_sz_width=image_size, norm_sz_height=image_size)
        trainning_cls = class_list
        # train_dataloader = torch.utils.data.DataLoader(ds_train, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader_list = []
        for i in range(len(trainning_cls)):
            ds_test = dl.visaDLoaderTest(data_set=Patch_list, class_name=trainning_cls[i])
            testLoader = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)
            test_loader_list.append(testLoader)
    elif data_set_name == 'MVTec':
        data_transform, gt_transform = get_data_transforms(image_size, image_size)

        test_loader_list = []
        for cls_name in class_list:
            test_path = data_path + cls_name
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
            test_loader_list.append(test_dataloader)
    elif data_set_name == 'MVTecVisA':
        data_set = dl.LoadImg(file_path_mv=mv_data_path, file_path_vi=vi_data_path, norm_sz_width=image_size,
                              norm_sz_height=image_size)

        test_loader_list = data_set.test_loader_list
        class_list = data_set.test_class_list

    model = UniRD(backbone=model_name).to(device)

    evl_metrics = {'auc': [{'name': 'max'}, {'name': 'pixel'}, {'name': 'pro'}, {'name': 'appx'}, {'name': 'apsp'},
                           {'name': 'f1px'}, {'name': 'f1sp'}]}

    if os.path.exists(best_ckp_path):
        model.load_state_dict(torch.load(best_ckp_path), strict=True)
        # model.load_state_dict(torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda(local_rank)), strict=False)
        print('load param from ckp')
        fr = open(loss_path, 'a')

        rlt_dict = {}
        for i in range(len(class_list)):
            scores, gts = evaluation_array(model, test_loader_list[i], device, _class_)
            rlt_dict[class_list[i]] = {'preds': scores, 'mask': gts}

        ret_metrics = performances(class_list, rlt_dict, evl_metrics)
        for i in range(len(class_list)):
            value_list = []
            for me in metric_list:
                value_list.append(ret_metrics['{}_{}'.format(class_list[i], me)])
            fr.write(
                '{}, {}, {}, {}, {}, {}, {}, {}\n'.format(class_list[i], value_list[0], value_list[1], value_list[2],
                                                          value_list[3],
                                                          value_list[4], value_list[5], value_list[6]))
        value_list = []
        for me in metric_list:
            value_list.append(ret_metrics['mean_{}'.format(me)])
        fr.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format('mean', value_list[0], value_list[1], value_list[2],
                                                           value_list[3],
                                                           value_list[4], value_list[5], value_list[6]))
        fr.close()
    return


if __name__ == '__main__':
    
    # eval_fun(backbone='wide_resnet50', data_set_name='MVTec', image_size=256)
    eval_fun(backbone='resnet18', data_set_name='MVTec', image_size=256)
    eval_fun(backbone='wide_resnet50', data_set_name='VisA', image_size=256)
    eval_fun(backbone='resnet18', data_set_name='VisA', image_size=256)
    # eval_fun(backbone='wide_resnet50', data_set_name='MVTecVisA', image_size=256)

