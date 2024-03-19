import argparse
from glob import glob
import cv2
import os
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import KFold
import time
from sklearn.metrics import roc_auc_score
from metric import *
from data_utils import *
from Grad_cam import GradCam

model_dict = {
                'resnet18':    {'model':models.resnet18,    'weights': models.ResNet18_Weights.DEFAULT},
                'resnet34':    {'model':models.resnet34,    'weights': models.ResNet34_Weights.DEFAULT},
                'resnet50':    {'model':models.resnet50,    'weights': models.ResNet50_Weights.DEFAULT},
                'vgg16':       {'model':models.vgg16,       'weights': models.VGG16_Weights.DEFAULT},
                'densenet121': {'model':models.densenet121, 'weights': models.DenseNet121_Weights.DEFAULT}
            }
model_list = ['resnet18','resnet34','resnet50','vgg16','densenet121']

def compute_centroid(tensor):
    x_sum = torch.sum(torch.sum(tensor, dim=0) * torch.arange(tensor.shape[0]).cuda())
    y_sum = torch.sum(torch.sum(tensor, dim=1) * torch.arange(tensor.shape[1]).cuda())
    total_sum = torch.sum(tensor)

    center_x = x_sum / total_sum
    center_y = y_sum / total_sum

    center_x = torch.unsqueeze(center_x, dim=0)
    center_y = torch.unsqueeze(center_y, dim=0)
    center = torch.cat((center_x, center_y), dim=0)

    return  center

def preprocess_attn(attn_path):
    input_attn =  cv2.resize(cv2.imread(os.path.join('./dataset/label', attn_path)), (224, 224), interpolation=cv2.INTER_AREA)[:, :, 0:1]
    input_attn = np.array(input_attn, dtype=np.float32).transpose((2, 0, 1)) / 255.0
    
    preprocessed_img = torch.from_numpy(input_attn)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def show_cam_on_image(img, mask, path, file_name):
    save_path = os.path.join(path, file_name)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_BONE)
    heatmap = np.float32(heatmap) / 255
    cv2.imwrite(save_path, np.uint8(255 * heatmap))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50',choices=model_list, help='type of model')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--n_epoch', type=int, default=50,
                        help='Number of epoch to run')
    parser.add_argument('--data_dir', default='dataset', type=str)
    parser.add_argument('--save_path', type=str, default='./model/TLS/',
                        help='The address for storing the models.')
    parser.add_argument('--load_path', type=str, default='./model/baseline/',
                        help='The address for the base models.')
    parser.add_argument('--train-batch', default=20, type=int, metavar='N',
                        help='train batchsize (default: 256)')
    parser.add_argument('--test-batch', default=20, type=int, metavar='N',
                        help='test batchsize (default: 200)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-e', '--evaluate', default=False, type=bool, 
                        help='evaluate model on validation set')
    parser.add_argument('--trainWithMap', default=False, type=bool,
                        help='train with attention map')
    parser.add_argument('--baseline', action='store_true',
                        help='Whether baseline or fine-tuned')
    parser.add_argument('--alpha', default=0.8, type=float,
                        help='weight for balance between diag loss and attention loss')
    parser.add_argument('--beta', default=10, type=float,
                        help='weight for balance between diag loss and attention loss')
    parser.add_argument('--gamma', default=1, type=float,
                        help='weight for balance between diag loss and attention loss')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()



    return args

args = get_args()


def model_test(model, test_loader, output_attention= False, output_iou = False, output_reasonbality = None, n_fold=0):
    print('start testing')
    # model.eval()
    iou = AverageMeter()
    kl = AverageMeter()
    cc = AverageMeter()
    sim = AverageMeter()
    nss = AverageMeter()
    ious = {}
    st = time.time()
    outputs_all = []
    targets_all = []
    img_fns = []

    # load grad_cam module

    #resnet
    if args.model == 'resnet18' or args.model == 'resnet34':
        grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["1"], use_cuda=args.use_cuda)        
    elif args.model == 'resnet50':
        grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)   
    #vgg
    elif args.model == 'vgg16':
        grad_cam = GradCam(model=model, feature_module=model.features, \
                        target_layer_names=["30"], use_cuda=args.use_cuda)
    #densenet
    elif args.model == 'densenet121':
        grad_cam = GradCam(model=model, feature_module=model.features, \
                        target_layer_names=["denseblock4"], use_cuda=args.use_cuda)

    y_true = np.array([])
    y_pred = np.array([])
    misclassified = np.array([])

    pred_iou = np.array([])
    pred_kld = np.array([])

    for batch_idx, (inputs, targets, attn_maps, paths) in enumerate(test_loader):
        y_true = np.append(y_true, targets)
        misclassified = np.append(misclassified, paths)
        if args.use_cuda:
            inputs, targets, attn_maps = inputs.cuda(), targets.cuda(non_blocking=True), attn_maps.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            y_pred = np.append(y_pred, predicted)
        if output_attention:
            for img_path in paths:
                _, img_fn = os.path.split(img_path)
                img_fns.append(img_fn)

                img = cv2.imread(img_path, 1)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                input = preprocess_image(img)

                mask = grad_cam(input)
                cam_attn = mask.copy()
                if  n_fold == 4:
                    show_cam_on_image(img, cam_attn, os.path.join(args.load_path, f'attentions/'), img_fn)

                if output_iou and img_fn in path_to_attn:

                    threshold = np.quantile(mask,0.9)
                    item_att_binary = (mask > threshold)
                    target_att = path_to_attn[img_fn]
                    target_att_binary = (target_att > 0)
                    single_iou = compute_iou(item_att_binary, target_att_binary)
                    pred_iou = np.append(pred_iou, single_iou.item())
                    iou.update(single_iou.item(), 1)
                    ious[img_fn]=single_iou.item()
                    kl.update(KLdiv(np.array(mask, dtype='float32'), np.array(target_att, dtype='float32')), 1)
                    pred_kld = np.append(pred_kld, KLdiv(np.array(mask, dtype='float32'), np.array(target_att, dtype='float32')))
                    cc.update(CC(np.array(mask, dtype='float32'), np.array(target_att, dtype='float32')), 1)
                    sim.update(similarity(np.array(mask, dtype='float32'), np.array(target_att, dtype='float32')), 1)
                    nss.update(NSS(np.array(mask, dtype='float32'), np.array(target_att, dtype='float32')), 1)


        outputs_all += outputs
        targets_all += targets

    et = time.time()
    test_time = et - st

    misclassified_img = []
    # get misclassified img set
    for idx in range(len(misclassified)):
        if y_true[idx] != y_pred[idx]:
            _, img_path = os.path.split(misclassified[idx])
            misclassified_img.append(img_path)

    test_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()
    metric = metric_acc(torch.argmax(torch.stack(outputs_all), dim=1), torch.stack(targets_all))
    auc = roc_auc_score(torch.stack(targets_all).cpu().numpy(), torch.softmax(torch.stack(outputs_all   ), dim=1)[:, 1].cpu().numpy())

    if output_reasonbality:
        reasonbality={}
        reasonbality_matrix =np.zeros((2,2))

        for item in misclassified:
            _, img_path = os.path.split(item)
            att_acc = 'unreasonable' if ious[img_path] < 0.5 else 'reasonable'

            if img_path == '':
                continue
            # Four cases
            if att_acc == 'unreasonable':
                if img_path in misclassified_img:
                    reasonbality[img_path]='Unreasonable Inaccurate'
                else:
                    reasonbality[img_path]='Unreasonable Accurate'
            else:
                if img_path in misclassified_img:
                    reasonbality[img_path]='Reasonable Inaccurate'
                else:
                    reasonbality[img_path]='Reasonable Accurate'

        # generate the reasonablity matrix
        for label in reasonbality.values():
            if label == 'Unreasonable Inaccurate':
                reasonbality_matrix[1, 1] += 1
            elif label == 'Unreasonable Accurate':
                reasonbality_matrix[0, 1] += 1
            elif label == 'Reasonable Inaccurate':
                reasonbality_matrix[1, 0] += 1
            elif label == 'Reasonable Accurate':
                reasonbality_matrix[0, 0] += 1
            else:
                print('Unknown label detected:', label)

        # print the reasonablity matrix
        print(reasonbality_matrix)
        with open(os.path.join(args.load_path, f'reasonablity_test_{n_fold}.json'), 'w') as fp:
            json.dump(reasonbality, fp)

        pred = torch.softmax(torch.stack(outputs_all), dim=1).cpu().numpy()
        np.save(os.path.join(args.load_path, f'mask_pred_{n_fold}.npy'), pred)
        np.save(os.path.join(args.load_path, f'iou_pred_{n_fold}.npy'), pred_iou)
        np.save(os.path.join(args.load_path, f'kld_pred_{n_fold}.npy'), pred_kld)
    return test_acc, iou.avg, metric+(auc, kl.avg, cc.avg, sim.avg, nss.avg)


def model_train_with_map(model, train_loader, val_loader, n_fold):
    ####################################################################################################
    # adjust the model
    ####################################################################################################
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    centroid_criterion = nn.MSELoss(reduction='none')
    attention_criterion = nn.L1Loss(reduction='none')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    best_val_iou = 0
    best_val_acc = 0

   # load grad_cam module
    #resnet
    if args.model == 'resnet18' or args.model == 'resnet34':
        grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["1"], use_cuda=args.use_cuda)        
    elif args.model == 'resnet50':
        grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)   
    #vgg
    elif args.model == 'vgg16':
        grad_cam = GradCam(model=model, feature_module=model.features, \
                        target_layer_names=["30"], use_cuda=args.use_cuda)
    #densenet
    elif args.model == 'densenet121':
        grad_cam = GradCam(model=model, feature_module=model.features, \
                        target_layer_names=["denseblock4"], use_cuda=args.use_cuda)

    for epoch in np.arange(args.n_epoch) + 1:
        # train
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

        for batch_idx, (inputs, targets,  target_maps, pred_weight, att_weight,target_centroids) in enumerate(train_loader):
            task_loss = 0
            centroid_loss = 0
            consistency_loss = 0
            attn_loss = 0
            if args.use_cuda:
                inputs, targets,  target_maps, pred_weight, att_weight,target_centroids = inputs.cuda(), targets.cuda(non_blocking=True),  target_maps.cuda(), pred_weight.cuda(), att_weight.cuda(),target_centroids.cuda()
            att_maps = []
            att_maps_centroid = []

            outputs = model(inputs)

            for input, target, valid_weight in zip(inputs, targets, att_weight):
                if valid_weight > 0.0:
                    # get attention maps from grad-CAM
                    att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(input, 0), target)
                    att_maps.append(att_map)
                    # get attention maps' centroid
                    att_map_cen = compute_centroid(att_map)
                    att_maps_centroid.append(att_map_cen)

            # compute diga loss
            task_loss = task_criterion(outputs, targets)
            task_loss = torch.mean(pred_weight * task_loss)

            # compute centroid_loss and consistency_loss
            if att_maps:
                att_maps = torch.stack(att_maps)                

                # compute centroid_loss
                att_maps_centroid = torch.stack(att_maps_centroid)
                centroid_loss = centroid_criterion(att_maps_centroid, target_centroids)
                centroid_loss = torch.mean(att_weight * torch.mean(centroid_loss, dim=-1))

                # compute consistency_loss
                target_maps_clone = target_maps.clone()
            
                target_maps_flatten = target_maps_clone.view(target_maps_clone.shape[0],-1)
                core_sh = torch.quantile(target_maps_flatten, 0.90, dim=-1)
                outlying_sh = torch.quantile(target_maps_flatten, 0.10, dim=-1)

                #core_loss
                core_clone = target_maps.clone()
                core_clone[core_clone>=core_sh.unsqueeze(-1).unsqueeze(-1)] = 1
                core_clone[core_clone<core_sh.unsqueeze(-1).unsqueeze(-1)] = 0
                core_loss = attention_criterion(att_maps * core_clone, target_maps * core_clone)
                core_loss = torch.mean(att_weight * torch.mean(torch.mean(core_loss, dim=-1), dim=-1))
                #outlying_loss
                outlying_clone = target_maps.clone()
                outlying_clone_mask = outlying_clone.clone()
                outlying_clone[outlying_clone_mask<=outlying_sh.unsqueeze(-1).unsqueeze(-1)] = 1
                outlying_clone[outlying_clone_mask>outlying_sh.unsqueeze(-1).unsqueeze(-1)] = 0
                outlying_loss = attention_criterion(att_maps * outlying_clone, target_maps * outlying_clone)
                outlying_loss = torch.mean(att_weight * torch.mean(torch.mean(outlying_loss, dim=-1), dim=-1))
                #normal_loss
                general_clone = target_maps.clone()
                general_clone[general_clone<=outlying_sh.unsqueeze(-1).unsqueeze(-1)] = 0
                general_clone[general_clone>=core_sh.unsqueeze(-1).unsqueeze(-1)] = 0
                general_loss = attention_criterion(att_maps * general_clone, target_maps * general_clone)
                general_loss = torch.mean(att_weight * torch.mean(torch.mean(general_loss, dim=-1), dim=-1))

                consistency_loss = args.beta * (core_loss + outlying_loss) + args.gamma * general_loss

                loss = task_loss  + consistency_loss + centroid_loss
            else:
                loss = task_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]

            outputs_all += outputs
            targets_all += targets
            print(f'Epoch {epoch} Batch_idx : {batch_idx}, task_loss : {task_loss.item():6f})')#, '


        et = time.time()
        train_time = et - st

        train_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()

        #valid
        print('start validation')
        model.eval()
        st = time.time()
        outputs_all = []
        targets_all = []

        
        iou = AverageMeter()
        for batch_idx, (inputs, targets, attn, paths) in enumerate(val_loader):
            if args.use_cuda:
                inputs, targets, attn = inputs.cuda(), targets.cuda(non_blocking=True), attn.cuda()

            # compute output
            with torch.no_grad():
                outputs = model(inputs)

            for img_path in paths:
                _, img_fn = os.path.split(img_path)

                img = cv2.imread(img_path, 1)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                input = preprocess_image(img)
                mask = grad_cam(input )
                # show_cam_on_image(img, mask, 'attention', img_path)

                if img_fn in path_to_attn:
                    threshold = np.quantile(mask,0.75)
                    item_att_binary = (mask > threshold)
                    target_att = path_to_attn[img_fn]
                    target_att_binary = (target_att > 0)
                    single_iou = compute_iou(item_att_binary, target_att_binary)
                    # print('iou:', single_iou)
                    iou.update(single_iou.item(), 1)
            outputs_all += outputs
            targets_all += targets

        et = time.time()
        test_time = et - st

        val_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()
        val_iou = iou.avg
        if (val_acc >= best_val_acc)and (val_iou > best_val_iou):          
            best_val_iou = val_iou
            best_val_acc = val_acc
            torch.save(model, os.path.join(os.path.join(args.save_path,args.model), f'model_out_{n_fold}.pth'))
            print('UPDATE!!!')
        print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc, 'Val IOU:', val_iou)
    return best_val_iou

def model_train(model, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    best_val_acc = 0

    for epoch in np.arange(args.n_epoch) + 1:
        # train
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

        for batch_idx, (inputs, targets, _,) in enumerate(train_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]

            outputs_all+=outputs
            targets_all+=targets

            print('Batch_idx :', batch_idx, ', loss', loss)

        et = time.time()
        train_time = et - st

        train_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()

        # valid           
        print('start validation')
        model.eval()
        st = time.time()
        outputs_all = []
        targets_all = []
        with torch.no_grad():
            for batch_idx, (inputs, targets, _, _  ) in enumerate(val_loader):
                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # compute output
                outputs = model(inputs)

                outputs_all += outputs
                targets_all += targets

        et = time.time()
        val_time = et - st

        val_acc = accuracy(torch.stack(outputs_all), torch.stack(targets_all))[0].cpu().detach()

        if val_acc > best_val_acc and epoch>0 :
            best_val_acc = val_acc
            torch.save(model, os.path.join(args.save_path, f'model_out_0.pth'))
            print('UPDATE!!!')
        print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc)

    return best_val_acc


if __name__ == '__main__':

    # Data loading code
    basedir = os.path.join(args.data_dir, 'base_train')
    traindir = os.path.join(args.data_dir, 'total')
    valdir = os.path.join(args.data_dir, 'val')
    testdir = os.path.join(args.data_dir, 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        ImageFolderWithAttn(testdir, transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        test_acc_mean = []
        test_prc_mean = []
        test_rec_mean = []
        test_f1_mean = []
        test_iou_mean = []
        test_auc_mean = []
        test_kl_mean = []
        test_cc_mean = []
        test_sim_mean = []
        test_nss_mean = []

        train_data = ImageFolderWithAttn(traindir, transforms.Compose([
                    transforms.Resize((224, 224)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        for i in range(5):
            model = torch.load(os.path.join(args.load_path, f'model_out_{i}.pth'))
            model.eval()
            if args.use_cuda:
                model.cuda()

            # evaluate model on trainset to categories MSKUS instances into UP/RP/RIP/RIP for adjusting phase
            train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.train_batch, num_workers=args.workers, shuffle=False)
            test_acc, test_iou, metric = model_test(model, train_loader, output_attention=True, output_iou=True, output_reasonbality=True, n_fold=i)

            # # evaluate model on testset
            # test_acc, test_iou, metric = model_test(model, test_loader, output_attention=True, output_iou=True, output_reasonbality=True, n_fold=i)

            print(f'Finish Testing. Test Acc:{metric[2]:.4f}, Test IOU:{test_iou:.4f},f1:{metric[3]:.4f}, auc:{metric[4]:.4f} kl:{metric[5]:.4f}, cc:{metric[6]:.4f} sim:{metric[7]:.4f} ' )

            test_acc_mean.append(metric[2])
            test_prc_mean.append(metric[0])
            test_rec_mean.append(metric[1])
            test_f1_mean.append(metric[3])
            test_iou_mean.append(test_iou)
            test_auc_mean.append(metric[4])
            test_kl_mean.append(metric[5])
            test_cc_mean.append(metric[6])
            test_sim_mean.append(metric[7])
            test_nss_mean.append(metric[8])

        test_acc_mean = np.array(test_acc_mean)
        test_prc_mean = np.array(test_prc_mean)
        test_rec_mean = np.array(test_rec_mean)
        test_f1_mean = np.array(test_f1_mean)
        test_iou_mean = np.array(test_iou_mean)
        test_auc_mean = np.array(test_auc_mean)
        test_kl_mean = np.array(test_kl_mean)
        test_cc_mean = np.array(test_cc_mean)
        test_sim_mean = np.array(test_sim_mean)
        test_nss_mean = np.array(test_nss_mean)
        print(test_acc_mean)
        print(f'acc: {np.mean(test_acc_mean):.4f} ± {np.std(test_acc_mean):.4f}')
        print(f'prc: {np.mean(test_prc_mean):.4f} ± {np.std(test_prc_mean):.4f}')
        print(f'rec: {np.mean(test_rec_mean):.4f} ± {np.std(test_rec_mean):.4f}')
        print(f'f1: {np.mean(test_f1_mean):.4f} ± {np.std(test_f1_mean):.4f}')
        print(f'auc: {np.mean(test_auc_mean):.4f} ± {np.std(test_auc_mean):.4f}')
        print(f'iou: {np.mean(test_iou_mean):.4f} ± {np.std(test_iou_mean):.4f}')
        print(f'kl: {np.mean(test_kl_mean):.4f} ± {np.std(test_kl_mean):.4f}')
        print(f'cc: {np.mean(test_cc_mean):.4f} ± {np.std(test_cc_mean):.4f}')
        print(f'sim: {np.mean(test_sim_mean):.4f} ± {np.std(test_sim_mean):.4f}')
        print(f'nss: {np.mean(test_nss_mean):.4f} ± {np.std(test_nss_mean):.4f}')
        print(f'{np.mean(test_acc_mean):.4f} ± {np.std(test_acc_mean):.4f}')
        print(f'{np.mean(test_prc_mean):.4f} ± {np.std(test_prc_mean):.4f}')
        print(f'{np.mean(test_rec_mean):.4f} ± {np.std(test_rec_mean):.4f}')
        print(f'{np.mean(test_f1_mean):.4f} ± {np.std(test_f1_mean):.4f}')
        print(f'{np.mean(test_auc_mean):.4f} ± {np.std(test_auc_mean):.4f}')
        print(f'{np.mean(test_iou_mean):.4f} ± {np.std(test_iou_mean):.4f}')
        print(f'{np.mean(test_kl_mean):.4f} ± {np.std(test_kl_mean):.4f}')
        print(f'{np.mean(test_cc_mean):.4f} ± {np.std(test_cc_mean):.4f}')
        print(f'{np.mean(test_sim_mean):.4f} ± {np.std(test_sim_mean):.4f}')
        print(f'{np.mean(test_nss_mean):.4f} ± {np.std(test_nss_mean):.4f}')

    else:
        if args.trainWithMap:
            # adjust the model on dataset with sonographers' eye gaze maps

            # 5-fold cross verification
            kf = KFold(5, shuffle=True, random_state=2023)
            total_data = ImageFolderWithAttn(traindir, transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize,]))
            for i, (train_id, val_id) in enumerate(kf.split(total_data)):
                print(f'Fold: {i}')
                # valset
                val_sampler = torch.utils.data.SubsetRandomSampler(val_id)
                val_loader = torch.utils.data.DataLoader(
                    total_data, 
                    batch_size=args.train_batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True, sampler=val_sampler)

                model = torch.load(os.path.join(args.load_path, f'model_out_{0}.pth'))
                if args.use_cuda:
                    model.cuda()

                # remain trainset
                train_sampler = torch.utils.data.SubsetRandomSampler(train_id)
                if args.baseline: #baseline: train without TLS
                    train_loader = torch.utils.data.DataLoader(
                    ImageFolderWithPaths(traindir, transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,])), 
                    batch_size=args.train_batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True, sampler=train_sampler)
                    best_val_acc= model_train(model, train_loader, val_loader, i)
                    print('Finish Training. Best Validation acc:', best_val_acc)

                else: #TLS: train without TLS             
                    train_loader = torch.utils.data.DataLoader(
                        ImageFolderWithMapsAndWeights(traindir,args.load_path, 0, transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,])), 
                        batch_size=args.train_batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
                    best_val_iou = model_train_with_map(model, train_loader, val_loader, i)
                    print('Finish Training. Best Validation IOU:', best_val_iou)
               
        #pre-train to get a base model               
        else:
            train_data = ImageFolderWithPaths(basedir, transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            val_data = ImageFolderWithAttn(testdir, transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                ]))
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

            model = model_dict[args.model]['model'](weights=model_dict[args.model]['weights'])
            if args.model == 'resnet50':
                model.fc = nn.Linear(2048, 2)
            elif args.model == 'resnet18' or args.model == 'resnet34':
                model.fc = nn.Linear(512, 2)
            elif args.model == 'vgg16':
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
            else: #densenet121
                model.classifier = nn.Linear(1024, 2)
 
            if args.use_cuda:
                model.cuda()
            best_val_acc= model_train(model, train_loader, val_loader)
            print('Finish Training. Best Validation acc:', best_val_acc)

