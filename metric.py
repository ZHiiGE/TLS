import torch as t
import numpy as np
import math

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def KLdiv(pred_map, gt_map):
    eps = 2.2204e-16
    pred_map = pred_map/(np.sum(pred_map)+eps)
    gt_map = gt_map/(np.sum(gt_map)+eps)
    div = np.sum(gt_map * np.log(eps + gt_map/(pred_map+eps)))
    return div 
        
    
def CC(pred_map,gt_map):
    eps = 2.2204e-16
    gt_map_ = (gt_map - np.mean(gt_map))
    pred_map_ = (pred_map - np.mean(pred_map))
    cc = np.sum((gt_map_*pred_map_))/(math.sqrt(np.sum((gt_map_*gt_map_))*np.sum((pred_map_*pred_map_)))+eps)
    return cc


def similarity(pred_map,gt_map):
    eps = 2.2204e-16
    gt_map = (gt_map - np.min(gt_map))/(np.max(gt_map)-np.min(gt_map)+eps)
    gt_map = gt_map/(np.sum(gt_map)+eps)
    
    pred_map = (pred_map - np.min(pred_map))/(np.max(pred_map)-np.min(pred_map)+eps)
    pred_map = pred_map/(np.sum(pred_map)+eps)
    
    diff = np.min(np.stack([gt_map,pred_map]), axis=0)
    score = np.sum(diff)
    
    return score
    
    
def NSS(pred_map,fix_map):
    '''ground truth here is fixation map'''
    eps = 2.2204e-16
    pred_map_ = (pred_map - np.mean(pred_map))/(np.std(pred_map)+eps)
    mask = fix_map > 0
    score = np.mean(pred_map_[mask==True])
    if score != score:
        print(mask)
    return score


def AUC(pred_map,fix_map):
    #pred_map = t.squeeze(pred_map)
    jitter = t.ones((pred_map.shape[0],pred_map.shape[1]))
    jitter = jitter/10000
    pred_map = pred_map + jitter


    a1, a2= pred_map.size()
    pred_map = pred_map.reshape(-1)
    pred_map = (pred_map - pred_map.min(0)[0])/(pred_map.max(0)[0] - pred_map.min(0)[0])
    pred_map = pred_map.reshape((a1, a2))
    pred_map[t.isnan(pred_map)] = 1

    P = pred_map.reshape(-1)
    f = fix_map.reshape(-1)
    
    Pth = P[t.gt(f,0)]

    Nfixation = Pth.shape[0]
    Npixel = P.shape[0]

    allthreshes = Pth.clone()
    t.sort(allthreshes, dim=-1, descending=True)
    tp = t.zeros(Nfixation+2)
    fp = t.zeros(Nfixation+2)
    tp[1] = 0
    tp[-1] = 1
    fp[1] = 0
    fp[-1] = 1

    for i in range(1,Nfixation):
        thresh = allthreshes[i]
        aboveth = t.sum(P >= thresh)
        tp[i+1] = i / Nfixation
        fp[i+1] = (aboveth-i) / (Npixel - Nfixation)
        score = t.trapz(fp,tp)

    return score

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def metric_acc(output, target):
    TP = ((output==1) & (target==1)).int().sum()
    TN = ((output==0) & (target==0)).int().sum()
    FP = ((output==1) & (target==0)).int().sum()
    FN = ((output==0) & (target==1)).int().sum()

    Smooth = 1e-3
    PRC = (TP+Smooth) / (TP+FP+Smooth)
    REC = (TP+Smooth) / (TP+FN+Smooth)
    ACC = (TP+TN+Smooth) / (TP+TN+FP+FN+Smooth)
    F1 = (2*PRC*REC) / (PRC+REC)
    return PRC.cpu().numpy(), REC.cpu().numpy(), ACC.cpu().numpy(), F1.cpu().numpy()

def compute_iou(x, y):
    intersection = np.bitwise_and(x,y)
    union = np.bitwise_or(x,y)

    iou = np.sum(intersection)/np.sum(union)

    return iou
