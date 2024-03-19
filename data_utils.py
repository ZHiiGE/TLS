import os
import json
import cv2
import numpy as np
from glob import glob
import torch.utils.data as data
import torchvision.datasets as datasets
def generate_path_to_attentions():
    path_to_attn = {}

    attn = glob(r'./dataset/label/' + '*')
    for item in attn:
        _, img_path = os.path.split(item)
        attention = cv2.resize(cv2.imread(item), (224, 224), interpolation=cv2.INTER_AREA)[:, :, 0]
        path_to_attn[img_path] = np.array(attention) / 255.0

    return path_to_attn

def resize_attention_label(path_to_attn, width = 7, height = 7):
    path_to_attn_resized = {}
    path_to_attn_centroid = {}
    for img_path, img_att in path_to_attn.items():
        att_map = np.uint8(img_att*255)

        M = cv2.moments(att_map)
        cx = (M['m10']/M['m00'])
        cy = (M['m01']/M['m00'])

        # 7x7 can be too small to show attention details, blur it to reward near by pixels
        img_att_resized = cv2.resize(att_map, (width,height), interpolation=cv2.INTER_AREA)

        img_att_resized = cv2.GaussianBlur(img_att_resized, (3, 3), 0)

        if np.max(img_att_resized)==0:
            print(img_path)

        path_to_attn_resized[img_path] = np.float32(img_att_resized/np.max(img_att_resized) if np.max(img_att_resized)!=0 else img_att_resized)
        path_to_attn_centroid[img_path] = np.float32([cx/32.0, cy/32.0])
    return path_to_attn_resized,path_to_attn_centroid

path_to_attn = generate_path_to_attentions()
# resize attention label from 224x224 to 7x7
path_to_attn_resized,path_to_attn_centroid = resize_attention_label(path_to_attn)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class ImageFolderWithAttn(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithAttn, self).__getitem__(index)
        path = self.imgs[index][0]

        head, tail = os.path.split(path)
        attention = cv2.resize(cv2.imread(os.path.join('./dataset/label', tail)), (14, 14), interpolation=cv2.INTER_AREA)[:, :, 0:1]
        attention_map = np.array(attention, dtype=np.float32).transpose((2, 0, 1)) / 255.0
        attention_map = attention_map.astype(np.float32)


        tuple_with_path = (original_tuple + (attention_map, path,))
        return tuple_with_path

class ImageFolderWithMapsAndWeights(datasets.ImageFolder):
    def __init__(self, root,mx_path, n_fold, transform, attn_weight):
        super().__init__(root, transform=transform)
        self.alpha = attn_weight
        f = open(os.path.join(mx_path, f'reasonablity_test_{n_fold}.json'))
        self.reasonablity_labels = json.load(f)
        self.n_fold = n_fold
        f.close()

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithMapsAndWeights, self).__getitem__(index)
        path = self.imgs[index][0]

        head, tail = os.path.split(path)

        # weights of RP (1-alpha)
        pred_weight = 1-self.alpha
        att_weight = 1-self.alpha
        if tail in path_to_attn_resized:
            true_attention_map = path_to_attn_resized[tail]

            if tail in self.reasonablity_labels:
                label = self.reasonablity_labels[tail]

                if label == 'Unreasonable Inaccurate':
                    pred_weight = self.alpha
                    att_weight = self.alpha

                elif label == 'Unreasonable Accurate':
                    pred_weight = 1-self.alpha
                    att_weight = self.alpha


                elif label == 'Reasonable Inaccurate':
                    pred_weight = self.alpha
                    att_weight = 1-self.alpha

        else:
            true_attention_map = np.zeros((7,7), dtype=np.float32)
    
        true_attention_map = path_to_attn_resized[tail] + 0.0001
        true_attn_centroid = path_to_attn_centroid[tail]
        tuple_with_map_and_weights = (original_tuple + ( true_attention_map, pred_weight, att_weight,true_attn_centroid))
        return tuple_with_map_and_weights
