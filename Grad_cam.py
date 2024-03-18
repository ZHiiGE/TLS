from typing import Any
import cv2
import numpy as np
import torch

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = torch.flatten(x, 1)
            else:
                x = module(x)
        
        return target_activations, x
    
    
    # # densenet
    # def __call__(self, x):
    #     target_activations = []
    #     for name, module in self.model._modules.items():
    #         if module == self.feature_module:
    #             target_activations, x = self.feature_extractor(x)
    #         else :
    #             x = F.relu(x, inplace=True)
    #             x = F.adaptive_avg_pool2d(x, (1, 1))
    #             x = torch.flatten(x, 1)
    #             x = module(x)
    #     return target_activations, x
    
class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def get_attention_map(self, input,  index=None):
        torch.autograd.set_detect_anomaly(True)
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]
        target = features[-1].squeeze()
        if grads_val.dim() == 3:
            target = target[1:,:]
            target = target.permute(1,0)
            target = target.reshape(target.size(0),14,14)
            grads_val = grads_val[:,1:,:]
            grads_val = grads_val.permute(0,2,1)
            grads_val = grads_val.reshape(grads_val.size(0),grads_val.size(1), 14,14)
        weights = torch.mean(grads_val, axis=(2, 3)).squeeze()

        if self.cuda:
            cam = torch.zeros(target.shape[1:]).cuda()
        else:
            cam = torch.zeros(target.shape[1:])

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)

        if(torch.isnan(cam).any() == True):
            print(1)
        return cam, output


    def __call__(self, input ,index=None):#
        torch.autograd.set_detect_anomaly(True)
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        if grads_val.ndim == 3:
            target = target[1:,:]
            target = target.transpose(1,0)
            target = target.reshape(target.shape[0],14,14)
            grads_val = grads_val[:,1:,:]
            grads_val = grads_val.transpose(0,2,1)
            grads_val = grads_val.reshape(grads_val.shape[0],grads_val.shape[1], 14,14)

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam

        return cam
    

