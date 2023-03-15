import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import time
import os
from torch.nn.modules.loss import CrossEntropyLoss


class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight=[0.1, 0.45, 0.45]):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        # print(intersect.item())
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        weight = self.weight

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0

        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * weight[i]
        return loss / self.n_classes



class ce_loss(nn.module):
    def __init__(self, n_classes, weight=[0.1, 0.45, 0.45]):
        super(ce_loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

        self.ce_loss = CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, target):
        ce_loss = self.ce_loss(inputs, target)
        return ce_loss
