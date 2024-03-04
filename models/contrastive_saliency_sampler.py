import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from .contrastive_resnet import *
from .saliency_network import *


__all__ = [
    "sup_con_resnet",
    "linear_classifier",
]


def Gaussian_2d(size, fwhm=9):
    """ Make a square gaussian kernel.
    size is the length of a side of the square, fwhm is the effective radius.
    Return: a gaussian matrix of shape: [size, size]
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    x0 = y0 = size // 2

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def uniform_grids_2d(grid_size, padding_size):
    """
    generate uniform grids with size (91, 91, 2), each element has the value between -1 and 2.
    """
    global_size = grid_size + 2 * padding_size
    uniform_coords = torch.zeros(2, global_size, global_size)

    for k in range(2):
        for i in range(global_size):
            for j in range(global_size):
                uniform_coords[k, i, j] = k * (i - padding_size) / (grid_size - 1.0) \
                                          + (1.0 - k) * (j - padding_size) / (grid_size - 1.0)

    return uniform_coords


def projector(args):
    if args.projector == "Linear":
        head = nn.Linear(2048, 128)

    elif args.projector == "MLP_2":
        head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
        )

    elif args.projector == "MLP_3":
        head = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 128)
            )

    return head


class SaliencySampler(nn.Module):
    def __init__(self, args, task_network, saliency_network, task_input_size, saliency_input_size):
        super(SaliencySampler, self).__init__()

        self.args = args
        self.task_network = task_network  # High resolution network which is used to generate the saliency map
        self.saliency_network = saliency_network  # The saliency network
        self.grid_size = 31  # The size of the grid
        self.padding_size = 30  # The padding size of the saliency map (grid_size + padding_size * 2)
        self.global_size = self.grid_size + 2 * self.padding_size  # The size of the saliency map (grid_size + padding_size * 2)
        self.input_size = saliency_input_size  # The input size of the saliency network
        self.input_size_net = task_input_size  # The input size of the task network

        conv_last_basic_channel = 256
        self.conv_last_1 = nn.Conv2d(conv_last_basic_channel, 1, kernel_size=1, stride=1, padding=0, bias=False)  # The last convolution layer of the task network
        self.conv_last_2 = nn.Conv2d(conv_last_basic_channel * 2, 1, kernel_size=1, stride=1, padding=0, bias=False)  # The last convolution layer of the task network
        self.conv_last_3 = nn.Conv2d(conv_last_basic_channel * 4, 1, kernel_size=1, stride=1, padding=0, bias=False)  # The last convolution layer of the task network
        self.conv_last_4 = nn.Conv2d(conv_last_basic_channel * 8, 1, kernel_size=1, stride=1, padding=0, bias=False)  # The last convolution layer of the task network

        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1), stride=1, padding=0, bias=False)  # The convolution layer for the spatial transformer
        gaussian_weights = torch.FloatTensor(
            Gaussian_2d(2 * self.padding_size + 1, fwhm=13))  # The Gaussian weights for the spatial transformer
        self.filter.weight[0].data[:, :, :] = gaussian_weights  # Set the Gaussian weights to the convolution layer
        self.uniform_coords = uniform_grids_2d(self.grid_size, self.padding_size)

    def create_2d_grid(self, xs):
        x = nn.ReplicationPad2d(self.padding_size)(xs)

        P = torch.autograd.Variable(torch.zeros(1, 2, self.global_size, self.global_size).cuda(), requires_grad=False)
        P[0, :, :, :] = self.uniform_coords
        P = P.expand(x.size(0), 2, self.global_size, self.global_size)

        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size, self.global_size)
        all_filter = self.filter(x_mul).view(-1, 2, self.grid_size, self.grid_size)

        x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        x_filter = x_filter / p_filter
        x_grids = x_filter * 2 - 1
        x_grids = torch.clamp(x_grids, min=-1, max=1)
        x_grids = x_grids.view(-1, 1, self.grid_size, self.grid_size)

        y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        y_filter = y_filter / p_filter
        y_grids = y_filter * 2 - 1
        y_grids = torch.clamp(y_grids, min=-1, max=1)
        y_grids = y_grids.view(-1, 1, self.grid_size, self.grid_size)

        grid = torch.cat((x_grids, y_grids), 1)

        grid = F.interpolate(grid, size=(self.input_size_net, self.input_size_net), mode="bilinear", align_corners=True)

        grid = torch.permute(grid, (0, 2, 3, 1))

        return grid

    def generate_sampled_image(self, x, feature_maps, conv_layer):

        feature_maps = nn.ReLU()(feature_maps)
        feature_map = conv_layer(feature_maps)

        feature_map = F.interpolate(feature_map, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=True)

        feature_map = feature_map.view(-1, self.grid_size * self.grid_size)
        saliency_map = nn.Softmax(dim=1)(feature_map / self.args.temperature)
        saliency_map = saliency_map.view(-1, 1, self.grid_size, self.grid_size)

        grid = self.create_2d_grid(saliency_map)

        sampled_image = F.grid_sample(x, grid, align_corners=True)

        return sampled_image, saliency_map

    def forward(self, x, p):
        x_low = nn.AdaptiveAvgPool2d((self.input_size, self.input_size))(x)

        x1_saliency, x2_saliency, x3_saliency, x4_saliency, outputs_saliency = self.saliency_network(x_low)

        x1_sampled, saliency_map = self.generate_sampled_image(x, x1_saliency.detach(), conv_layer=self.conv_last_1)
        x2_sampled, saliency_map = self.generate_sampled_image(x, x2_saliency.detach(), conv_layer=self.conv_last_2)
        x3_sampled, saliency_map = self.generate_sampled_image(x, x3_saliency.detach(), conv_layer=self.conv_last_3)
        x4_sampled, saliency_map = self.generate_sampled_image(x, x4_saliency.detach(), conv_layer=self.conv_last_4)

        if random.random() > p:
            s = random.randint(64, 224)
            x_sampled = nn.AdaptiveAvgPool2d((s, s))(x_sampled)
            x_sampled = nn.Upsample(size=(self.input_size_net, self.input_size_net), mode="bilinear", align_corners=True)(x_sampled)

        images = torch.cat([x_low, x1_sampled, x2_sampled, x3_sampled, x4_sampled], dim=0)
        outputs_contrastive = self.task_network(images)

        return outputs_saliency, outputs_contrastive


def contrastive_saliency50_50(args, **kwargs):
    model = SaliencySampler(args=args, task_network=contrastive_resnet50(**kwargs), saliency_network=saliency_network_resnet50(**kwargs), task_input_size=224, saliency_input_size=224)
    return model


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, args, **kwargs):
        super(SupConResNet, self).__init__()
        self.args = args
        self.encoder = contrastive_saliency50_50(self.args, **kwargs)
        self.head = projector(self.args)

    def forward(self, x, p):
        outputs_saliency, outputs_contrastive = self.encoder(x, p)
        outputs_contrastive = F.normalize(self.head(outputs_contrastive), dim=1)
        return outputs_saliency, outputs_contrastive


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, args, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, features):
        return self.fc(features)

def sup_con_resnet(args, **kwargs):
    model = SupConResNet(args, **kwargs)
    return model

def linear_classifier(args, **kwargs):
    model = LinearClassifier(args, **kwargs)
    return model




