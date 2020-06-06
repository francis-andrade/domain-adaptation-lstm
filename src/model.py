#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import settings

class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(ctx, inputs):
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0] # read in N, C, H, W
        return x.reshape(batch_size, -1)  # "flatten" the C * H * W values into a single vector per image
        #return x.view(batch_size, -1)

class MDANet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, num_domains):
        super(MDANet, self).__init__()
        self.num_domains = num_domains
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        if settings.DATASET == 'webcamt':
            self.layer_size = 1858560
        elif settings.DATASET == 'ucspeds':
            self.layer_size = 3239808
        self.domains = nn.ModuleList([nn.Sequential(nn.Linear(self.layer_size, 10), nn.Linear(10, 2)) for _ in range(self.num_domains)])
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]
        self.flatten = [Flatten() for _ in range(self.num_domains)]

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv1_1', nn.Conv2d(3, 64, (3, 3), padding=1)),
                ('ReLU1_1', nn.ReLU()),
                ('Conv1_2', nn.Conv2d(64, 64, (3, 3), padding=1)),
                ('ReLU1_2', nn.ReLU()),
                ('MaxPool1', nn.MaxPool2d((2, 2))),
                ('Conv2_1', nn.Conv2d(64, 128, (3, 3), padding=1)),
                ('ReLU2_1', nn.ReLU()),
                ('Conv2_2', nn.Conv2d(128, 128, (3, 3), padding=1)),
                ('ReLU2_2', nn.ReLU()),
                ('MaxPool2', nn.MaxPool2d((2, 2))),
            ])))
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv3_1', nn.Conv2d(128, 256, (3, 3), padding=1)),
                ('ReLU3_1', nn.ReLU()),
                ('Conv3_2', nn.Conv2d(256, 256, (3, 3), padding=1)),
                ('ReLU3_2', nn.ReLU()),
                ('Atrous1', nn.Conv2d(256, 256, (3, 3), dilation=2, padding=2)),
                ('ReLU_A1', nn.ReLU()),
            ])))
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv4_1', nn.Conv2d(256, 256, (3, 3), padding=1)),
                ('ReLU4_1', nn.ReLU()),
                ('Conv4_2', nn.Conv2d(256, 256, (3, 3), padding=1)),
                ('ReLU4_2', nn.ReLU()),
                ('Atrous2', nn.Conv2d(256, 512, (3, 3), dilation=2, padding=2)),
                ('ReLU_A2', nn.ReLU()),
            ])))
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Atrous3', nn.Conv2d(512, 512, (3, 3), dilation=2, padding=2)),
                ('ReLU_A3', nn.ReLU()),
                ('Atrous4', nn.Conv2d(512, 512, (3, 3), dilation=2, padding=2)),
                ('ReLU_A4', nn.ReLU()),
            ])))
        if settings.DATASET == 'webcamt':
            padding_deconv2 = 1
        elif settings.DATASET == 'ucspeds':
            padding_deconv2 = 0
        self.conv_blocks.append(
            nn.Sequential(OrderedDict([
                ('Conv5', nn.Conv2d(1408, 512, (1, 1))),  # 1408 = 128 + 256 + 512 + 512 (hyper-atrous combination)
                ('ReLU5', nn.ReLU()),
                ('Deconv1', nn.ConvTranspose2d(512, 256, (3, 3), stride=2, padding=1, output_padding=1)),
                ('ReLU_D1', nn.ReLU()),
                ('Deconv2', nn.ConvTranspose2d(256, 64, (3, 3), stride=2, padding=padding_deconv2, output_padding=1)),
                ('ReLU_D2', nn.ReLU()),
                ('Conv6', nn.Conv2d(64, 1, (1, 1))),
            ])))

    def forward_cnn(self, input, mask = None):
        
        if mask is not None:
            input = input * mask  # zero input values outside the active region

        #print(input.shape)
        h1 = self.conv_blocks[0](input)
        h2 = self.conv_blocks[1](h1)
        h3 = self.conv_blocks[2](h2)
        h4 = self.conv_blocks[3](h3)
        h = torch.cat((h1, h2, h3, h4), dim=1)  # hyper-atrous combination
        g = self.conv_blocks[4](h)
        #print(g.shape)
        if mask is not None:
            g = g * mask  # zero output values outside the active region

        density = g  # predicted density map
        #count = g.sum(dim=(1, 2, 3))  # predicted vehicle count

        return h, density

    def forward(self, sinputs, tinputs, mask=None):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """

        sdensity = []
        scount = []
        sh = []
        for i in range(self.num_domains):
            h, density = self.forward_cnn(sinputs[i], mask)
            count = density.sum(dim=(1,2,3))
            sh.append(h)
            sdensity.append(density)
            scount.append(count)

        th, _ = self.forward_cnn(tinputs)

        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(sh[i]))), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(th))), dim=1))

        return sdensity, scount, sdomains, tdomains
    
    def inference(self, inputs):
        _, densities = self.forward_cnn(inputs)
        counts = densities.sum(dim=(1,2,3))
        return densities, counts
    
    def to_string(self):
        return "simple_model"

   

