#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


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
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class MDANet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, num_domains):
        super(MDANet, self).__init__()
        self.num_domains = num_domains
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        self.domains = nn.ModuleList([nn.Linear(1024, 2) for _ in range(self.num_domains)])
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

    def forward_single_input(self, input, mask = None):
        
        if mask is not None:
            input = input * mask  # zero input values outside the active region

        h1 = self.conv_blocks[0](input)
        h2 = self.conv_blocks[1](h1)
        h3 = self.conv_blocks[2](h2)
        h4 = self.conv_blocks[3](h3)
        h = torch.cat((h1, h2, h3, h4), dim=1)  # hyper-atrous combination
        h = self.conv_blocks[4](h)
        if mask is not None:
            h = h * mask  # zero output values outside the active region

        density = h  # predicted density map
        count = h.sum(dim=(1, 2, 3))  # predicted vehicle count

        return density, count

    def forward(self, sinputs, tinputs, mask=None):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """

        sdensity = []
        scount = []
        for i in range(self.num_domains):
            density, count = self.forward_single_input(sinputs[i], mask)
            sdensity.append(density)
            scount.append(count)

        tdensity, tcount = self.forward_single_input(tinputs)

        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i].apply(self.flatten[i](sdensity[i]))), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i].apply(self.flatten[i](tdensity)))))

        return sdensity, sdomains, tdomains

   

