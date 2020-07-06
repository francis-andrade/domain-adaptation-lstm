"""
Module that implements the Simple Model, as described in the dissertation.
"""

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
    """
    Flattens a tensor.
    """
    def forward(self, x):
        batch_size = x.shape[0] # read in N, C, H, W
        return x.reshape(batch_size, -1)  # "flatten" the C * H * W values into a single vector per image
        #return x.view(batch_size, -1)

class MDANet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, num_domains):
        """
        Args:
            num_domains: Number of source domains
        """
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

    def forward_cnn(self, inputs, mask = None):
        """Forward pass on the convolutional neural network

        Args:
            inputs: tensor with shape (batch_size, channels, height, width).
            mask: binary tensor with same shape as inputs to mask values outside the active region;
                if `None`, no masking is applied (default: `None`).
        
        Returns:
            h: Output of the hyper-atrous combination. It is a tensor with shape (fc_size, 1, height, width). fc_size is dependent on the dataset.
            density: Tensor with the densities predicted by the CNN. It has shape (batch_size, 1, height, width).
        """
        
        if mask is not None:
            inputs = inputs * mask  # zero input values outside the active region

        h1 = self.conv_blocks[0](input)
        h2 = self.conv_blocks[1](h1)
        h3 = self.conv_blocks[2](h2)
        h4 = self.conv_blocks[3](h3)
        h = torch.cat((h1, h2, h3, h4), dim=1)  # hyper-atrous combination
        g = self.conv_blocks[4](h)
        if mask is not None:
            g = g * mask  # zero output values outside the active region

        density = g  # predicted density map

        return h, density

    def forward(self, sinputs, tinputs, smask=None, tmask=None):
        """Forward pass.

        Args:
            sinputs: A list of k tensors from k source domains, each with shape (batch_size, channels, height, width).
            tinputs: Tensor with target domain frames, with shape (batch_size, channels, height, width).
            smask: A list of k masks from k source domains with shape (batch_size, channels, height, width).
            tmask: Tensor representing the target domain frames. It has shape (batch_size, channels, height, width).
        
        Returns: 
            sdensity: List  of predicted densities for the k source domains. Each element is a tensor with shape (batch_size, channels, height, width).
            scount: List  of predicted counts for the k source domains. Each element is a tensor with shape (batch_size).
            sdomains: A list of k elements where, each element is a tensor with shape (batch_size, 2) that represents the predicted log probabilities of the source insts from a certain domain being part of a target and source domain
            tdomains: Tensor with the predicted probabilities of the target insts being part of a target and source domain. It has shape (batch_size, 2).
        """

        sdensity = []
        scount = []
        sh = []
        for i in range(self.num_domains):
            if smask is None:
                cnn_mask = None
            else:
                cnn_mask = smask[i]
            h, density = self.forward_cnn(sinputs[i], cnn_mask)
            count = density.sum(dim=(1,2,3))
            sh.append(h)
            sdensity.append(density)
            scount.append(count)

        th, _ = self.forward_cnn(tinputs, tmask)

        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(sh[i]))), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(th))), dim=1))

        return sdensity, scount, sdomains, tdomains
    
    def inference(self, inputs, mask=None):
        """Computes predicted densities and counts.

        Args:
            inputs: tensor with shape (batch_size, channels, height, width).
            mask: binary tensor with same shape as inputs to mask values outside the active region;
                if `None`, no masking is applied (default: `None`).
        
        Returns:
            densities: predicted density map, tensor with shape (batch_size, 1, height, width).
            count: predicted number of vehicles in each image, tensor with shape (batch_size).
        """
        _, densities = self.forward_cnn(inputs, mask)
        counts = densities.sum(dim=(1,2,3))
        return densities, counts
    

   

