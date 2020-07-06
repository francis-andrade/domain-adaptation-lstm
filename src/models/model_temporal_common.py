"""
Module that implements the CommonLSTM, as described in the dissertation.
"""

import torch
from torch import nn
from torch.nn.utils import rnn
from models.model_temporal import MDANTemporal
import torch.nn.functional as F

class MDANTemporalCommon(MDANTemporal):

    def __init__(self, num_domains, image_dim):
        """
        Args: 
            numdomains: number of source domains.
            image_dim: frame dimensions.
        """
        super(MDANTemporalCommon, self).__init__(num_domains, image_dim)
        
      
    
    def forward_temporal(self, inputs, mask=None):
        """
        Forward pass on the temporal layers.

        Args:
            inputs: tensor with shape (batch_size, sequence_size, channels, height, width).
            mask: binary tensor with same shape as inputs to mask values outside the active region;
                if `None`, no masking is applied (default: `None`).

        Returns:
            densities: predicted density map, tensor with shape (batch_size, 1, height, width).
            h: output of the feature extractor, tensor with has shape (batch_size, sequence_size, 100). 
            counts: predicted counts, tensor with shape (batch_size, sequence_size).
        """
        N, T, C, H, W = inputs.shape
        inputs = inputs.reshape(T*N, C, H, W)
        if mask is not None:
            mask = mask.reshape(T*N, 1, H, W)

        _, densities = super().forward_cnn(inputs, mask)

        densities = densities.reshape(N, T, 1, H , W)

        h, count_fcn, count_lstm = self.forward_lstm((N, T, C, H, W), densities)
       
        counts = count_fcn + count_lstm  # predicted vehicle count

        return densities, h, counts


    def forward(self, sinputs, tinputs, mask=None, tmask=None):
        """Forward pass.

        Args:
            sinputs: A list of k tensors from k source domains, each with shape (batch_size, sequence_size, channels, height, width).
            tinputs: Tensor with target domain frames, with shape (batch_size, sequence_size, channels, height, width).
            smask: A list of k masks from k source domains with shape (batch_size, sequence_size, channels, height, width).
            tmask: Tensor representing the target domain frames. It has shape (batch_size, sequence_size, channels, height, width).
        
        Returns: 
            sdensity: List  of predicted densities for the k source domains. Each element is a tensor with shape (batch_size, sequence_size, channels, height, width).
            scount: List  of predicted counts for the k source domains. Each element is a tensor with shape (batch_size, sequence_size).
            sdomains: A list of k elements where, each element is a tensor with shape (batch_size, sequence_size, 2) that represents the predicted log probabilities of the source insts from a certain domain being part of a target and source domain
            tdomains: Tensor with the predicted probabilities of the target insts being part of a target and source domain. It has shape (batch_size, sequence_size, 2).
        """
        
        sdensity = []
        scount = []
        sh = []
        for i in range(self.num_domains):
            X = sinputs[i]
            if mask is None:
                cnn_mask = None
            else:
                cnn_mask = mask[i]
            density, h, count = self.forward_temporal(X, cnn_mask)
            sdensity.append(density)
            scount.append(count)
            sh.append(h)

        _, th, _ = self.forward_temporal(tinputs, tmask)  

        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(sh[i]))), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(th))), dim=1))

        return sdensity, scount, sdomains, tdomains
    
    