"""
Module that implements the DoubleLSTM, as described in the dissertation.
"""

import torch
from torch import nn
from torch.nn.utils import rnn
from models.model_temporal import MDANTemporal
import torch.nn.functional as F
import settings

class MDANTemporalDouble(MDANTemporal):

    def __init__(self, num_domains, image_dim):
        """
        Args: 
            numdomains: number of source domains.
            image_dim: frame dimensions.
        """
        super(MDANTemporalDouble, self).__init__(num_domains, image_dim)
        H, W = image_dim
        self.domains_deconv = nn.ModuleList([nn.Sequential(nn.Conv2d(1408, 512, (3, 3)), nn.ReLU(), nn.Conv2d(512, 16, (3,3)), nn.ReLU()) for _ in range(self.num_domains)])
        if settings.DATASET == 'webcamt':
            self.lstm_layer_size = 16640
        elif settings.DATASET == 'ucspeds':
            self.lstm_layer_size = 30800
        self.domains_lstm = nn.ModuleList([nn.LSTM(self.lstm_layer_size, 100, num_layers=3, batch_first=True) for _ in range(self.num_domains)])
        
    
    def forward_temporal(self, inputs, mask=None):
        """
        Forward pass on the temporal layers.

        Args:
            inputs: tensor with shape (batch_size, sequence_size, channels, height, width).
            mask: binary tensor with same shape as inputs to mask values outside the active region;
                if `None`, no masking is applied (default: `None`).

        Returns:
            densities: predicted density map, tensor with shape (batch_size, 1, height, width).
            h: output of the feature extractor, tensor with has shape (batch_size, sequence_size, fc_size). 
            counts: predicted counts, tensor with shape (batch_size, sequence_size).
        """
        N, T, C, H, W = inputs.shape
        inputs = inputs.reshape(T*N, C, H, W)
        if mask is not None:
            mask = mask.reshape(T*N, 1, H, W)

        h, density = super().forward_cnn(inputs, mask)

        density = density.reshape(N, T, 1, H , W)

        _, count_fcn, count_lstm = self.forward_lstm((N, T, C, H, W), density)
       
        count = count_fcn + count_lstm  # predicted vehicle count

        return density, h, count

    


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

        N, T, C, H, W = sinputs[0].shape
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
            h_lstm = self.grls[i].apply(sh[i])
            h0 = self.init_hidden(N)
            h_lstm = self.domains_deconv[i](sh[i])
            h_lstm = h_lstm.reshape(N, T, -1)
            h_lstm, _ = self.domains_lstm[i](h_lstm, h0)
            sdomains.append(F.log_softmax(self.domains[i](self.flatten[i](h_lstm)), dim=1))
            h_lstm = self.grls[i].apply(sh[i])
            h0 = self.init_hidden(N)
            h_lstm = self.domains_deconv[i](th)
            h_lstm = h_lstm.reshape(N, T, -1)
            h_lstm, _ = self.domains_lstm[i](h_lstm, h0)
            tdomains.append(F.log_softmax(self.domains[i](self.flatten[i](h_lstm)), dim=1))

        return sdensity, scount, sdomains, tdomains
    
        
        