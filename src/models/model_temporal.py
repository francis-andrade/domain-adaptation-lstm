"""
Module that implements the Base class of temporal models.
"""

import torch
from torch import nn
from torch.nn.utils import rnn
from model import MDANet
import torch.nn.functional as F
import settings

class MDANTemporal(MDANet):

    def __init__(self, num_domains, image_dim):
        """
        Args: 
            numdomains: number of source domains.
            image_dim: frame dimensions.
        """
        super(MDANTemporal, self).__init__(num_domains)
        H, W = image_dim
        self.lstm_block = nn.LSTM(H*W, 100, num_layers=3, batch_first=True)
        self.final_layer = nn.Linear(100, 1)
        self.domains = nn.ModuleList([nn.Sequential(nn.Linear(100*settings.SEQUENCE_SIZE, 10), nn.Linear(10, 2)) for _ in range(self.num_domains)])
    
    def init_hidden(self, batch_size):
        """
        Provides Gaussian samples to be used as initial values for the hidden states of the LSTM.

        Args:
            batch_size: the size of the current batch.

        Returns:
            h0: tensor with shape (num_layers, batch_size, hidden_dim).
            c0: tensor with shape (num_layers, batch_size, hidden_dim).
        """
        h0, c0 = torch.randn(3, batch_size, 100), torch.randn(3, batch_size, 100)  # each LSTM state has shape (num_layers, batch_size, hidden_dim)
        device = next(self.parameters()).device
        h0, c0 = h0.to(device), c0.to(device)
        return h0, c0
    
    def forward_temporal(self, inputs, mask=None):
        """
        Forward pass on the temporal layers.

        Args:
            inputs: tensor with shape (batch_size, sequence_size, channels, height, width).
            mask: binary tensor with same shape as inputs to mask values outside the active region;
                if `None`, no masking is applied (default: `None`).
        """
        raise('Not implemented error')

    def forward_lstm(self, shape, densities):
        """
        Forward pass on the LSTM layers.

        Args:
            shape: Shape of the input tensor with frames, given as parameter in the main forward pass function.
            densities: predicted density map, tensor with shape (batch_size, 1, height, width). 

         Returns:
            h: Output of the last LSTM layer. It is a tensor with shape (batch_size, sequence_size, 100). 
            count_fcn: Tensor with the counts predict by the FCN. It has shape (batch_size, sequence_size).
            count_lstm: Tensor with the residual counts predicted by the LSTM. It has shape (batch_size, sequence_size)
        """
        N, T, C, H, W = shape

        density_reshaped = densities.reshape(N, T, -1)
        count_fcn = density_reshaped.sum(dim=2)


        h0 = self.init_hidden(N)
        h, _ = self.lstm_block(density_reshaped, h0)

        count_lstm = self.final_layer(h.reshape(T*N, -1)).reshape(N, T)

        return h, count_fcn, count_lstm

    
    def inference(self, inputs, mask=None):
        """Computes predicted densities and counts.

        Args:
            inputs: tensor with shape (batch_size, sequence_size, channels, height, width).
            mask: binary tensor with same shape as inputs to mask values outside the active region;
                if `None`, no masking is applied (default: `None`).
        
        Returns:
            densities: predicted density map, tensor with shape (batch_size, 1, height, width).
            count: predicted number of vehicles in each image, tensor with shape (batch_size).
        """
        densities, _, counts = self.forward_temporal(inputs, mask)
        return densities, counts