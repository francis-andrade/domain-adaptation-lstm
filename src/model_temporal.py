import torch
from torch import nn
from torch.nn.utils import rnn
from model import MDANet
import torch.nn.functional as F
import settings

class MDANTemporal(MDANet):

    def __init__(self, num_domains, image_dim):
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
    
    def forward_temporal(self, X, mask=None, lengths=None):
        raise('Not implemented error')

    def forward_lstm(self, shape, density, lengths = None):
        N, T, C, H, W = shape

        density_clone = density.clone().reshape(N, T, -1)
        count_fcn = density_clone.sum(dim=2)

        if lengths is not None:
            # pack padded sequence so that padded items in the sequence are not shown to the LSTM
            density_clone = rnn.pack_padded_sequence(density_clone, lengths, batch_first=True, enforce_sorted=True)

        h0 = self.init_hidden(N)
        h, _ = self.lstm_block(density_clone, h0)

        if lengths is not None:
            # undo the packing operation
            h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=T)

        count_lstm = self.final_layer(h.reshape(T*N, -1)).reshape(N, T)

        return h, count_fcn, count_lstm

    
    def inference(self, inputs, mask=None):
        densities, _, counts = self.forward_temporal(inputs, mask)
        return densities, counts