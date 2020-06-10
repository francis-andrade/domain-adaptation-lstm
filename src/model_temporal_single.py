import torch
from torch import nn
from torch.nn.utils import rnn
from model_temporal import MDANTemporal
import torch.nn.functional as F

class MDANTemporalSingle(MDANTemporal):

    def __init__(self, num_domains, image_dim):
        super(MDANTemporalSingle, self).__init__(num_domains, image_dim)
        self.domains = nn.ModuleList([nn.Sequential(nn.Linear(self.layer_size, 10), nn.Linear(10, 2)) for _ in range(self.num_domains)])
        #self.domains = nn.ModuleList([nn.Sequential(nn.Linear(1000, 10), nn.Linear(10, 2)) for _ in range(self.num_domains)])
      
    
    def forward_temporal(self, X, mask=None, lengths=None):
        N, T, C, H, W = X.shape
        X = X.reshape(T*N, C, H, W)
        if mask is not None:
            mask = mask.reshape(T*N, 1, H, W)

        h, density = super().forward_cnn(X, mask)

        _, count_fcn, count_lstm = self.forward_lstm((N, T, C, H, W), density, lengths)
       
        count = count_fcn + count_lstm  # predicted vehicle count

        #print(h.shape)
        h = h.reshape(N, T, -1)
        h = torch.mean(h, dim=1)
        #print(h.shape)
        return density, h, count


    def forward(self, sinputs, tinputs, mask=None, tmask = None, lengths=None):
       
        
        sdensity = []
        scount = []
        sh = []
        for i in range(self.num_domains):
            X = sinputs[i]
            if mask is None:
                cnn_mask = None
            else:
                cnn_mask = mask[i]
            density, h, count = self.forward_temporal(X, cnn_mask, lengths)
            sdensity.append(density)
            scount.append(count)
            sh.append(h)

        _, th, _ = self.forward_temporal(tinputs, tmask, lengths)

        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(sh[i]))), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(th))), dim=1))

        return sdensity, scount, sdomains, tdomains
    
    def to_string(self):
        return "temporal_single"
    