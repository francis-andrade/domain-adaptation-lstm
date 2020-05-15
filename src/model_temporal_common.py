import torch
from torch import nn
from torch.nn.utils import rnn
from model_temporal import MDANTemporal
import torch.nn.functional as F

class MDANTemporalCommon(MDANTemporal):

    def __init__(self, num_domains, image_dim):
        super(MDANTemporalCommon, self).__init__(num_domains, image_dim)
        #self.domains = nn.ModuleList([nn.Sequential(nn.Linear(1000, 10), nn.Linear(10, 2)) for _ in range(self.num_domains)])
      
    
    def forward_temporal(self, X, mask=None, lengths=None):
        N, T, C, H, W = X.shape
        X = X.reshape(T*N, C, H, W)
        if mask is not None:
            mask = mask.reshape(T*N, 1, H, W)

        _, density = super().forward_cnn(X, mask)

        h, count_fcn, count_lstm = self.forward_lstm((N, T, C, H, W), density, mask, lengths)
       
        count = count_fcn + count_lstm  # predicted vehicle count

        return density, h, count


    def forward(self, sinputs, tinputs, mask=None, lengths=None):
       
        
        sdensity = []
        scount = []
        sh = []
        for i in range(self.num_domains):
            X = sinputs[i]
            density, h, count = self.forward_temporal(X, mask, lengths)
            sdensity.append(density)
            scount.append(count)
            sh.append(h)

        _, th, _ = self.forward_temporal(tinputs, mask, lengths)  

        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(sh[i]))), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.flatten[i](self.grls[i].apply(self.flatten[i](th)))))

        return sdensity, scount, sdomains, tdomains
    