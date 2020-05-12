import torch
from torch import nn
from torch.nn.utils import rnn
from model_temporal import MDANTemporal
import torch.nn.functional as F
import settings

class MDANTemporalDouble(MDANTemporal):

    def __init__(self, num_domains, image_dim):
        super(MDANTemporalDouble, self).__init__(num_domains, image_dim)
        H, W = image_dim
        self.domains_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(1408, 512, (3, 3)), nn.ReLU(), nn.Conv2d(512, 16, (3,3)), nn.ReLU()) for _ in range(self.num_domains)])
        self.domains_lstm = nn.ModuleList([nn.LSTM(16640, 100, num_layers=3, batch_first=True) for _ in range(self.num_domains)])
        
    
    def forward_temporal(self, X, mask=None, lengths=None):
        N, T, C, H, W = X.shape
        X = X.reshape(T*N, C, H, W)
        if mask is not None:
            mask = mask.reshape(T*N, 1, H, W)

        h, density = super().forward_cnn(X, mask)

        _, count_fcn, count_lstm = self.forward_lstm((N, T, C, H, W), density, mask, lengths)
       
        count = count_fcn + count_lstm  # predicted vehicle count

        return density, h, count

    


    def forward(self, sinputs, tinputs, mask=None, lengths=None):
        #return [], [], []
        N, T, C, H, W = sinputs[0].shape
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
            h0 = self.init_hidden(N)
            h_lstm = self.domains_conv[i](sh[i])
            h_lstm = h_lstm.reshape(N, T, -1)
            h_lstm, _ = self.domains_lstm[i](h_lstm, h0)
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i].apply(self.flatten[i](h_lstm))), dim=1))
            h0 = self.init_hidden(N)
            h_lstm = self.domains_conv[i](th)
            h_lstm = h_lstm.reshape(N, T, -1)
            h_lstm, _ = self.domains_lstm[i](h_lstm, h0)
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i].apply(self.flatten[i](h_lstm)))))

        return sdensity, scount, sdomains, tdomains
        
        