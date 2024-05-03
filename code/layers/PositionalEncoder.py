import torch 
from torch import nn 
import math 


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, self.d_model)
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x
               
    
# test
# a = torch.rand(size=(32, 10, 512))
# net = PositionalEncoder(seq_len=10, d_model=512)
# b = net(a)
# b.requires_grad
# b.shape