import torch 
from torch import nn 
import math 

from TransformerBlock import TransformerBlock 
from Embedder import Embedder 
from PositionalEncoder import PositionalEncoder 

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, d_model=512, num_layer=6, factor=4, n_head=8):
        super(TransformerEncoder, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.factor = factor
        self.n_head = n_head
        self.d_model = d_model

        self.embedding_layer = Embedder(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoder = PositionalEncoder(seq_len=seq_len, d_model=d_model)

        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, n_head=n_head, factor=factor) for i in range(num_layer)])

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)

        return out

x = torch.randint(size=(32, 10), low=1, high=100)
net = TransformerEncoder(seq_len=10, vocab_size=200, num_layer=1)
a = net(x)
print(a.shape)