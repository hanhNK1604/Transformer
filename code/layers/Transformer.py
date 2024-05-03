from torch import nn 
import torch 
import math 

from Embedder import Embedder
from PositionalEncoder import PositionalEncoder

from MultiheadAttention import MultiheadAttention 
from TransformerBlock import TransformerBlock
from DecoderBlock import DecoderBlock
from TransformerDecoder import TransformerDecoder
from TransformerEncoder import TransformerEncoder


class Transformer(nn.Module):
  def __init__(self, d_model, src_vocab_size, target_vocab_size, seq_len, num_layer=6, factor=4, n_head=8):
    super(Transformer, self).__init__()
    self.d_model = d_model
    self.src_vocab_size = src_vocab_size
    self.target_vocab_size = target_vocab_size
    self.seq_len = seq_len
    self.num_laye = num_layer
    self.factor = factor
    self.n_head = n_head

    self.encoder = TransformerEncoder(seq_len=seq_len, vocab_size=src_vocab_size, d_model=d_model, num_layer=num_layer, factor=factor, n_head=n_head)
    self.decoder = TransformerDecoder(seq_len=seq_len, vocab_size=target_vocab_size, d_model=d_model, num_layer=num_layer, factor=factor, n_head=n_head)

  def make_target_mask(self, trg): 
    tgt_mask = (trg != 0).unsqueeze(1).unsqueeze(3)
    seq_length = trg.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return tgt_mask
  
  def decode(self, src):
        """
        for inference
        Args:
            src: input to encoder
        """

        batch_size, seq_len = src.shape[0], src.shape[1]
        trg = torch.zeros(size=(batch_size, seq_len), dtype=torch.int)
        trg[:, 0] = 1 

        enc_out = self.encoder(src)
        
        for i in range(1, seq_len):
          trg_mask = self.make_target_mask(trg)
          out = self.decoder(x=trg, encoder_out=enc_out, mask=trg_mask)
          out = out.argmax(-1)[:, i]
          trg[:, i] = out
        
        return trg

  def forward(self, src, trg):
    """
    Args:
        src: input to encoder 
        trg: input to decoder
    out:
        out: final vector which returns probabilities of each target word
    """
    trg_mask = self.make_target_mask(trg)
    enc_out = self.encoder(src)

    outputs = self.decoder(trg, enc_out, trg_mask)
    return outputs