import torch 
from torch import nn 
from PositionalEncoding import * 
from EncoderLayer import * 
from DecoderLayer import * 

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")


if tokenizer.bos_token is None:
    tokenizer.bos_token = '<bos>'
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<bos>')

if tokenizer.eos_token is None:
    tokenizer.eos_token = '<eos>'
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<eos>')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model  
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        tgt_sub_mask = torch.tril(torch.ones((tgt.size(1), tgt.size(1)), device=tgt.device)).bool()
        tgt_mask = tgt_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)
        
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.dropout(self.positional_encoding(src))
        
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.dropout(self.positional_encoding(tgt))
        
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        
        output = self.fc(tgt)
        return output
