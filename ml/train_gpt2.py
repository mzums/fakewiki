from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# ---------------------------------------------

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.c_attn = nn.Linear(config.n_emb, 3*config.n_emb)
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        # buffer because it should be constant, it's not a parameter
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
                            # (1,1) is because later we use self.bias[:,:,:T,:T]
                            # then we want to broadcast it to (batch_size, n_head)
        
    def forward(self, x):
        # (batch_size, token_len, n_emb)
        B, T, C = x.size()
        # k,q,v are not learned, they are only acivations computed for every input
        # the model learns only weights in c_attn and weights in c_proj
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_emb, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # for every query all the keys should sum to 1
        att = F.softmax(att, dim=-1)
        y = att @ v
        # to revert .transpose(1, 2) ^
        # transpose doesn't physically revert data, only changes metadata and shape but view requires contiguous data so we must physically revert the data
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_emb, 4*config.n_emb)
        # historical, it was like this in GPT2, so I use it
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_emb, config.n_emb)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # norm before attention (unlike in the original transformer paper)
        self.ln_1 = nn.LayerNorm(config.n_emb)  # layer norm
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))     # residual connection
        x = x + self.mlp(self.ln_2(x))      # residual connection
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257     # 50k BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),                # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_emb),                # positional encodings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]), # all the blocks in the transformer
            ln_f = nn.LayerNorm(config.n_emb),                                  # linear layer at the end
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)