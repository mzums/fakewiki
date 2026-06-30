from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import json

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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # all the blocks in the transformer
            ln_f = nn.LayerNorm(config.n_emb),                                  # linear layer at the end
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward token and position embeddings
        # [0, 1, 2, ..., T-1] of shape (T,)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # shape (T)
        pos_emb = self.transformer.wpe(pos) # (T,) -> (T, n_emb)
        tok_emb = self.transformer.wte(idx) # (B, T) -> (B, T, n_emb)
        x = tok_emb + pos_emb               # (B, T, n_emb)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the layernorm and the classifier
        x = self.transformer.ln_f(x)        # (B, T, n_emb)
        logits = self.lm_head(x)            # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),    # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),   # 350M
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # ignore buffers, they shouldn't be trained
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        # originally the linear layers were Conv1D with kernel=1 and were kept in (kernel_size, in_channels, out_channels)
        # in pytorch linear layers require (out, in) (it's more afficient) so we need to transpose the weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


import tiktoken

class DataLoaderLite:
    
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('../dev/wiki_clean.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    
# --------------------------------------

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

num_return_sequences = 5
max_length = 30

train_loader = DataLoaderLite(B=4, T=32)

# get logits
#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.to(device)
#logits, loss = model(x, y)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()  }")


print(loss)
#import sys; sys.exit(0)

# prefix tokens
# load data
flat_tokens = np.load('../dev/tokens_flat.npy')   # (total_length,)
lengths = np.load('../dev/tokens_lengths.npy')    # (liczba_artykułów,)

# get tokens for every article
all_tokens = []
start = 0
for length in lengths:
    all_tokens.append(flat_tokens[start:start+length].tolist())
    start += length

# get article
tokens_list = all_tokens[0]

# get starting tensor
tokens = torch.tensor(tokens_list, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)   # (num_return_sequences, seq_len)
x = tokens.to('cuda')


def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    # delete all the 0 bytes
    tokens = tokens.replace(b'\x00', b'')
    text = tokens.decode("utf-8", errors="replace")
    return text


with open('../dev/vocab.json', 'r', encoding='utf-8') as f:
    vocab_as_strings = json.load(f)

vocab = [token.encode('utf-8') for token in vocab_as_strings]


# generating
# x is (B, T) where B=5, T=8
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get logits
    with torch.no_grad():
        logits = model(x)           # (B, T, vocab_size)
        # predictions for the last token, inefective but okay here
        logits = logits[:, -1, :]   # (B, T, vocab_size) -> (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # top-k sampling of 50 (huggingface pipeline default) for every element in the batch
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # both(5, 50)
        # select a token from the top-k probabilities, returns an index
        ix = torch.multinomial(topk_probs, 1)
        # gather the corresponding indices (gets the elements of a given index)
        xcol = torch.gather(topk_indices, -1, ix)   # (B, 1)
        # append to the sequence (x += xcol)
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = decode(tokens)
    print(">", decoded)
