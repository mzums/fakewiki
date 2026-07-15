import torch
import math
from dataclasses import dataclass
import numpy as np
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.c_attn = nn.Linear(config.n_emb, 3*config.n_emb)
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_emb, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_emb, 4*config.n_emb)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_emb: int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            wpe = nn.Embedding(config.block_size, config.n_emb),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emb),
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, token_file="tokens_dyk.bin", 
                 split='train', val_frac=0.05):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        
        # open in memory mapped mode
        self.tokens = np.memmap(token_file, dtype=np.uint32, mode='r')
        total_tokens = len(self.tokens)
        
        split_idx = int(total_tokens * (1 - val_frac))
        if split == 'train':
            self.tokens = self.tokens[:split_idx]
        else:
            self.tokens = self.tokens[split_idx:]
            
        self.num_tokens = len(self.tokens)
        print(f"Loaded {self.num_tokens} tokens for {split} split")
        print(f"1 epoch = {self.num_tokens // (B*T)} batches")
        
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        pos = self.current_position
        buf = torch.from_numpy(self.tokens[pos : pos + B*T + 1]).to(torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B*T*self.num_processes + 1) > self.num_tokens:
            if self.split == 'train':
                self.current_position = self.B * self.T * self.process_rank
            else:
                self.current_position = self.B * self.T * self.process_rank
        return x, y
    



@torch.no_grad()
def evaluate_loss(model, val_loader, grad_accum_steps, device, ddp):
    model.eval()
    loss_accum = 0.0
    num_batches = 20  # num bathes toaverage
    val_loader.current_position = val_loader.B * val_loader.T * val_loader.process_rank
    for micro_step in range(num_batches):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / num_batches   # avg
        loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    model.train()
    return loss_accum.item()



# ---------- CONFIGURATION ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

B, T = 16, 1024
total_batch_size = 131072
ddp = False
ddp_world_size = 1
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

# read checkpoint
checkpoint = torch.load("model_final2.pt", map_location=device, weights_only=False)
config = checkpoint['config']
state_dict = checkpoint['model_state_dict']

new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('_orig_mod.'):
        new_state_dict[key[len('_orig_mod.'):]] = value
    else:
        new_state_dict[key] = value

# reads weights
model = GPT(config)
model.load_state_dict(new_state_dict, strict=True)

model.to(device)
model = torch.compile(model)

# if ddp:
#     model = DDP(model, device_ids=[ddp_local_rank])
# raw_model = model.module if ddp else model

raw_model = model

train_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split='train')
val_loader   = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split='val')

max_lr = 6e-4
min_lr = max_lr * 0.1   # 6e-5
learning_rate = min_lr   # const
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, device=device)

extra_steps = 250

for step in range(extra_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = B * T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    if step % 50 == 0:
        print(f"step {step}, loss: {loss_accum.item():.6f}, dt: {dt:.2f}s, tok/sec: {tokens_per_sec:.2f}")

    if step % 500 == 0: 
        val_loss = evaluate_loss(raw_model, val_loader, grad_accum_steps, device, ddp=False)
        print(f"--- validation loss: {val_loss:.6f} ---")

torch.save({
    'model_state_dict': raw_model.state_dict(),
    'config': config
}, "model_dyk.pt")
