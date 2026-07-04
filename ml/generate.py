#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import tiktoken
import sys
import os

# ============================================
# 1. Model definitions (must match training)
# ============================================
from dataclasses import dataclass
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
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
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)
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
            wte=nn.Embedding(config.vocab_size, config.n_emb),
            wpe=nn.Embedding(config.block_size, config.n_emb),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_emb),
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
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


# ============================================
# 2. LOADING CHECKPOINT
# ============================================
print("Loading model from: model_final.pt")
print("This may take a moment...")

checkpoint = torch.load("model_final.pt", map_location='cpu', weights_only=False)

print("\nCheckpoint contents:")
print("  Keys:", list(checkpoint.keys()))

cfg = checkpoint.get('config')
if cfg is None:
    print("  Warning: 'config' not found - using default")
    cfg = GPTConfig()
else:
    print(f"  Config: n_layer={cfg.n_layer}, n_emb={cfg.n_emb}, vocab={cfg.vocab_size}")

state_dict = checkpoint['model_state_dict']

# Check for prefixes
keys_sample = list(state_dict.keys())[:5]
print(f"\n  Sample keys in state dict:")
for k in keys_sample:
    print(f"    {k}")

# Remove 'module.' and '_orig_mod.' prefixes
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        k = k[7:]
    if k.startswith('_orig_mod.'):
        k = k[9:]
    if k.startswith('.'):
        k = k[1:]
    new_state_dict[k] = v
state_dict = new_state_dict
print("  Removed 'module.' and '_orig_mod.' prefixes if present")

# ============================================
# 3. CREATE MODEL AND LOAD WEIGHTS
# ============================================
print("\nCreating model...")
model = GPT(cfg)
model.eval()

model_keys = set(model.state_dict().keys())
loaded_keys = set(state_dict.keys())
print(f"Model keys: {len(model_keys)}")
print(f"Loaded keys: {len(loaded_keys)}")
if model_keys != loaded_keys:
    print("Differences:")
    print("  Missing in loaded:", model_keys - loaded_keys)
    print("  Extra in loaded:", loaded_keys - model_keys)
else:
    print("All keys match!")


try:
    model.load_state_dict(state_dict, strict=True)
    print("Weights loaded successfully (strict=True)")
except RuntimeError as e:
    print(f"\nError with strict=True: {str(e)[:300]}...")
    print("\nAttempting with strict=False (ignoring missing/unexpected keys)...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        if len(missing) <= 10:
            print("  Examples:", missing[:5])
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            print("  Examples:", unexpected[:5])
    print("Weights loaded with strict=False")

# ============================================
# 4. MOVE TO DEVICE
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = model.to(device)
    print(f"Model moved to {device}")
except RuntimeError as e:
    print(f"Failed to move to GPU: {e}")
    device = "cpu"
    model = model.cpu()

# ============================================
# 5. GENERATION
# ============================================
enc = tiktoken.get_encoding('gpt2')

def generate_text(prompt, max_new_tokens=200, temperature=0.0, top_k=1):
    prompt_tokens = enc.encode(prompt)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = []
    
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if x.size(1) > cfg.block_size:
                x = x[:, -cfg.block_size:]
            
            logits, _ = model(x)
            logits = logits[:, -1, :]
            
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_k is not None:
                    topk_probs, topk_indices = torch.topk(probs, top_k)
                    ix = torch.multinomial(topk_probs, 1)
                    xcol = torch.gather(topk_indices, -1, ix)
                else:
                    xcol = torch.multinomial(probs, 1)
            else:
                xcol = logits.argmax(dim=-1, keepdim=True)
            
            if xcol.item() == 50256:  # EOS token
                print("EOS token encountered - stopping generation")
                break
            
            x = torch.cat((x, xcol), dim=1)
            generated.append(xcol.item())
    
    print(f"Generated {len(generated)} tokens (max_new_tokens={max_new_tokens})")
    full_tokens = prompt_tokens + generated
    return enc.decode(full_tokens)

# ============================================
# 6. TEST
# ============================================
print("\n" + "="*60)
print("GENERATION")
print("="*60)

prompts = [
    "TITLE: Albert Einstein",
]

for p in prompts:    
    print(f"\nPROMPT: {p}")
    print("-"*40)
    try:
        output = generate_text(p, max_new_tokens=1000, temperature=0.6, top_k=15)
        print(output)
    except Exception as e:
        print(f"Error: {e}")
    print("-"*40)
    print()