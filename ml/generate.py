#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
import sys
import os
import re
import json

# ============================================
# 1. Model definitions (must match training)
# ============================================


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

def load_checkpoint():
    print("Loading model from: model_final2.pt")
    print("This may take a moment...")

    checkpoint = torch.load("model_final2.pt", map_location='cpu', weights_only=False)

    print("\nCheckpoint contents:")
    print("  Keys:", list(checkpoint.keys()))

    return checkpoint


def get_cfg(checkpoint):
    cfg = checkpoint.get('config')
    if cfg is None:
        print("  Warning: 'config' not found - using default")
        cfg = GPTConfig()
    else:
        print(f"  Config: n_layer={cfg.n_layer}, n_emb={cfg.n_emb}, vocab={cfg.vocab_size}")

    return cfg

def get_state_dict(checkpoint):
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
    return state_dict

def check_eval_keys(model, state_dict):
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
# 5. GENERATION
# ============================================

def generate_text(model, prompt, config, max_new_tokens=100, temperature=0.8, top_k=50, 
                  frequency_penalty=0.5, presence_penalty=0.3, eos_penalty=10.0):
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    prompt_tokens = enc.encode(prompt)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = []
    token_counts = {}
    eos_token_id = 50256

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if x.size(1) > config.block_size:
                x = x[:, -config.block_size:]
            logits, _ = model(x)
            logits = logits[:, -1, :]

            # ---------- EOS PENALTY ----------
            if eos_penalty > 0:
                logits[0, eos_token_id] -= eos_penalty
            # -----------------------------------------

            # ---------- REPETITION PENALTIES ----------
            if frequency_penalty > 0 or presence_penalty > 0:
                penalty = torch.zeros_like(logits)
                for token, count in token_counts.items():
                    if frequency_penalty > 0:
                        penalty[0, token] -= frequency_penalty * count
                    if presence_penalty > 0:
                        penalty[0, token] -= presence_penalty
                logits = logits + penalty
            # -----------------------------------------

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

            if xcol.item() == eos_token_id:
                break

            token_id = xcol.item()
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

            x = torch.cat((x, xcol), dim=1)
            generated.append(token_id)

    model.train()
    full_tokens = prompt_tokens + generated
    return enc.decode(full_tokens)


# ============================================
# 6. TEST
# ============================================

def gen_and_print(model, cfg):

    print("\n" + "="*60)
    print("GENERATION")
    print("="*60)

    prompts = [
        "TITLE: Game of Thrones",
        "TITLE: 404.php"
    ]

    for prompt in prompts:    
        print(f"\nPROMPT: {prompt}")
        print("-"*40)
        try:
            output = generate_text(
                model, 
                prompt, 
                cfg, 
                max_new_tokens=800, 
                temperature=0.8, 
                top_k=50,
                frequency_penalty=0.2,
                presence_penalty=0.1
            )
            print(output)
        except Exception as e:
            print(f"Error: {e}")
        print("-"*40)
        print()


def parse_article(title, text: str) -> dict:
    lines = text.strip().splitlines()
    sections = {}
    current_section = None
    current_content = []

    for line in lines:
        if line.startswith("## "):
            if current_section is not None:
                sections[current_section] = "\n".join(current_content).strip()
            section_name = line[3:].strip()
            current_section = section_name
            current_content = []
        else:
            if current_section is not None:
                current_content.append(line)

    if current_section is not None:
        sections[current_section] = "\n".join(current_content).strip()

    return {"title": title, "sections": sections}


def to_json(model, cfg, output_file='articles.json'):
    file_path = 'a.txt'
    with open(file_path, 'r') as file:
        titles = [line.strip() for line in file if line.strip()]

    articles = []

    cnt = 0
    for title in titles:
        prompt = f"TITLE: {title}"
        try:
            output = generate_text(
                model,
                prompt,
                cfg,
                max_new_tokens=800,
                temperature=0.8,
                top_k=50,
                frequency_penalty=0.2,
                presence_penalty=0.1
            )
            parsed = parse_article(title, output)
            if parsed["title"] and parsed["sections"]:
                articles.append(parsed)
            else:
                print(f"Unable to parse for: {title}, trying again later")
                titles.append(title)
        except Exception as e:
            print(f"Error for title {title}: {e}")
        cnt += 1
        if cnt % 10 == 0:
            print(f"{cnt}/{len(titles)} done")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(articles)} articles to file {output_file}")
    return articles

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = load_checkpoint()
    cfg = get_cfg(checkpoint)
    state_dict = get_state_dict(checkpoint)
    print("\nCreating model...")
    model = GPT(cfg)
    model.eval()

    try:
        model = model.to(device)
        print(f"Model moved to {device}")
    except RuntimeError as e:
        print(f"Failed to move to GPU: {e}")
        device = "cpu"
        model = model.cpu()

    check_eval_keys(model, state_dict)
    
    gen_and_print(model, cfg)
    #to_json(model, cfg)

