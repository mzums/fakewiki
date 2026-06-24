import tiktoken
import torch

with open('wiki_clean.txt', 'r', encoding='utf-8-sig') as f:
    text = f.read()

base_enc = tiktoken.get_encoding("cl100k_base")

my_special_tokens = {
    "<SOA>": 200000,
    "<EOA>": 200001,
}

extended_enc = tiktoken.Encoding(
    name="cl100k_base_extended",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **my_special_tokens},
)

all_tokens = extended_enc.encode(text, allowed_special="all")
decoded_text = extended_enc.decode(all_tokens)

print("Reconstruction correct:", decoded_text == text)