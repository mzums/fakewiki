import regex as re

with open('wiki_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()

def prepare_training_sequences(articles, gpt2pat):
    all_seq = []
    for article in articles:
        matches = gpt2pat.findall(article)

        for match in matches:
            seq = list(match.encode("utf-8")) + [0]
            all_seq.append(seq)

    return all_seq

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def encode(text):
    tokens = list(text.encode("utf-8"))
    tokens.append(0)
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p:merges.get(p, float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    # delete all the 0 bytes
    tokens = tokens.replace(b'\x00', b'')
    text = tokens.decode("utf-8", errors="replace")
    return text

def get_stats_all(seqences):
    counts = {}
    for seq in seqences:
        for i in range(len(seq)-1):
            pair = (seq[i], seq[i+1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_all(seqences, pair, idx):
    new_seqences = []
    for seq in seqences:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq)-1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
                new_seq.append(idx)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_seqences.append(new_seq)
    return new_seqences

article_pat = re.compile(r"<SOA>(.*?)<EOA>", re.DOTALL)
articles = re.findall(article_pat, text)

init_tokens = list(text.encode("utf-8"))

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
train_seq = prepare_training_sequences(articles, gpt2pat)

vocab_size = 5000
num_merges = vocab_size - 256

merges = {}
for i in range(num_merges):
    stats = get_stats_all(train_seq)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    train_seq = merge_all(train_seq, pair, idx)
    merges[pair] = idx
    if i % 100 == 0:
        print(f"iteration: {i}")

with open('merges.bpe', 'w', encoding='utf-8') as f:
    for pair, idx in sorted(merges.items(), key=lambda item: item[1]):
        f.write(f"{pair[0]} {pair[1]}\n")

vocab = [bytes([i]) for i in range(256)]
vocab.extend([b''] * (vocab_size - 256))

for i in range(256):
    vocab[i] = bytes([i])

for pair, idx in sorted(merges.items(), key=lambda item: item[1]):
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

SOA_ID = vocab_size
EOA_ID = vocab_size + 1
vocab.extend([b""] * (EOA_ID - len(vocab) + 1))
vocab[SOA_ID] = b"<SOA>"
vocab[EOA_ID] = b"<EOA>"


all_tokenized_articles = []

for article in articles:
    fragments = gpt2pat.findall(article)
    tokenized_article = []
    
    for fragment in fragments:
        tokenized_fragment = encode(fragment)
        tokenized_article.extend(tokenized_fragment)
    
    final_article_tokens = [SOA_ID] + tokenized_article + [EOA_ID]
    all_tokenized_articles.append(final_article_tokens)

all_tokens = []
for article_tokens in all_tokenized_articles:
    all_tokens.extend(article_tokens)

print("before length:", len(init_tokens))
print("after length:", len(all_tokens))
print(f"compression ratio: {len(init_tokens) / len(all_tokens):.2f}x")

decoded_text = decode(all_tokens)
print("Reconstruction correct:", decoded_text == text)
print(len(decoded_text))