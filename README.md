useful datasets:

- https://dumps.wikimedia.org/enwiki/latest/?utm_source=chatgpt.com (enwiki-latest-pages-articles.xml.bz2)
- https://www.kaggle.com/datasets/conjuring92/wiki-stem-corpus

**decode-only transformer**

Roadmap:

- bigrams
- single head self-attention
- multi head self-attention
- feed forward
- transformer block
- skip connection
- layer norm
- tokenizing
  - bpe
  - 3.4x compression ratio (for 1041830 characters in the dataset and vocab_size=5000)
  - I will probably use tiktoken in the end (4.3x compressionratio for now)
- building the actual model

![alt text](image.png)
_from https://medium.com/@vipul.koti333/from-theory-to-code-step-by-step-implementation-and-code-breakdown-of-gpt-2-model-7bde8d5cecda_
