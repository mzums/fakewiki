import tiktoken
import numpy as np
import os
import time

#input_file = "../dev/wiki_clean.txt"
input_file = "dyk.txt"
output_file = "tokens_dyk.bin"

total_size = os.path.getsize(input_file)
print(f"file size: {total_size / (1024**3):.2f} GB")
print(f"tokenization {input_file}...")
print("-" * 60)

enc = tiktoken.get_encoding('gpt2')

processed_bytes = 0
total_tokens = 0
start_time = time.time()
chunk_size = 1024 * 1024  # 1 MB
last_log_mb = 0

with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'wb') as fout:
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break

        tokens = enc.encode(chunk, allowed_special={"<|endoftext|>"})
        np.array(tokens, dtype=np.uint32).tofile(fout)

        chunk_bytes = len(chunk.encode('utf-8'))
        processed_bytes += chunk_bytes
        total_tokens += len(tokens)

        # Log every 100 MB
        current_mb = processed_bytes // (1024 * 1024)
        if current_mb >= last_log_mb + 100:
            last_log_mb = current_mb
            elapsed = time.time() - start_time
            speed = total_tokens / elapsed if elapsed > 0 else 0
            percent = (processed_bytes / total_size) * 100
            eta_seconds = (total_size - processed_bytes) / (processed_bytes / elapsed) if elapsed > 0 else 0
            eta_min = eta_seconds / 60

            print(f"  {percent:.1f}% | {current_mb} MB | "
                  f"{total_tokens:,} tokenów | "
                  f"{speed:.0f} tok/s | "
                  f"ETA: ~{eta_min:.1f} min")

elapsed = time.time() - start_time
print("-" * 60)
print(f"✅ Preprocessing finished!")
print(f"    Saved {total_tokens:,} tokens")
print(f"    Time: {elapsed:.1f} s ({elapsed/60:.1f} min)")
print(f"    avg speed: {total_tokens / elapsed:.0f} tokens/s")
print(f"    output file: {output_file} ({os.path.getsize(output_file) / (1024**3):.2f} GB)")