import json
from pathlib import Path
import sentencepiece as spm

jsonl_path = Path(r"C:\Users\abina\OneDrive\Desktop\SLm Dataset\tokenizer\Final.jsonl")
model_path = Path(r"C:\Users\abina\OneDrive\Desktop\SLm Dataset\tokenizer\tokenizer_output\tokenizer.model")

sp = spm.SentencePieceProcessor()
sp.load(str(model_path))

total_documents = 0
total_tokens = 0
min_tokens = float("inf")
max_tokens = 0

with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = obj.get("text", "")
        if not text:
            continue

        n_tokens = len(sp.encode(text, out_type=int))

        total_documents += 1
        total_tokens += n_tokens
        min_tokens = min(min_tokens, n_tokens)
        max_tokens = max(max_tokens, n_tokens)

average_tokens = total_tokens / total_documents if total_documents else 0

print("=" * 50)
print(f"Documents       : {total_documents:,}")
print(f"Total Tokens    : {total_tokens:,}")
print(f"Average Tokens  : {average_tokens:.2f}")
print(f"Minimum Tokens  : {min_tokens}")
print(f"Maximum Tokens  : {max_tokens}")
print("=" * 50)