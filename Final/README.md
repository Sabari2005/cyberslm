# CyberSLM — Final models

Self-contained inference package. Two trained models, the tokenizer, and the
minimum code needed to run them.

```
Final/
├── infer_base.py     ← run the base model (text continuation)
├── infer_chat.py     ← run the instruct model (question answering)
├── models/
│   ├── base.pt                       134 MB  base weights, inference
│   ├── instruct.pt                   128 MB  instruction-tuned weights
│   ├── base_training_checkpoint.pt   384 MB  full checkpoint (resume training)
│   └── instruct_config.json                  architecture the SFT run used
├── tokenizer/tokenizer.model         32,000-piece SentencePiece BPE
├── cyberslm/model/                   architecture (both models use it)
└── cyberslm_sft/                     prompt formatter + config for chat
```

`base.pt` is `base_training_checkpoint.pt` with the optimizer, scheduler and RNG
state stripped — 403 MB → 134 MB. Identical weights; only use the full file if
you want to *resume training*.

---

## Requirements

```bash
pip install torch sentencepiece
```

Runs on CPU at roughly 45–60 tokens/sec. No GPU needed.

---

## Commands

### Instruct model — ask it questions

```bash
python Final/infer_chat.py --prompt "What is SQL injection and how do I prevent it?"
```

Interactive chat:

```bash
python Final/infer_chat.py --interactive
```

Options:

```bash
python Final/infer_chat.py \
    --prompt "What is a buffer overflow?" \
    --max-new-tokens 200 \
    --temperature 0.0          # 0 = greedy/deterministic (recommended)
```

Real output from that first command:

```
SQL injection (SQLi) is a security vulnerability that allows attackers to
manipulate database queries by injecting malicious SQL code through input
fields. It occurs when user-supplied data is improperly sanitized, allowing
attackers to manipulate the database. For example, if an attacker submits a
username like `admin' --` as the username, the query becomes:
`SELECT * FROM users WHERE username = '[input]' AND password = '[input]'`.
This could lead to unauthorized data access or data breaches.
```

### Base model — text continuation

The base model has **not** been instruction-tuned. Give it the *start of a
sentence* and it continues. Ask it a question and it will continue the question
rather than answer it — use `infer_chat.py` for questions.

```bash
python Final/infer_base.py --prompt "SQL injection is"
```

Interactive:

```bash
python Final/infer_base.py --interactive
```

Options:

```bash
python Final/infer_base.py \
    --prompt "A buffer overflow occurs when" \
    --max-new-tokens 120 \
    --temperature 0.8 \        # 0 = greedy
    --top-k 50 --top-p 0.95 \
    --repetition-penalty 1.15
```

Real output:

```
SQL injection is a common issue in the web interface of Cisco IOS and IOS XE
Software. It has been declared as critical for its security, integrity, and
availability. The vulnerability exists because the affected software does not
properly validate user-supplied input...
```

### Check a model loads

```bash
python Final/infer_base.py --model-info
```

### Force CPU or GPU

```bash
python Final/infer_chat.py --prompt "..." --device cpu
python Final/infer_chat.py --prompt "..." --device cuda
```

---

## What these models are

| | base | instruct |
|---|---|---|
| parameters | 33,531,264 | 33,531,264 |
| layers / d_model / heads | 12 / 384 / 6 | same |
| context | 2048 | 2048 |
| vocab | 32,000 | 32,000 |
| trained on | 786M tokens (4.04 epochs of a 194.8M-token security corpus) | + 23,540 conversations, 3 epochs |
| held-out perplexity | **10.62** | — |
| validation loss | 2.0247 | 2.2627 (response tokens) |

## What to expect

**Be realistic about this.** At 33.5M parameters the instruct model learned the
*shape* of a good answer — markdown structure, numbered steps, worked examples,
mitigation sections — and is frequently wrong about the *content*.

Measured over 8 greedy prompts:

| category | mean 8-gram repetition | stopped on EOS |
|---|---:|---:|
| security | 18.7% | 1 / 4 |
| general | 18.9% | 1 / 2 |
| code | 19.9% | 0 / 2 |

Well-covered in-domain questions (SQL injection, XSS) come out correct. Broader
questions produce confident nonsense: asked to contrast symmetric and asymmetric
encryption it answered about hashing and IKE; asked about buffer overflows it
returned a circular definition. Code generation collapses into repetition.

Only 2 of 8 prompts stopped on their own; the rest ran to the token limit.

Use it as a demonstration of a working pipeline and a base for scaling. Do not
use it as a factual reference, and do not run its generated code.

Full measurements: `../runs/reports/FINAL_REPORT.md`.
