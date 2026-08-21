"""
Convert a CyberSLM checkpoint to HuggingFace LlamaForCausalLM format.

    python cyberslm/scripts/convert_to_hf.py \
        --checkpoint models/base.pt --out hf_export/base

Why bother: every standard evaluation tool (lm-evaluation-harness, the official
ArithMark-3 script) loads models through AutoModelForCausalLM. A leaderboard
that promises to "independently verify" results can only do so if the weights
load with stock tooling. Converting also makes the published repos usable with
`AutoModelForCausalLM.from_pretrained(...)`.

The architecture is already LLaMA-class -- RoPE, RMSNorm, SwiGLU, tied head, no
biases -- so the mapping is one-to-one EXCEPT for one detail that silently
produces a subtly wrong model if missed:

  RoPE pair layout.
    CyberSLM rotates INTERLEAVED pairs: (x0,x1), (x2,x3), ...   [GPT-J style]
    HF Llama rotates SPLIT-HALF pairs:  (x0,x_{d/2}), ...       [GPT-NeoX style]

  The two are related by a permutation of the q/k projection OUTPUT rows within
  each head. Copying the weights straight across gives a model that loads
  cleanly, runs without error, and produces different logits -- the worst kind
  of bug. This script applies the permutation and then verifies the converted
  model reproduces the original's logits before writing anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cyberslm.model.config import CyberSLMConfig, default_config  # noqa: E402
from cyberslm.model.model import build_model                      # noqa: E402


def interleaved_to_half_perm(head_dim: int) -> torch.Tensor:
    """
    Row permutation turning GPT-J (interleaved) RoPE weights into GPT-NeoX
    (split-half) layout.

    HF index j < d/2 must read our even row 2j; HF index j >= d/2 must read our
    odd row 2(j - d/2) + 1.
    """
    even = torch.arange(0, head_dim, 2)
    odd = torch.arange(1, head_dim, 2)
    return torch.cat([even, odd])


def permute_qk(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Apply the RoPE layout permutation per head to a (n_heads*head_dim, in) matrix."""
    perm = interleaved_to_half_perm(head_dim)
    out_features, in_features = weight.shape
    w = weight.view(n_heads, head_dim, in_features)
    w = w[:, perm, :]
    return w.reshape(out_features, in_features).contiguous()


def load_cyberslm(path: Path, cfg_json: Path | None = None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "config" in payload:
        cfg, state = CyberSLMConfig(**payload["config"]), payload["model_state"]
    else:
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        if cfg_json is not None and Path(cfg_json).exists():
            # A bare state_dict carries no architecture. Falling back to
            # default_config() would silently claim max_seq_len=4096 for a model
            # trained at 2048, so read the run's own config instead.
            raw = json.loads(Path(cfg_json).read_text(encoding="utf-8"))
            m = raw.get("model", raw)
            cfg = CyberSLMConfig(
                vocab_size=m["vocab_size"], hidden_dim=m["hidden_size"],
                num_layers=m["num_layers"], num_heads=m["num_heads"],
                head_dim=m["head_dim"], ffn_hidden_dim=m["ffn_size"],
                max_seq_len=m["max_seq_len"], norm_eps=m.get("norm_eps", 1e-6),
                tie_weights=m.get("weight_tying", True),
            )
        else:
            cfg, state = default_config(), state
    model = build_model(cfg, device=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def build_hf(cfg: CyberSLMConfig):
    from transformers import LlamaConfig, LlamaForCausalLM

    hf_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_dim,
        intermediate_size=cfg.ffn_hidden_dim,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_heads,   # no GQA in this architecture
        hidden_act="silu",                   # SwiGLU
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        rope_theta=float(cfg.rope_base),
        tie_word_embeddings=cfg.tie_weights,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
        pad_token_id=0,
        torch_dtype="float32",
    )
    return LlamaForCausalLM(hf_cfg), hf_cfg


def convert(model, cfg, hf_model) -> None:
    sd = model.state_dict()
    new = {}
    n_heads, head_dim = cfg.num_heads, cfg.head_dim

    new["model.embed_tokens.weight"] = sd["embedding.weight"].clone()
    new["model.norm.weight"] = sd["final_norm.weight"].clone()

    for i in range(cfg.num_layers):
        p = f"layers.{i}"
        h = f"model.layers.{i}"
        # q/k carry the RoPE rotation and therefore need the layout permutation.
        new[f"{h}.self_attn.q_proj.weight"] = permute_qk(
            sd[f"{p}.attn.q_proj.weight"], n_heads, head_dim)
        new[f"{h}.self_attn.k_proj.weight"] = permute_qk(
            sd[f"{p}.attn.k_proj.weight"], n_heads, head_dim)
        # v and o are not rotated, so they copy straight across.
        new[f"{h}.self_attn.v_proj.weight"] = sd[f"{p}.attn.v_proj.weight"].clone()
        new[f"{h}.self_attn.o_proj.weight"] = sd[f"{p}.attn.o_proj.weight"].clone()

        new[f"{h}.mlp.gate_proj.weight"] = sd[f"{p}.ffn.gate_proj.weight"].clone()
        new[f"{h}.mlp.up_proj.weight"] = sd[f"{p}.ffn.val_proj.weight"].clone()
        new[f"{h}.mlp.down_proj.weight"] = sd[f"{p}.ffn.out_proj.weight"].clone()

        new[f"{h}.input_layernorm.weight"] = sd[f"{p}.attn_norm.weight"].clone()
        new[f"{h}.post_attention_layernorm.weight"] = sd[f"{p}.ffn_norm.weight"].clone()

    if not cfg.tie_weights:
        new["lm_head.weight"] = sd["lm_head.weight"].clone()

    missing, unexpected = hf_model.load_state_dict(new, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]  # tied
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch\n  missing={missing}\n  unexpected={unexpected}")
    if cfg.tie_weights:
        hf_model.tie_weights()


@torch.no_grad()
def verify(model, hf_model, cfg, n_trials: int = 4, seq_len: int = 64) -> float:
    """Both models must produce the same logits. This is the whole point."""
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(n_trials):
        ids = torch.randint(0, cfg.vocab_size, (2, seq_len))
        ours, _ = model(ids)
        theirs = hf_model(ids).logits
        worst = max(worst, (ours.float() - theirs.float()).abs().max().item())
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer",
                    default=str(_REPO_ROOT / "tokenizer" / "tokenizer_output" / "tokenizer.model"))
    ap.add_argument("--config-json", default=None,
                    help="architecture JSON for bare state_dict checkpoints")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    out = Path(args.out)
    print("=" * 66)
    print("  CyberSLM -> HuggingFace LlamaForCausalLM")
    print("=" * 66)

    model, cfg = load_cyberslm(Path(args.checkpoint),
                               Path(args.config_json) if args.config_json else None)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  source     : {args.checkpoint}")
    print(f"  params     : {n_params:,}")
    print(f"  arch       : {cfg.num_layers}L d={cfg.hidden_dim} h={cfg.num_heads} "
          f"ffn={cfg.ffn_hidden_dim} ctx={cfg.max_seq_len} vocab={cfg.vocab_size:,}")

    hf_model, hf_cfg = build_hf(cfg)
    convert(model, cfg, hf_model)
    hf_model.eval()

    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"  hf params  : {hf_params:,}  "
          f"({'match' if hf_params == n_params else 'MISMATCH'})")

    worst = verify(model, hf_model, cfg)
    print(f"  max |logit difference| over 4 random batches: {worst:.3e}")
    if worst > args.tolerance:
        print(f"\n  ABORT: exceeds tolerance {args.tolerance:.1e}. Nothing written.",
              file=sys.stderr)
        print("  The RoPE permutation or a weight mapping is wrong; a model that "
              "loads cleanly but scores differently is worse than no model.",
              file=sys.stderr)
        return 1
    print("  logits match -- conversion is faithful")

    out.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(out, safe_serialization=True)

    # Tokenizer: transformers' own converters cannot read this BPE spm file
    # (LlamaTokenizer yields vocab_size 3, SpmConverter raises), so it is
    # rebuilt explicitly and verified against SentencePiece before writing.
    tok_src = Path(args.tokenizer)
    if tok_src.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_hf_tokenizer import build as build_tok, verify as verify_tok
        tok, n_merges = build_tok(tok_src)
        ok, total, failures = verify_tok(tok, tok_src)
        print(f"  tokenizer  : {tok.get_vocab_size():,} vocab, {n_merges:,} merges, "
              f"{ok}/{total} probes match SentencePiece")
        if ok != total:
            print("")
            print("  ABORT: tokenizer does not reproduce SentencePiece.", file=sys.stderr)
            return 1
        tok.save(str(out / 'tokenizer.json'))
        import json as _json
        (out / 'tokenizer_config.json').write_text(_json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": cfg.max_seq_len,
            "bos_token": "<bos>", "eos_token": "<eos>",
            "unk_token": "<unk>", "pad_token": "<pad>",
            "clean_up_tokenization_spaces": False,
        }, indent=2), encoding='utf-8')
        (out / 'special_tokens_map.json').write_text(_json.dumps({
            "bos_token": "<bos>", "eos_token": "<eos>",
            "unk_token": "<unk>", "pad_token": "<pad>",
        }, indent=2), encoding='utf-8')
    else:
        print(f"  tokenizer  : NOT FOUND at {tok_src}", file=sys.stderr)
        return 1

    print(f"\n  written to {out}")
    for f in sorted(out.iterdir()):
        print(f"    {f.stat().st_size/1e6:8.2f} MB  {f.name}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
