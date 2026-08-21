import sys
from pathlib import Path

# Add the root directory (which contains 'cyberslm') to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cyberslm.model.model import CyberSLM as Stage1CyberSLM
from cyberslm.model.config import CyberSLMConfig

class CyberSLM(Stage1CyberSLM):
    """
    Adapter class to bridge SFT's ModelConfig to Stage 1's CyberSLMConfig,
    allowing us to use the original Stage 1 model architecture verbatim.
    """
    def __init__(self, cfg):
        # Convert SFT ModelConfig to Stage 1 CyberSLMConfig
        stage1_cfg = CyberSLMConfig(
            vocab_size=cfg.vocab_size,
            hidden_dim=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            ffn_hidden_dim=cfg.ffn_size,
            max_seq_len=cfg.max_seq_len,
            tie_weights=cfg.weight_tying,
            bias=cfg.bias,
            dropout=cfg.dropout,
            norm_eps=cfg.norm_eps,
        )
        super().__init__(stage1_cfg)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Wrapper around Stage 1's forward pass.
        Stage 1 returns (logits, all_attn_weights); the SFT trainer expects
        just the logits Tensor. The key-padding ``attention_mask`` (1=keep,
        0=pad) is forwarded so right-padded batches do not corrupt real tokens.
        """
        logits, _ = super().forward(input_ids, attention_mask=attention_mask)
        return logits
