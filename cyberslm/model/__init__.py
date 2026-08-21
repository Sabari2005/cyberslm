"""
CyberSLM model package.

Public API (grows as phases are added):
    Phase 1: CyberSLMConfig, default_config, RMSNorm, RotaryPositionEmbedding, apply_rope
    Phase 2: MultiHeadSelfAttention, SwiGLUFeedForward
    Phase 3: DecoderBlock, CyberSLM, build_model
    Phase 4: (training engine lives in cyberslm.training)
"""

from cyberslm.model.config import CyberSLMConfig, default_config
from cyberslm.model.norm import RMSNorm
from cyberslm.model.rope import RotaryPositionEmbedding, apply_rope
from cyberslm.model.attention import MultiHeadSelfAttention
from cyberslm.model.ffn import SwiGLUFeedForward
from cyberslm.model.block import DecoderBlock
from cyberslm.model.model import CyberSLM, build_model, count_parameters, model_summary

__all__ = [
    # Phase 1
    "CyberSLMConfig",
    "default_config",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "apply_rope",
    # Phase 2
    "MultiHeadSelfAttention",
    "SwiGLUFeedForward",
    # Phase 3
    "DecoderBlock",
    "CyberSLM",
    "build_model",
    "count_parameters",
    "model_summary",
]
