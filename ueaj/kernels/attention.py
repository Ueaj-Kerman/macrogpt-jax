"""Attention kernel shim around kvax.

Single re-export point for kvax-provided primitives. Replace the imports here
if/when we swap kvax out for a custom triton kernel.
"""

from kvax.ops.flash_attention_clean import flash_attention, create_attention_mask
from kvax.utils.common import FlashAttentionParamsConfig
from kvax.utils import PADDING_SEGMENT_ID

__all__ = [
    "flash_attention",
    "create_attention_mask",
    "FlashAttentionParamsConfig",
    "PADDING_SEGMENT_ID",
]
