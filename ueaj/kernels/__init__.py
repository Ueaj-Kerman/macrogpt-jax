"""Low-level attention kernels and kvax wrappers.

Anything that touches a custom triton kernel or wraps a kvax op lives here.
The rest of the codebase should import from `ueaj.kernels` rather than `kvax.*`
directly so we have a single chokepoint for kernel-level changes.
"""

from ueaj.kernels.attention import (
    flash_attention,
    create_attention_mask,
    FlashAttentionParamsConfig,
    PADDING_SEGMENT_ID,
)

__all__ = [
    "flash_attention",
    "create_attention_mask",
    "FlashAttentionParamsConfig",
    "PADDING_SEGMENT_ID",
]
