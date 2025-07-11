from ueaj.utils.argutils import either, promote_fp8
from ueaj.utils.collections import LOW_PRECISION
from ueaj.utils.config import DEFAULT_ACCUM_TYPE
from ueaj.utils.gradutils import astype_fwd_noop_bwd, debug_dtype
from ueaj.utils.tensorutil import chunked_scan, precision_aware_update

__all__ = [
    "either",
    "promote_fp8",
    "LOW_PRECISION",
    "DEFAULT_ACCUM_TYPE",
    "astype_fwd_noop_bwd",
    "debug_dtype"
]