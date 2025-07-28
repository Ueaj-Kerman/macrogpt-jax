"""
Kvax context manager for handling mesh and attention specs.
"""
import os
os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"  # Required for kvax

import jax
from kvax.utils import attention_specs
from kvax.ops import create_attention_mask
from contextlib import contextmanager


class KvaxContext:
    """Manages kvax mesh and attention specs for the model."""
    
    def __init__(self):
        # Create single device mesh
        devices = jax.devices()[:1]  # Use first device
        self.mesh = jax.sharding.Mesh(devices, ('x',))
        
        # Default attention specs for single device (no sharding)
        self.query_specs = (None, None, None, None)
        self.kv_specs = (None, None, None, None)
    
    @contextmanager
    def __call__(self):
        """Context manager for kvax operations."""
        with self.mesh:
            with attention_specs(
                query_specs=self.query_specs,
                kv_specs=self.kv_specs
            ):
                yield self
    
# Global instance for convenience
_kvax_context = None

def get_kvax_context():
    """Get or create the global kvax context."""
    global _kvax_context
    if _kvax_context is None:
        _kvax_context = KvaxContext()
    return _kvax_context