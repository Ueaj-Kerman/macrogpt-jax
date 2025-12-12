"""Test-Time Training (TTT) module implementation.

This module implements the TTT algorithm which learns to adapt a hidden state
during inference using gradient descent on a self-supervised objective.
"""

from .module import TTTModel
from .impl import ttt, make_scan_fn, make_update_fn, make_ttt_bwd

__all__ = ['TTTModel', 'ttt', 'make_scan_fn', 'make_update_fn', 'make_ttt_bwd']