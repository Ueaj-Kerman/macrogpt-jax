"""Test-Time Training (TTT) module implementation.

This module implements the TTT algorithm which learns to adapt a hidden state
during inference using gradient descent on a self-supervised objective.
"""

from .module import *
from .impl import *