"""
Bayesian Llama model implementation that replaces the last 3 layers with Bayesian MLP layers.
"""
from dataclasses import replace
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

from ueaj.model.llama.llama import LlamaModel, LlamaConfig
from ueaj.model.layer import TransformerLayer, TransformerLayerConfig


class BayesianLlamaModel(LlamaModel):
    """Llama model with the last 3 layers using Bayesian MLPs."""
    
    def __init__(self, config: LlamaConfig, rngs: rng.Rngs):
        # Determine how many layers are regular vs Bayesian
        num_bayesian_layers = min(3, config.num_layers)
        num_regular_layers = config.num_layers - num_bayesian_layers
        
        # Create a modified config with fewer layers for the base model
        base_config = replace(config, num_layers=num_regular_layers)
        
        # Initialize the base model with fewer layers
        super().__init__(base_config, rngs)
        
        # Create Bayesian layer config
        bayesian_layer_config = replace(
            config.layer_config,
            mlp_type="bayesian"
        )
        
        # Create Bayesian tail layers with vmap
        @nnx.split_rngs(splits=num_bayesian_layers)
        @nnx.vmap(axis_size=num_bayesian_layers)
        def create_bayesian_block(rngs: nnx.Rngs):
            return TransformerLayer(bayesian_layer_config, rngs)
        
        self.tail_layers = create_bayesian_block(rngs)
        
        # Store the original config and layer counts
        self.config = config  # Restore original config
        self.num_regular_layers = num_regular_layers
        self.num_bayesian_layers = num_bayesian_layers
    
    def get_activations(self, input_ids: jax.Array, **kwargs) -> jax.Array:
        """Get hidden states without final norm and lm_head projection."""
        # Process through regular layers using parent's method
        act = super().get_activations(input_ids, **kwargs)
        
        # Get kwargs for tail layers
        kwargs = self.default_kwargs(*input_ids.shape, **kwargs)
        
        # Process through Bayesian tail layers
        @nnx.split_rngs(splits=self.num_bayesian_layers)
        @nnx.scan
        @nnx.remat(policy=jax.checkpoint_policies.nothing_saveable)
        def scan_tail(act, layer):
            return layer(act, **kwargs), None
        
        act, _ = scan_tail(act, self.tail_layers)
        
        return act