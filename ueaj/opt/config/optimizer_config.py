"""Full optimizer configuration system with tree and tensor list access support.

This module provides a complete optimizer configuration system that supports:
- List accesses on trees: opt[['mlp', 'attn'], ...] = optimizer
- List accesses on tensors: opt['param', [1, 2, 3]] = optimizer
- Efficient tensor slicing: opt['param', :8k] = opt1, opt['param', 8k:] = opt2
- Aggregated optimizer calls (one init/update per optimizer instance)
- Complex nested patterns with tensor indexing
"""

import optax
from flax import nnx
from optax import GradientTransformation
from typing import Dict, Tuple, Any, List, Union, Optional, Set
from collections import defaultdict
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from .index_set import IndexSet
from .optimizer_state import TensorRegion, TensorSplitter


@dataclass
class PatternNode:
    """Node in the pattern tree supporting list-based access."""
    optimizer: Optional[GradientTransformation] = None
    children: Dict[str, 'PatternNode'] = field(default_factory=dict)
    tensor_splitter: Optional[TensorSplitter] = None
    is_wildcard: bool = False
    
    def get_or_create_child(self, key: str) -> 'PatternNode':
        if key not in self.children:
            self.children[key] = PatternNode()
        return self.children[key]


@dataclass(frozen=True)
class ParamAccess:
    """Represents access to a parameter or tensor slice."""
    tree_path: Tuple[str, ...]
    tensor_slices: Optional[Tuple[Any, ...]] = None
    
    def __hash__(self):
        if self.tensor_slices:
            hashable_slices = []
            for s in self.tensor_slices:
                if isinstance(s, slice):
                    hashable_slices.append((s.start, s.stop, s.step))
                elif isinstance(s, list):
                    hashable_slices.append(tuple(s))
                else:
                    hashable_slices.append(s)
            return hash((self.tree_path, tuple(hashable_slices)))
        return hash((self.tree_path, None))


class OptimizerConfig:
    """Full optimizer configuration with tree and tensor list access.
    
    Examples:
        >>> config = OptimizerConfig(model)
        >>> 
        >>> # List access on trees
        >>> config[['mlp', 'attn']] = optax.adam(1e-3)
        >>> 
        >>> # List access on tensors
        >>> config['embeddings', [0, 1, 2, 999]] = optax.adamw(2e-3)
        >>> 
        >>> # Efficient slicing
        >>> config['weights', :8000] = optax.lion(5e-4)
        >>> config['weights', 8000:] = optax.sgd(1e-2)
        >>> 
        >>> # Shared optimizer instances
        >>> opt1 = optax.adam(1e-3)
        >>> config['mlp'] = opt1
        >>> config['attn'] = opt1  # Shares state with mlp
        >>> 
        >>> # Complex nested patterns
        >>> config['mlp', 'fused_proj'] = optax.lion(1e-3)
        >>> config['mlp', 'fused_proj', 0] = optax.adamw(5e-4)
    """
    
    def __init__(self, model: nnx.Module):
        self.model = model
        self.root = PatternNode()
        self._param_shapes = None
        self._optimizer_instances = {}  # Track unique optimizer instances
        
    def __setitem__(self, key, optimizer: GradientTransformation):
        """Assign optimizer to pattern with support for lists."""
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)
        
        # Track optimizer instance
        opt_id = id(optimizer)
        if opt_id not in self._optimizer_instances:
            self._optimizer_instances[opt_id] = optimizer
            
        self._insert_pattern(key, optimizer)
    
    def _insert_pattern(self, pattern: Tuple, optimizer: GradientTransformation):
        """Insert pattern into tree with list expansion."""
        # Separate tree path from tensor indices
        tree_parts = []
        tensor_parts = []
        in_tensor = False
        
        for elem in pattern:
            if not in_tensor and self._is_tree_element(elem):
                tree_parts.append(elem)
            else:
                in_tensor = True
                tensor_parts.append(elem)
        
        # Expand tree lists into individual paths
        tree_paths = self._expand_tree_lists(tree_parts)
        
        # Insert each expanded path
        for tree_path in tree_paths:
            self._insert_single_pattern(tree_path, tensor_parts, optimizer)
    
    def _is_tree_element(self, elem) -> bool:
        """Check if element is a tree path element (not tensor index)."""
        if isinstance(elem, str) or elem is ...:
            return True
        if isinstance(elem, list) and all(isinstance(x, str) for x in elem):
            return True
        return False
    
    def _expand_tree_lists(self, tree_parts: List) -> List[Tuple[str, ...]]:
        """Expand lists in tree path to individual paths."""
        if not tree_parts:
            return [()]
        
        expanded = [[]]
        for part in tree_parts:
            if isinstance(part, list):
                # Expand list
                new_expanded = []
                for path in expanded:
                    for item in part:
                        new_expanded.append(path + [item])
                expanded = new_expanded
            else:
                # Add to all paths
                for path in expanded:
                    path.append(part)
        
        return [tuple(path) for path in expanded]
    
    def _insert_single_pattern(self, tree_path: Tuple, tensor_parts: List, 
                                optimizer: GradientTransformation):
        """Insert a single expanded pattern."""
        node = self.root
        
        # Navigate tree path
        for elem in tree_path:
            if elem is ...:
                node.is_wildcard = True
                if not tensor_parts:
                    node.optimizer = optimizer
            else:
                node = node.get_or_create_child(elem)
        
        # Handle tensor slicing
        if tensor_parts:
            # Get parameter shape
            shape = self._get_param_shape(tree_path)
            if not shape:
                raise ValueError(f"No parameter found at path {'/'.join(tree_path)} for tensor slicing")
                
            if node.tensor_splitter is None:
                node.tensor_splitter = TensorSplitter(shape)
                # Find the default optimizer for this parameter
                default_opt = self._find_default_optimizer(tree_path)
                if default_opt:
                    node.tensor_splitter.add_region(
                        tuple(slice(None) for _ in shape), 
                        default_opt
                    )
            
            # Convert tensor parts to proper slices
            normalized_slices = self._normalize_tensor_slices(tensor_parts, shape)
            # Validate that the slices are within bounds
            for i, (s, dim_size) in enumerate(zip(normalized_slices, shape)):
                if isinstance(s, slice):
                    if s.start is not None and s.start >= dim_size:
                        raise ValueError(f"Slice start {s.start} out of bounds for dimension {i} with size {dim_size}")
                    if s.stop is not None and s.stop > dim_size:
                        raise ValueError(f"Slice stop {s.stop} out of bounds for dimension {i} with size {dim_size}")
                elif isinstance(s, list):
                    for idx in s:
                        if idx >= dim_size or idx < -dim_size:
                            raise ValueError(f"Index {idx} out of bounds for dimension {i} with size {dim_size}")
            
            node.tensor_splitter.add_region(normalized_slices, optimizer)
        else:
            # No tensor slicing
            node.optimizer = optimizer
    
    def _get_param_shape(self, tree_path: Tuple[str, ...]) -> Optional[Tuple[int, ...]]:
        """Get shape of parameter at tree path."""
        if self._param_shapes is None:
            self._collect_param_shapes()
        
        # Direct lookup
        if tree_path in self._param_shapes:
            return self._param_shapes[tree_path]
        
        # Wildcard matching
        for param_path, shape in self._param_shapes.items():
            if self._matches_wildcard_path(list(tree_path), param_path):
                return shape
        
        return None
    
    def _collect_param_shapes(self):
        """Collect shapes of all parameters."""
        self._param_shapes = {}
        params = nnx.state(self.model, nnx.Param)
        
        def collect(tree, path=()):
            if hasattr(tree, 'value'):
                self._param_shapes[path] = tree.value.shape
            elif hasattr(tree, 'items'):
                for key, subtree in tree.items():
                    collect(subtree, path + (key,))
        
        collect(params)
    
    def _normalize_tensor_slices(self, tensor_parts: List, shape: Tuple[int, ...]) -> Tuple[Any, ...]:
        """Normalize tensor indices to slices/lists."""
        normalized = []
        
        for i, part in enumerate(tensor_parts):
            if i >= len(shape):
                break
                
            if isinstance(part, slice):
                normalized.append(part)
            elif isinstance(part, int):
                # Convert single int to slice
                normalized.append(slice(part, part + 1))
            elif isinstance(part, list):
                # Keep as list for IndexSet
                normalized.append(part)
            elif part is None or part is ...:
                normalized.append(slice(None))
            else:
                # Try to convert to slice
                try:
                    normalized.append(slice(part))
                except:
                    normalized.append(slice(None))
        
        # Pad with full slices
        while len(normalized) < len(shape):
            normalized.append(slice(None))
        
        return tuple(normalized)
    
    def _matches_wildcard_path(self, pattern: List, param_path: Tuple) -> bool:
        """Check if parameter path matches pattern with wildcards."""
        pattern_idx = 0
        param_idx = 0
        
        while pattern_idx < len(pattern) and param_idx < len(param_path):
            if pattern[pattern_idx] == '...':
                if pattern_idx == len(pattern) - 1:
                    return True
                pattern_idx += 1
                # Find next match
                while param_idx < len(param_path):
                    if pattern[pattern_idx] == param_path[param_idx]:
                        break
                    param_idx += 1
            elif pattern[pattern_idx] == param_path[param_idx]:
                pattern_idx += 1
                param_idx += 1
            else:
                return False
        
        return pattern_idx == len(pattern) and param_idx == len(param_path)
    
    def _find_default_optimizer(self, tree_path: Tuple[str, ...]) -> Optional[GradientTransformation]:
        """Find the default optimizer for a parameter by traversing up the tree."""
        # First check if there's a direct optimizer assigned to any parent
        node = self.root
        
        # Check wildcard first
        if node.is_wildcard and node.optimizer:
            default_opt = node.optimizer
        else:
            default_opt = None
            
        # Traverse down the path to find more specific assignments
        for i, key in enumerate(tree_path):
            if key in node.children:
                node = node.children[key]
                if node.optimizer:
                    default_opt = node.optimizer
            else:
                # No more specific path, use what we have
                break
                
        return default_opt
    
    def _combine_optimizers(self, optimizers: List[GradientTransformation]) -> GradientTransformation:
        """Combine multiple optimizers so each gets original gradients."""
        if not optimizers:
            return optax.identity()
        if len(optimizers) == 1:
            return optimizers[0]
        
        def init_fn(params):
            # Initialize all optimizers
            return tuple(opt.init(params) for opt in optimizers)
        
        def update_fn(updates, state, params=None):
            # Each optimizer gets the ORIGINAL updates (gradients)
            new_updates_list = []
            new_states = []
            
            for opt, opt_state in zip(optimizers, state):
                # Each optimizer processes the original gradients
                opt_updates, new_opt_state = opt.update(updates, opt_state, params)
                new_updates_list.append(opt_updates)
                new_states.append(new_opt_state)
            
            # Sum all updates - this works because each optimizer returns zeros
            # for parameters it doesn't manage
            combined_updates = jax.tree.map(
                lambda *args: sum(args),
                *new_updates_list
            )
            
            return combined_updates, tuple(new_states)
        
        return optax.GradientTransformation(init_fn, update_fn)
    
    def create_optimizer(self, include_state_mapping: bool = True) -> GradientTransformation:
        """Create optimizer with aggregated calls per instance.
        
        Args:
            include_state_mapping: Deprecated parameter, kept for backward compatibility.
                Has no effect as map_state is no longer used.
        """
        # Collect parameters by optimizer instance
        params_by_optimizer = self._collect_params_by_optimizer()
        
        if not params_by_optimizer:
            base = optax.identity()
        else:
            # Create masked optimizer for each unique instance
            masked_opts = []
            for opt_id, (opt, param_accesses) in params_by_optimizer.items():
                masked = self._create_masked_optimizer(opt, param_accesses, include_state_mapping)
                masked_opts.append(masked)
            
            # Use a custom combining function instead of chain
            base = self._combine_optimizers(masked_opts)
        
        # Note: map_state is now integrated into _create_masked_optimizer
        return base
    
    def get_optimizer_index_map(self) -> Dict[int, int]:
        """Get mapping from optimizer instance id to state tuple index.
        
        Returns:
            Dict mapping optimizer id (from id(optimizer)) to its index in the state tuple.
        """
        params_by_optimizer = self._collect_params_by_optimizer()
        return {opt_id: i for i, opt_id in enumerate(params_by_optimizer.keys())}
    
    def get_optimizer_state(self, state, optimizer: GradientTransformation) -> Any:
        """Get the state for a specific optimizer instance.
        
        Args:
            state: The full optimizer state tuple
            optimizer: The optimizer instance you want the state for
            
        Returns:
            The state for that specific optimizer
        """
        index_map = self.get_optimizer_index_map()
        opt_id = id(optimizer)
        if opt_id not in index_map:
            raise ValueError(f"Optimizer instance not found in configuration")
        return state[index_map[opt_id]]
    
    def _collect_params_by_optimizer(self) -> Dict[int, Tuple[GradientTransformation, List[ParamAccess]]]:
        """Collect parameters grouped by optimizer instance."""
        params_by_opt = defaultdict(list)
        params = nnx.state(self.model, nnx.Param)
        
        def traverse(tree, path: Tuple[str, ...], node: PatternNode):
            if hasattr(tree, 'value'):
                # Found parameter
                if node.tensor_splitter:
                    # Has tensor slicing
                    for slices, opt in node.tensor_splitter.get_regions():
                        access = ParamAccess(path, slices)
                        params_by_opt[id(opt)].append(access)
                elif node.optimizer:
                    # Whole parameter
                    access = ParamAccess(path)
                    params_by_opt[id(node.optimizer)].append(access)
            elif hasattr(tree, 'items'):
                # Tree node
                for key, subtree in tree.items():
                    child_node = node.children.get(key)
                    
                    if child_node:
                        traverse(subtree, path + (key,), child_node)
                    elif node.is_wildcard and node.optimizer:
                        # Wildcard match
                        virtual_node = PatternNode(optimizer=node.optimizer)
                        traverse(subtree, path + (key,), virtual_node)
                    else:
                        # No child node found, check if we should inherit from parent wildcard
                        # This happens when we have a wildcard at root but specific paths override
                        # We need to find the most specific optimizer that applies
                        current = self.root
                        found_optimizer = None
                        
                        # First check if root is wildcard
                        if current.is_wildcard and current.optimizer:
                            found_optimizer = current.optimizer
                        
                        # Then traverse down to find more specific matches
                        for i, part in enumerate(path + (key,)):
                            if part in current.children:
                                current = current.children[part]
                                # Override with more specific optimizer if found
                                if current.optimizer:
                                    found_optimizer = current.optimizer
                            else:
                                # Can't go further down this path
                                # The found_optimizer (if any) is the most specific match
                                break
                        
                        if found_optimizer:
                            virtual_node = PatternNode(optimizer=found_optimizer)
                            traverse(subtree, path + (key,), virtual_node)
        
        traverse(params, (), self.root)
        
        # Convert to final format
        result = {}
        for opt_id, accesses in params_by_opt.items():
            if opt_id in self._optimizer_instances:
                result[opt_id] = (self._optimizer_instances[opt_id], accesses)
        
        return result
    
    def _extract_tree_value(self, tree, access_node, param_path, extract_fn):
        """Extract values from tree following access_node structure.
        
        Args:
            tree: The tree to extract from (params, updates, etc.)
            access_node: The access tree node describing structure
            param_path: Current path in the tree
            extract_fn: Function to extract individual values
            
        Returns:
            Extracted value(s) maintaining proper structure
        """
        result = {}
        has_any_values = False
        if '_accesses' in access_node:
            # This is a tensor with slices
            if len(access_node['_accesses']) == 1:
                # Single access - preserve original structure
                idx, slices = access_node['_accesses'][0]
                access = ParamAccess(param_path, slices)
                value = extract_fn(tree, access)
                if value is not None:
                    has_any_values = True
                    # Check if there are children - if not, return value directly
                    children_keys = [k for k in access_node if k != '_accesses']
                    if not children_keys:
                        return value
                    else:
                        # Has children, need to maintain dict structure
                        result = value
            else:
                # Multiple accesses - collect sliced values into a list
                sliced_values = []
                
                for idx, slices in access_node['_accesses']:
                    access = ParamAccess(param_path, slices)
                    value = extract_fn(tree, access)
                    if value is not None:
                        # Extract the actual array value if wrapped
                        if hasattr(value, 'value'):
                            sliced_values.append(value.value)
                        else:
                            sliced_values.append(value)
                
                if sliced_values:
                    has_any_values = True
                    # Get the original value to preserve its structure
                    orig_value = tree
                    for key in param_path:
                        if hasattr(orig_value, 'items') and key in orig_value:
                            orig_value = orig_value[key]
                        else:
                            break
                    
                    # Wrap the list in the same structure as the original
                    if hasattr(orig_value, 'replace'):
                        # Preserve the original wrapper (VariableState, Param, etc)
                        result_value = orig_value.replace(value=sliced_values)
                    else:
                        # Just return the list if no wrapper
                        result_value = sliced_values
                    
                    # Check if there are children
                    children_keys = [k for k in access_node if k != '_accesses']
                    if not children_keys:
                        return result_value
                    else:
                        # Has children, need to maintain dict structure
                        result = result_value
        
        # Process children
        for key, child in access_node.items():
            if key != '_accesses':
                child_result = self._extract_tree_value(tree, child, param_path + (key,), extract_fn)
                if child_result is not None:
                    result[key] = child_result
                    has_any_values = True
        
        # Only return a result if we found any values
        if has_any_values or result:
            return result
        else:
            return None

    def _insert_tree_updates(self, result, update_tree, access_node, param_path):
        """Insert updates back into result following access_node structure.
        
        Args:
            result: The result tree to insert into (modified in place)
            update_tree: The updates to insert
            access_node: The access tree node describing structure
            param_path: Current path in the tree
        """
        if '_accesses' in access_node:
            # This is a tensor with slices
            if len(access_node['_accesses']) == 1:
                # Single access - update_tree is the value directly
                idx, slices = access_node['_accesses'][0]
                access = ParamAccess(param_path, slices)
                # Check if this is a leaf node
                children_keys = [k for k in access_node if k != '_accesses']
                if not children_keys:
                    # Leaf node - update_tree is the value
                    result = self._insert_update(result, access, update_tree)
                else:
                    # Has children - update_tree might be a dict
                    if isinstance(update_tree, dict):
                        # Process children below
                        pass
                    else:
                        # update_tree is the value
                        result = self._insert_update(result, access, update_tree)
            else:
                # Multiple accesses - update_tree should contain a list
                if hasattr(update_tree, 'value') and isinstance(update_tree.value, list):
                    # Extract the list from wrapper
                    update_list = update_tree.value
                    for i, (idx, slices) in enumerate(access_node['_accesses']):
                        if i < len(update_list):
                            access = ParamAccess(param_path, slices)
                            # Create update with same structure as update_tree but individual value
                            if hasattr(update_tree, 'replace'):
                                individual_update = update_tree.replace(value=update_list[i])
                            else:
                                individual_update = update_list[i]
                            result = self._insert_update(result, access, individual_update)
                elif isinstance(update_tree, list):
                    # Direct list
                    for i, (idx, slices) in enumerate(access_node['_accesses']):
                        if i < len(update_tree):
                            access = ParamAccess(param_path, slices)
                            result = self._insert_update(result, access, update_tree[i])
        
        # Process children if update_tree is a dict
        if isinstance(update_tree, dict):
            for key, child in access_node.items():
                if key != '_accesses' and key in update_tree:
                    self._insert_tree_updates(result, update_tree[key], child, param_path + (key,))
        
        return result
    
    def _create_masked_optimizer(self, opt: GradientTransformation,
                                 accesses: List[ParamAccess],
                                 include_state_mapping: bool) -> GradientTransformation:
        """Create optimizer that handles specific parameter accesses."""
        # Helper function to create zeros while preserving structure
        def zeros_like_preserving_structure(x):
            if hasattr(x, 'value'):
                return x.replace(value=jnp.zeros_like(x.value))
            else:
                return jnp.zeros_like(x)
        
        # Build tree structure matching param_access
        def build_access_tree(accesses: List[ParamAccess]) -> Dict:
            """Build tree structure matching param_access paths."""
            tree = {}
            for i, access in enumerate(accesses):
                current = tree
                for key in access.tree_path:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                # Store access info at leaf
                if '_accesses' not in current:
                    current['_accesses'] = []
                current['_accesses'].append((i, access.tensor_slices))
            return tree
        
        access_tree = build_access_tree(accesses)
        
        # Create optimizer functions
        def init_fn(params):
            # Extract parameters using helper method
            extracted = self._extract_tree_value(params, access_tree, (), self._extract_param)
            return opt.init(extracted)
        
        def update_fn(updates, state, params=None):
            # Extract updates and params using helper method
            extracted_updates = self._extract_tree_value(updates, access_tree, (), self._extract_param)
            extracted_params = self._extract_tree_value(params, access_tree, (), self._extract_param) if params is not None else None
            
            # Single call to optimizer
            new_updates, new_state = opt.update(
                extracted_updates,
                state,
                extracted_params
            )
            
            # Insert updates back
            result = jax.tree.map(zeros_like_preserving_structure, updates)
            result = self._insert_tree_updates(result, new_updates, access_tree, ())
            
            return result, new_state
        
        return optax.GradientTransformation(init_fn, update_fn)
    
    
    def _extract_param(self, tree, access: ParamAccess):
        """Extract parameter value from tree."""
        # Navigate path
        value = tree
        for key in access.tree_path:
            if hasattr(value, 'items') and key in value:
                value = value[key]
            else:
                return None
        
        # Apply slicing if needed
        if access.tensor_slices:
            # For slicing, we need to slice the actual array
            # which might be inside a VariableState
            if hasattr(value, 'value'):
                # Slice the inner value
                try:
                    return value.replace(value=value.value[access.tensor_slices])
                except:
                    return None
            else:
                # Direct array slicing
                try:
                    return value[access.tensor_slices]
                except:
                    return None
        
        # Return the whole subtree as-is
        return value
    
    def _insert_update(self, tree, access: ParamAccess, update):
        """Insert update into tree."""
        if not access.tree_path:
            return update
        
        # Navigate to parent
        current = tree
        for key in access.tree_path[:-1]:
            if hasattr(current, 'items'):
                current = current[key]
            else:
                return tree
        
        # Handle final key
        final_key = access.tree_path[-1]
        if hasattr(current, 'items') and final_key in current:
            target = current[final_key]
            
            if access.tensor_slices:
                # Sliced update
                if hasattr(target, 'value'):
                    # Target has a value attribute, update the inner value
                    update_value = update.value if hasattr(update, 'value') else update
                    new_value = target.value.at[access.tensor_slices].add(update_value)
                    current[final_key] = target.replace(value=new_value)
                else:
                    # Direct array update
                    update_value = update.value if hasattr(update, 'value') else update
                    current[final_key] = target.at[access.tensor_slices].add(update_value)
            else:
                # Full update - just add the trees together
                current[final_key] = jax.tree.map(lambda t, u: t + u, target, update)
        
        return tree
    
    def print_optimizer_assignment(self):
        """Print optimizer assignments showing instance sharing."""
        params_by_opt = self._collect_params_by_optimizer()
        
        print("Optimizer assignments:")
        
        # Group by actual optimizer parameters for display
        opt_groups = defaultdict(list)
        for opt_id, (opt, accesses) in params_by_opt.items():
            # Create a key based on optimizer type and params
            opt_key = (type(opt).__name__, str(opt))
            opt_groups[opt_key].append((opt_id, accesses))
        
        for (opt_name, opt_str), instances in opt_groups.items():
            print(f"\n{opt_name}:")
            if len(instances) > 1:
                print(f"  (Shared across {len(instances)} instance(s))")
            
            all_accesses = []
            for opt_id, accesses in instances:
                all_accesses.extend(accesses)
            
            # Group by path
            by_path = defaultdict(list)
            for access in all_accesses:
                by_path[access.tree_path].append(access.tensor_slices)
            
            for path, slices_list in sorted(by_path.items()):
                path_str = "/".join(path) if path else "(root)"
                
                if not any(slices_list):
                    print(f"  - {path_str}")
                else:
                    for slices in slices_list:
                        if slices:
                            parts = []
                            for s in slices:
                                if isinstance(s, slice):
                                    start = s.start if s.start is not None else ""
                                    stop = s.stop if s.stop is not None else ""
                                    step = f":{s.step}" if s.step not in (None, 1) else ""
                                    parts.append(f"{start}:{stop}{step}")
                                elif isinstance(s, list):
                                    if len(s) <= 4:
                                        parts.append(str(s))
                                    else:
                                        parts.append(f"[{s[0]}, {s[1]}, ..., {s[-1]}]")
                                else:
                                    parts.append(str(s))
                            print(f"  - {path_str}[{', '.join(parts)}]")