"""Distributed utilities for JAX mesh operations."""

import jax
import jax.numpy as jnp
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Optional
from flax import nnx

# Store reference to built-in slice before we define our own function
_builtin_slice = slice


def this_host_has_first(mesh: jax.sharding.Mesh, axis_name: str) -> bool:
    """Check if current host has the first device along specified axis."""
    if axis_name not in mesh.axis_names:
        raise ValueError(f"Axis '{axis_name}' not found in mesh with axes {mesh.axis_names}")
    
    axis_idx = mesh.axis_names.index(axis_name)
    mesh_shape = tuple(mesh.shape.values())
    
    for device in mesh.local_devices:
        for coords in np.ndindex(mesh_shape):
            if mesh.devices[coords] == device and coords[axis_idx] == 0:
                return True
    return False


class MeshSlice:
    """Helper for NumPy-style slicing of JAX device meshes.

    Supports both positional and named (dict-based) axis slicing.

    Positional slicing:
        >>> sub_mesh = mesh_slice(mesh)[0:2, :]

    Named slicing (more explicit, order-independent):
        >>> sub_mesh = mesh_slice(mesh)[{'data': slice(0, 2), 'model': 0}]
        >>> sub_mesh = mesh_slice(mesh)[{'tensor': slice(None, None, 2)}]
    """

    def __init__(self, mesh: jax.sharding.Mesh):
        self.mesh = mesh

    def __getitem__(self, key) -> jax.sharding.Mesh:
        """Slice mesh along axes using NumPy-style indexing or dict-based indexing.

        Supports both positional indexing and named axis indexing:
        - Positional: mesh_slice(mesh)[0:2, :]
        - Named dict: mesh_slice(mesh)[{'data': slice(0, 2), 'model': slice(None)}]
        """
        # Handle dict-based indexing by axis name
        if isinstance(key, dict):
            key = tuple(key.get(axis, _builtin_slice(None)) for axis in self.mesh.axis_names)

        if not isinstance(key, tuple):
            key = (key,)

        key = key + (_builtin_slice(None),) * (len(self.mesh.axis_names) - len(key))
        
        if len(key) > len(self.mesh.axis_names):
            raise ValueError(f"Too many dimensions: {len(key)} > {len(self.mesh.axis_names)}")
        
        sliced_devices = self.mesh.devices[key]
        new_axes = []
        new_shape = []
        
        for i, (s, axis) in enumerate(zip(key, self.mesh.axis_names)):
            if isinstance(s, _builtin_slice):
                axis_size = list(self.mesh.shape.values())[i]
                size = len(range(*s.indices(axis_size)))
                if size > 0:
                    new_axes.append(axis)
                    new_shape.append(size)
        
        if sliced_devices.size > 0:
            sliced_devices = sliced_devices.reshape(new_shape)
        
        return jax.sharding.Mesh(sliced_devices, new_axes)


def slice(mesh: jax.sharding.Mesh) -> MeshSlice:
    """Create a mesh slicer for convenient axis slicing.

    This is the primary interface for slicing JAX meshes. Returns a MeshSlice
    wrapper that supports NumPy-style indexing or dict-based indexing to extract sub-meshes.

    Args:
        mesh: The JAX Mesh to slice.

    Returns:
        MeshSlice wrapper that supports `[]` indexing.

    Example:
        >>> from ueaj.utils.distutil import slice as mesh_slice
        >>> import jax
        >>>
        >>> mesh = jax.sharding.Mesh(devices.reshape(4, 2), ('data', 'model'))
        >>>
        >>> # Positional slicing
        >>> sub_mesh = mesh_slice(mesh)[0:2, :]
        >>> data_only = mesh_slice(mesh)[:, 0]
        >>>
        >>> # Named slicing (more explicit, order-independent)
        >>> sub_mesh = mesh_slice(mesh)[{'data': slice(0, 2)}]
        >>> sub_mesh = mesh_slice(mesh)[{'model': 0}]  # Collapse model axis

    Note:
        Named 'slice' to provide clean import: `from distutil import slice as mesh_slice`
        This avoids shadowing Python's built-in `slice` when renamed on import.
    """
    return MeshSlice(mesh)


def _compute_square_blocks(height: int, width: int, num_blocks: int) -> Tuple[int, int, List[Tuple]]:
    """Compute square-like block divisions."""
    if height * width < num_blocks:
        return height, width, [(0, 0, height, 0, width)]
    
    # Find best square-like division
    best_ratio = float('inf')
    best_division = None
    
    for blocks_h in range(1, min(num_blocks + 1, height + 1)):
        if num_blocks % blocks_h != 0:
            continue
        blocks_w = num_blocks // blocks_h
        if blocks_w > width:
            continue
        
        block_h = (height + blocks_h - 1) // blocks_h
        block_w = (width + blocks_w - 1) // blocks_w
        ratio = max(block_h / block_w, block_w / block_h)
        
        if ratio < best_ratio:
            best_ratio = ratio
            best_division = (block_h, block_w, blocks_h, blocks_w)
    
    if best_division is None:
        # Fallback: split along longer dimension
        if height >= width:
            blocks_h = min(num_blocks, height)
            best_division = ((height + blocks_h - 1) // blocks_h, width, blocks_h, 1)
        else:
            blocks_w = min(num_blocks, width)
            best_division = (height, (width + blocks_w - 1) // blocks_w, 1, blocks_w)
    
    block_h, block_w, blocks_h, blocks_w = best_division
    
    # Generate block coordinates
    blocks = []
    for i in range(min(num_blocks, blocks_h * blocks_w)):
        row, col = divmod(i, blocks_w) if blocks_w > 0 else (0, 0)
        blocks.append((
            i,
            row * block_h,
            min((row + 1) * block_h, height),
            col * block_w,
            min((col + 1) * block_w, width)
        ))
    
    return block_h, block_w, blocks


def block_allocations(model: nnx.Module, mesh: jax.sharding.Mesh, 
                     einsum_paths: Optional[List[Tuple[str, ...]]] = None) -> Dict[str, Any]:
    """Compute block allocations for distributed training."""
    if 'tensors' not in mesh.axis_names or 'blocks' not in mesh.axis_names:
        raise ValueError("Mesh must have 'tensors' and 'blocks' axes")
    
    block_axis_size = mesh.shape['blocks']
    tensor_axis_size = mesh.shape['tensors']
    
    # Collect einsum metadata
    einsum_metadata = {}
    def collect_metadata(obj, path=""):
        if hasattr(obj, 'get_einsum_metadata'):
            for name, meta in obj.get_einsum_metadata().items():
                einsum_metadata[f"{path}.{name}" if path else name] = meta
        for name, value in obj.__dict__.items():
            if not name.startswith('_') and isinstance(value, nnx.Module):
                collect_metadata(value, f"{path}.{name}" if path else name)
    
    collect_metadata(model)
    einsum_prefixes = ['.'.join(p) for p in einsum_paths] if einsum_paths else None
    
    # Flatten state
    state = nnx.state(model)
    flat_params = []
    
    def flatten_state(obj, path=""):
        if hasattr(obj, 'items'):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                if hasattr(value, 'value') and isinstance(value.value, jnp.ndarray):
                    flat_params.append((new_path, value.value))
                else:
                    flatten_state(value, new_path)
    
    flatten_state(state)
    
    # Allocate parameters
    einsum_allocations = []
    other_params = []
    max_block_size = 0
    tensor_id = 0
    
    for path, param_value in flat_params:
        param_path = path.rsplit('.', 1)[0] if path.endswith('.value') else path
        
        # Check if should use einsum blocks
        is_einsum = param_path in einsum_metadata
        use_blocks = is_einsum and (
            einsum_prefixes is None or 
            any(param_path.startswith(p + '.') or param_path == p for p in einsum_prefixes)
        )
        
        if use_blocks and len(param_value.shape) >= 2:
            metadata = einsum_metadata[param_path]
            shape = param_value.shape
            
            # Extract 2D layout from reduced_shape (always last 2 dims)
            # reduced_shape format: batch_shape + (reducing_size, non_reducing_size)
            reduced_shape = metadata.reduced_shape
            height, width = reduced_shape[-2], reduced_shape[-1]
            
            # Compute batch size
            # The total parameter size should equal batch_size * height * width
            total_size = np.prod(shape)
            layout_size = height * width
            batch_size = total_size // layout_size if layout_size > 0 else 1
            
            # Compute blocks
            block_h, block_w, blocks = _compute_square_blocks(height, width, block_axis_size)
            
            # Create allocations for each batch element
            for batch_idx in range(batch_size):
                einsum_allocations.append({
                    'path': path,
                    'tensor_id': tensor_id + batch_idx,
                    'blocks': blocks,
                    'shape': shape,
                    'batch_size': batch_size,
                    'batch_idx': batch_idx,
                    'block_shape': (block_h, block_w),
                    'height': height,
                    'width': width
                })
            
            tensor_id += batch_size
            max_block_size = max(max_block_size, block_h * block_w)
        else:
            other_params.append((path, param_value))
    
    # Allocate other parameters as flat tensor
    other_allocations = []
    other_metadata = None
    
    if other_params:
        total_size = sum(p.size for _, p in other_params)
        tensor_capacity = block_axis_size * max_block_size
        padded_size = ((total_size + tensor_capacity - 1) // tensor_capacity) * tensor_capacity
        num_tensors = padded_size // tensor_capacity
        
        offset = 0
        for path, param in other_params:
            other_allocations.append({
                'path': path,
                'shape': param.shape,
                'size': param.size,
                'flat_offset': offset
            })
            offset += param.size
        
        other_metadata = {
            'start_tensor_id': tensor_id,
            'num_tensors': num_tensors,
            'total_size': total_size,
            'padded_size': padded_size
        }
        tensor_id += num_tensors
    
    # Pad tensor count
    padded_tensors = ((tensor_id + tensor_axis_size - 1) // tensor_axis_size) * tensor_axis_size
    
    return {
        'einsum_allocations': einsum_allocations,
        'other_allocations': other_allocations,
        'other_metadata': other_metadata,
        'max_block_size': max_block_size,
        'num_tensors': tensor_id,
        'padded_tensors': padded_tensors,
        'tensor_array_shape': (padded_tensors, block_axis_size)
    }


def blockify(allocations: Dict[str, Any], state: nnx.State) -> jax.Array:
    """Pack parameters into blocked tensor array."""
    num_tensors, num_blocks = allocations['tensor_array_shape']
    max_block_size = allocations['max_block_size']
    
    # Initialize array
    blocked = np.zeros((num_tensors, num_blocks, max_block_size), dtype=np.float32)
    
    # Helper to get parameter from state
    def get_param(path):
        parts = path.split('.')
        current = state
        for part in parts:
            current = current[part] if hasattr(current, '__getitem__') else getattr(current, part)
        return current.value if hasattr(current, 'value') else current
    
    # Pack einsum tensors - group by path
    by_path = defaultdict(list)
    for alloc in allocations['einsum_allocations']:
        by_path[alloc['path']].append(alloc)
    
    for path, allocs in by_path.items():
        param = get_param(path)
        batch_size = allocs[0]['batch_size']
        height = allocs[0]['height']
        width = allocs[0]['width']
        
        # Reshape to 2D for block extraction
        param_2d = param.reshape(batch_size * height, width)
        
        for alloc in allocs:
            batch_idx = alloc['batch_idx']
            batch_offset = batch_idx * height
            
            for block_idx, start_h, end_h, start_w, end_w in alloc['blocks']:
                block = param_2d[batch_offset + start_h:batch_offset + end_h, start_w:end_w]
                block_flat = np.asarray(block.flatten(), dtype=np.float32)
                
                if block_flat.size < max_block_size:
                    block_flat = np.pad(block_flat, (0, max_block_size - block_flat.size))
                
                blocked[alloc['tensor_id'], block_idx] = block_flat
    
    # Pack other parameters
    if allocations['other_allocations']:
        meta = allocations['other_metadata']
        flat = np.zeros(meta['padded_size'], dtype=np.float32)
        
        for alloc in allocations['other_allocations']:
            param = get_param(alloc['path'])
            offset = alloc['flat_offset']
            flat[offset:offset + alloc['size']] = np.asarray(param.flatten(), dtype=np.float32)
        
        # Split into blocks
        start_id = meta['start_tensor_id']
        for t in range(meta['num_tensors']):
            for b in range(num_blocks):
                block_start = (t * num_blocks + b) * max_block_size
                block_end = block_start + max_block_size
                blocked[start_id + t, b] = flat[block_start:block_end]
    
    return jnp.array(blocked, dtype=jnp.bfloat16)


def deblockify(blocked_array: jax.Array, allocations: Dict[str, Any], target_state: nnx.State) -> nnx.State:
    """Unpack blocked tensor array to state structure."""
    value_updates = {}
    
    # Unpack einsum tensors - group by path
    by_path = defaultdict(list)
    for alloc in allocations['einsum_allocations']:
        by_path[alloc['path']].append(alloc)
    
    for path, allocs in by_path.items():
        # Sort by batch index
        allocs = sorted(allocs, key=lambda a: a['batch_idx'])
        
        shape = allocs[0]['shape']
        batch_size = allocs[0]['batch_size']
        height = allocs[0]['height']
        width = allocs[0]['width']
        
        if len(shape) >= 2:
            # Reconstruct 2D array
            reconstructed = jnp.zeros((batch_size * height, width), dtype=blocked_array.dtype)
            
            for alloc in allocs:
                batch_offset = alloc['batch_idx'] * height
                
                for block_idx, start_h, end_h, start_w, end_w in alloc['blocks']:
                    block_flat = blocked_array[alloc['tensor_id'], block_idx]
                    block_size = (end_h - start_h) * (end_w - start_w)
                    block = block_flat[:block_size].reshape(end_h - start_h, end_w - start_w)
                    reconstructed = reconstructed.at[batch_offset + start_h:batch_offset + end_h, start_w:end_w].set(block)
            
            value_updates[path] = reconstructed.reshape(shape)
        else:
            # 1D case
            value_updates[path] = blocked_array[allocs[0]['tensor_id'], 0, :shape[0]]
    
    # Unpack other parameters
    if allocations['other_allocations']:
        meta = allocations['other_metadata']
        num_blocks = blocked_array.shape[1]
        
        # Reconstruct flat tensor
        flat = jnp.zeros(meta['padded_size'], dtype=blocked_array.dtype)
        start_id = meta['start_tensor_id']
        
        for t in range(meta['num_tensors']):
            for b in range(num_blocks):
                block_start = (t * num_blocks + b) * allocations['max_block_size']
                block_end = block_start + allocations['max_block_size']
                flat = flat.at[block_start:block_end].set(blocked_array[start_id + t, b])
        
        # Extract individual parameters
        for alloc in allocations['other_allocations']:
            offset = alloc['flat_offset']
            size = alloc['size']
            shape = alloc['shape']
            value_updates[alloc['path']] = flat[offset:offset + size].reshape(shape)
    
    # Build nested update dict
    updates = {}
    for path, value in value_updates.items():
        parts = path.split('.')
        current = updates
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    
    # Apply updates recursively
    def apply_updates(state_node, update_node, is_root=False):
        if isinstance(state_node, nnx.State):
            result = {}
            for key in state_node:
                result[key] = apply_updates(state_node[key], update_node.get(key, {})) if key in update_node else state_node[key]
            return nnx.State(result) if is_root else result
        elif isinstance(state_node, dict):
            result = {}
            for key in state_node:
                result[key] = apply_updates(state_node[key], update_node.get(key, {})) if isinstance(update_node, dict) and key in update_node else state_node[key]
            return result
        elif hasattr(state_node, 'value') and isinstance(update_node, jnp.ndarray):
            return nnx.VariableState(type=state_node.type, value=update_node)
        else:
            return state_node
    
    return apply_updates(target_state, updates, is_root=True)


def shard(blocked_array: jax.Array, allocations: Dict[str, Any], mesh: jax.sharding.Mesh) -> jax.Array:
    """Apply sharding to blocked array."""
    spec = jax.sharding.PartitionSpec('tensors', 'blocks', None)
    sharding = jax.sharding.NamedSharding(mesh, spec)
    return jax.lax.with_sharding_constraint(blocked_array, sharding)