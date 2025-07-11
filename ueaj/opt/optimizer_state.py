"""Optimizer state mapping utilities."""

import optax
from flax import nnx
from optax import GradientTransformation
from ueaj.model import ueajsum
from typing import Dict, Tuple, Any, List, Optional
from collections import defaultdict
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from .index_set import IndexSet


def map_state(model: nnx.Module, from_: bool = False) -> GradientTransformation:
	"""Transform state to/from optimizer format for Ueajsum modules.
	
	This creates an optax transformation that calls map_state on each Ueajsum
	module in the model to transform parameters between normal and optimizer formats.
	"""
	def map_state_fn(state: nnx.State):
		# Create a deep copy of the state to avoid modifying the original
		import copy
		result = copy.deepcopy(state)
		
		# Each Ueajsum module knows how to transform its own parameters
		for path, module in model.iter_modules():
			if isinstance(module, ueajsum.Ueajsum):
				# Extract the substate for this module
				current = result
				for key in path[:-1]:
					if key in current:
						current = current[key]
					else:
						break
				else:
					# We found the parent, now get the module's state
					module_key = path[-1] if path else None
					if module_key and module_key in current:
						module_state = current[module_key]
						
						# Transform the module's parameters
						transformed = module.map_state(module_state, from_optimizer=from_)
						
						# Update in place
						current[module_key] = transformed
		
		return result

	return optax.stateless(lambda updates, _: map_state_fn(updates))


@dataclass
class TensorRegion:
	"""Represents a region of a tensor with an assigned optimizer."""
	index_sets: List[IndexSet]  # One per dimension
	optimizer: GradientTransformation
	
	def to_slices(self) -> Tuple[Any, ...]:
		"""Convert to tuple of slices/indices for array indexing."""
		result = []
		for idx_set in self.index_sets:
			val = idx_set.to_slice_or_list()
			result.append(val)
		return tuple(result)
	
	def intersects(self, other: 'TensorRegion', shape: Optional[Tuple[int, ...]] = None) -> bool:
		"""Check if two regions intersect."""
		for i, (s1, s2) in enumerate(zip(self.index_sets, other.index_sets)):
			dim_size = shape[i] if shape and i < len(shape) else None
			if s1.intersect(s2, dim_size).is_empty():
				return False
		return True


@dataclass
class TensorSplitter:
	"""Manages non-overlapping regions for a tensor."""
	shape: Tuple[int, ...]
	regions: List[TensorRegion] = field(default_factory=list)
	
	def add_region(self, slices: Tuple[Any, ...], optimizer: GradientTransformation):
		"""Add a new region, splitting existing ones as needed."""
		# Convert slices to IndexSets
		index_sets = []
		for i, s in enumerate(slices):
			if i < len(self.shape):
				idx_set = IndexSet.from_any(s)
				index_sets.append(idx_set)
			else:
				break
		
		# Pad with full dimension coverage if needed
		while len(index_sets) < len(self.shape):
			index_sets.append(IndexSet(None))
		
		new_region = TensorRegion(index_sets, optimizer)
		
		# Process existing regions
		updated_regions = []
		for existing in self.regions:
			if existing.intersects(new_region, self.shape):
				# Split the existing region by the new one
				split_regions = self._split_region(existing, new_region)
				updated_regions.extend(split_regions)
			else:
				# No intersection, keep as is
				updated_regions.append(existing)
		
		# Add the new region
		updated_regions.append(new_region)
		
		# Merge adjacent regions with the same optimizer
		self.regions = self._merge_adjacent_regions(updated_regions)
	
	def _split_region(self, existing: TensorRegion, new: TensorRegion) -> List[TensorRegion]:
		"""Split existing region by subtracting the new region."""
		# Check if regions intersect
		if not existing.intersects(new, self.shape):
			return [existing]
		
		# Use a more efficient splitting algorithm
		# that generates minimal non-overlapping regions
		pieces = []
		
		# First, compute the intersection
		intersection_sets = []
		for i in range(len(self.shape)):
			intersection = existing.index_sets[i].intersect(new.index_sets[i], self.shape[i])
			intersection_sets.append(intersection)
		
		# For 2D case (most common), use optimized splitting
		if len(self.shape) == 2:
			return self._split_2d_optimized(existing, new, intersection_sets)
		
		# For general case, use dimension-by-dimension approach
		# but only create regions that don't overlap with new
		for dim in range(len(self.shape)):
			existing_set = existing.index_sets[dim]
			intersection_set = intersection_sets[dim]
			
			# Get pieces for this dimension (parts not in intersection)
			dim_pieces = existing_set.subtract(intersection_set, self.shape[dim])
			
			for piece in dim_pieces:
				# Create a region with this piece in dimension 'dim'
				# and the full existing region in other dimensions
				new_index_sets = []
				for j in range(len(self.shape)):
					if j == dim:
						new_index_sets.append(piece)
					else:
						new_index_sets.append(existing.index_sets[j])
				
				pieces.append(TensorRegion(new_index_sets, existing.optimizer))
		
		return pieces
	
	def _split_2d_optimized(self, existing: TensorRegion, new: TensorRegion, 
	                        intersection_sets: List[IndexSet]) -> List[TensorRegion]:
		"""Optimized splitting for 2D tensors to minimize region count."""
		pieces = []
		
		# Get the pieces for each dimension
		dim0_pieces = existing.index_sets[0].subtract(intersection_sets[0], self.shape[0])
		dim1_pieces = existing.index_sets[1].subtract(intersection_sets[1], self.shape[1])
		
		# Create regions more intelligently
		# First, add full strips along dimension 0
		for piece0 in dim0_pieces:
			pieces.append(TensorRegion([piece0, existing.index_sets[1]], existing.optimizer))
		
		# Then add remaining strips along dimension 1 
		# (only for the intersection part of dimension 0)
		if not intersection_sets[0].is_empty():
			for piece1 in dim1_pieces:
				pieces.append(TensorRegion([intersection_sets[0], piece1], existing.optimizer))
		
		return pieces
	
	def _merge_adjacent_regions(self, regions: List[TensorRegion]) -> List[TensorRegion]:
		"""Merge adjacent regions that have the same optimizer."""
		if len(regions) <= 1:
			return regions
		
		# Group regions by optimizer
		by_optimizer = defaultdict(list)
		for region in regions:
			by_optimizer[id(region.optimizer)].append(region)
		
		merged = []
		for opt_id, opt_regions in by_optimizer.items():
			if len(opt_regions) == 1:
				merged.extend(opt_regions)
				continue
			
			# Try to merge regions with the same optimizer
			# For now, we'll just check if regions can be combined
			# This is a simplified version - a full implementation would
			# check for true adjacency
			to_merge = opt_regions[:]
			merged_any = True
			
			while merged_any and len(to_merge) > 1:
				merged_any = False
				new_to_merge = []
				used = set()
				
				for i, r1 in enumerate(to_merge):
					if i in used:
						continue
					
					# Check if we can merge with any other region
					merged_with_any = False
					for j, r2 in enumerate(to_merge[i+1:], i+1):
						if j in used:
							continue
						
						# Check if regions are adjacent (simplified check)
						if self._can_merge_regions(r1, r2):
							# Merge the regions
							merged_region = self._merge_two_regions(r1, r2)
							new_to_merge.append(merged_region)
							used.add(i)
							used.add(j)
							merged_any = True
							merged_with_any = True
							break
					
					if not merged_with_any:
						new_to_merge.append(r1)
						used.add(i)
				
				to_merge = new_to_merge
			
			merged.extend(to_merge)
		
		return merged
	
	def _can_merge_regions(self, r1: TensorRegion, r2: TensorRegion) -> bool:
		"""Check if two regions can be merged (are adjacent)."""
		if r1.optimizer is not r2.optimizer:
			return False
		
		# Check if regions differ in exactly one dimension and are adjacent
		diff_dims = []
		for i in range(len(self.shape)):
			s1 = r1.index_sets[i]
			s2 = r2.index_sets[i]
			
			# Convert to concrete indices to compare
			indices1 = set(s1.to_concrete_indices(self.shape[i]))
			indices2 = set(s2.to_concrete_indices(self.shape[i]))
			
			if indices1 != indices2:
				diff_dims.append((i, s1, s2, indices1, indices2))
		
		# Regions should differ in exactly one dimension
		if len(diff_dims) != 1:
			return False
		
		# Check if they're adjacent in that dimension
		dim, s1, s2, indices1, indices2 = diff_dims[0]
		
		# Check if the union would form a contiguous range
		union_indices = sorted(indices1 | indices2)
		if not union_indices:
			return False
		
		# Check if union forms a contiguous range (possibly with step)
		if len(union_indices) >= 2:
			# Check for contiguous (step=1)
			if union_indices == list(range(union_indices[0], union_indices[-1] + 1)):
				return True
			
			# Check for regular step
			if len(union_indices) >= 3:
				step = union_indices[1] - union_indices[0]
				if step > 1:
					expected = list(range(union_indices[0], union_indices[-1] + 1, step))
					if union_indices == expected:
						return True
		
		return False
	
	def _merge_two_regions(self, r1: TensorRegion, r2: TensorRegion) -> TensorRegion:
		"""Merge two adjacent regions."""
		new_index_sets = []
		
		for i in range(len(self.shape)):
			s1 = r1.index_sets[i]
			s2 = r2.index_sets[i]
			
			# If they're the same, use it
			indices1 = set(s1.to_concrete_indices(self.shape[i]))
			indices2 = set(s2.to_concrete_indices(self.shape[i]))
			
			if indices1 == indices2:
				new_index_sets.append(s1)
			else:
				# Merge the two sets
				merged = s1.union(s2, self.shape[i])
				new_index_sets.append(merged)
		
		return TensorRegion(new_index_sets, r1.optimizer)
	
	def get_regions(self) -> List[Tuple[Tuple[Any, ...], GradientTransformation]]:
		"""Get all regions as (slices, optimizer) tuples."""
		return [(r.to_slices(), r.optimizer) for r in self.regions]