"""Index set implementation for intelligent tensor slicing.

This module provides an IndexSet class that represents sets of indices along a dimension
and supports set operations (union, intersection, subtraction) while maintaining the
most efficient representation (preferring slices over lists when possible).
"""

from typing import Union, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IndexSet:
    """Represents a set of indices along one dimension.
    
    Can be:
    - A slice (start:stop:step) for efficient range access
    - A list of specific indices for arbitrary access
    - A single integer
    - None (representing all indices)
    
    The class automatically converts between representations to maintain
    the most efficient form (preferring slices over lists).
    """
    
    indices: Union[slice, Tuple[int, ...], int, None] = None
    
    @staticmethod
    def from_any(indices: Union[slice, List[int], int, None, 'IndexSet']) -> 'IndexSet':
        """Create IndexSet from any index specification."""
        if isinstance(indices, IndexSet):
            return indices
        elif indices is None or indices is ...:
            return IndexSet(None)
        elif isinstance(indices, slice):
            return IndexSet(indices)
        elif isinstance(indices, int):
            return IndexSet(indices)
        elif isinstance(indices, (list, tuple)):
            # Try to convert to slice if possible
            if len(indices) == 0:
                return IndexSet(tuple())
            
            sorted_indices = sorted(set(indices))
            if len(sorted_indices) == 1:
                return IndexSet(sorted_indices[0])
            
            # Check if it's a contiguous range
            if len(sorted_indices) > 1:
                start = sorted_indices[0]
                stop = sorted_indices[-1] + 1
                if list(range(start, stop)) == sorted_indices:
                    return IndexSet(slice(start, stop))
                
                # Check if it's a strided range
                if len(sorted_indices) > 2:
                    diffs = [sorted_indices[i+1] - sorted_indices[i] for i in range(len(sorted_indices)-1)]
                    if all(d == diffs[0] for d in diffs):
                        step = diffs[0]
                        return IndexSet(slice(start, stop, step))
            
            return IndexSet(tuple(sorted_indices))
        else:
            raise ValueError(f"Cannot create IndexSet from {type(indices)}")
    
    def to_concrete_indices(self, dim_size: Optional[int] = None) -> List[int]:
        """Convert to a concrete list of indices."""
        if self.indices is None:
            if dim_size is None:
                raise ValueError("Need dim_size for unbounded IndexSet")
            return list(range(dim_size))
        elif isinstance(self.indices, int):
            return [self.indices]
        elif isinstance(self.indices, tuple):
            return list(self.indices)
        elif isinstance(self.indices, slice):
            if self.indices.stop is None:
                if dim_size is None:
                    raise ValueError("Need dim_size for unbounded slice")
                stop = dim_size
            else:
                stop = self.indices.stop
            
            start = self.indices.start or 0
            step = self.indices.step or 1
            
            # Handle negative indices
            if start < 0 and dim_size is not None:
                start = dim_size + start
            if stop < 0 and dim_size is not None:
                stop = dim_size + stop
                
            return list(range(start, stop, step))
        else:
            raise ValueError(f"Unknown indices type: {type(self.indices)}")
    
    def intersect(self, other: 'IndexSet', dim_size: Optional[int] = None) -> 'IndexSet':
        """Compute intersection of two index sets."""
        # Handle None (all indices)
        if self.indices is None:
            return other
        if other.indices is None:
            return self
        
        # For bounded sets, convert to concrete indices
        self_indices = set(self.to_concrete_indices(dim_size))
        other_indices = set(other.to_concrete_indices(dim_size))
        
        intersection = sorted(self_indices & other_indices)
        
        if not intersection:
            return IndexSet(tuple())
        
        return IndexSet.from_any(intersection)
    
    def subtract(self, other: 'IndexSet', dim_size: Optional[int] = None) -> List['IndexSet']:
        """Subtract another index set, returning a list of disjoint pieces.
        
        This method tries to return the most efficient representation,
        preferring slices over lists of indices.
        """
        # Handle special cases
        if other.indices is None:
            return []  # Other covers everything
        
        if self.indices is None:
            if dim_size is None:
                raise ValueError("Need dim_size for subtraction from unbounded set")
            # Need to compute the complement of other
            other_indices = set(other.to_concrete_indices(dim_size))
            all_indices = set(range(dim_size))
            remaining = sorted(all_indices - other_indices)
            
            if not remaining:
                return []
            
            # Try to group into efficient pieces
            return self._group_indices_efficiently(remaining)
        
        # For bounded sets
        self_indices = set(self.to_concrete_indices(dim_size))
        other_indices = set(other.to_concrete_indices(dim_size))
        
        remaining = sorted(self_indices - other_indices)
        
        if not remaining:
            return []
        
        if remaining == sorted(self_indices):
            return [self]  # No overlap, return self
        
        # Group remaining indices into efficient pieces
        return self._group_indices_efficiently(remaining)
    
    def _group_indices_efficiently(self, indices: List[int]) -> List['IndexSet']:
        """Group a list of indices into efficient IndexSets (preferring slices)."""
        if not indices:
            return []
        
        result = []
        i = 0
        
        while i < len(indices):
            start_idx = i
            start_val = indices[i]
            
            # First, try to find a contiguous range (step=1)
            j = i + 1
            while j < len(indices) and indices[j] == indices[j-1] + 1:
                j += 1
            
            if j - i >= 2:  # At least 2 elements in contiguous range
                result.append(IndexSet(slice(start_val, indices[j-1] + 1)))
                i = j
                continue
            
            # If not contiguous, try to find a strided pattern
            if i + 1 < len(indices):
                step = indices[i + 1] - indices[i]
                if step > 1:
                    j = i + 1
                    while j < len(indices) and indices[j] - indices[j-1] == step:
                        j += 1
                    
                    if j - i >= 3:  # At least 3 elements in strided range
                        result.append(IndexSet(slice(start_val, indices[j-1] + 1, step)))
                        i = j
                        continue
            
            # Can't form an efficient slice, collect individual elements
            # Try to collect a small group that doesn't form a pattern
            j = i + 1
            while j < len(indices) and j - i < 5:  # Limit small groups to 5 elements
                # Stop if we see the start of a pattern
                if j + 1 < len(indices):
                    if indices[j] + 1 == indices[j + 1]:  # Start of contiguous
                        break
                    if j > i and indices[j] - indices[j-1] == indices[j+1] - indices[j]:  # Start of strided
                        break
                j += 1
            
            if j - i == 1:
                result.append(IndexSet(indices[i]))
            else:
                result.append(IndexSet(tuple(indices[i:j])))
            
            i = j
        
        return result
    
    def union(self, other: 'IndexSet', dim_size: Optional[int] = None) -> 'IndexSet':
        """Compute union of two index sets."""
        if self.indices is None or other.indices is None:
            return IndexSet(None)
        
        self_indices = set(self.to_concrete_indices(dim_size))
        other_indices = set(other.to_concrete_indices(dim_size))
        
        union = sorted(self_indices | other_indices)
        
        return IndexSet.from_any(union)
    
    def is_empty(self) -> bool:
        """Check if the index set is empty."""
        if self.indices is None:
            return False
        elif isinstance(self.indices, tuple):
            return len(self.indices) == 0
        elif isinstance(self.indices, int):
            return False
        elif isinstance(self.indices, slice):
            start = self.indices.start or 0
            stop = self.indices.stop
            if stop is None:
                return False
            return start >= stop
        return False
    
    def __repr__(self):
        if self.indices is None:
            return "IndexSet(:)"
        elif isinstance(self.indices, int):
            return f"IndexSet({self.indices})"
        elif isinstance(self.indices, tuple):
            if len(self.indices) == 0:
                return "IndexSet([])"
            elif len(self.indices) <= 5:
                return f"IndexSet({list(self.indices)})"
            else:
                return f"IndexSet([{self.indices[0]}, {self.indices[1]}, ..., {self.indices[-1]}])"
        elif isinstance(self.indices, slice):
            parts = []
            if self.indices.start is not None:
                parts.append(str(self.indices.start))
            else:
                parts.append("")
            
            if self.indices.stop is not None:
                parts.append(str(self.indices.stop))
            else:
                parts.append("")
            
            s = ":".join(parts)
            if self.indices.step is not None and self.indices.step != 1:
                s += f":{self.indices.step}"
            
            return f"IndexSet({s})"
        else:
            return f"IndexSet({self.indices})"
    
    def to_slice_or_list(self) -> Union[slice, List[int], int]:
        """Convert to the most natural Python indexing form."""
        if self.indices is None:
            return slice(None)
        elif isinstance(self.indices, (slice, int)):
            return self.indices
        elif isinstance(self.indices, tuple):
            if len(self.indices) == 1:
                return self.indices[0]
            else:
                return list(self.indices)
        else:
            return self.indices


def test_index_set():
    """Test IndexSet functionality."""
    print("=== Testing IndexSet ===\n")
    
    # Test creation and normalization
    print("1. Creation and normalization:")
    sets = [
        IndexSet.from_any([0, 1, 2, 3]),
        IndexSet.from_any([0, 2, 4, 6]),
        IndexSet.from_any([5, 3, 1, 7]),  # Unsorted
        IndexSet.from_any(slice(10, 20)),
        IndexSet.from_any(5),
        IndexSet.from_any(None),
    ]
    
    for s in sets:
        print(f"  {s}")
    
    # Test intersections
    print("\n2. Intersections:")
    s1 = IndexSet.from_any(slice(0, 10))
    s2 = IndexSet.from_any(slice(5, 15))
    s3 = IndexSet.from_any([3, 7, 11, 15])
    
    print(f"  {s1} ∩ {s2} = {s1.intersect(s2)}")
    print(f"  {s1} ∩ {s3} = {s1.intersect(s3)}")
    print(f"  {s2} ∩ {s3} = {s2.intersect(s3)}")
    
    # Test subtraction
    print("\n3. Subtraction:")
    s1 = IndexSet.from_any(slice(0, 20))
    s2 = IndexSet.from_any(slice(5, 10))
    s3 = IndexSet.from_any([3, 7, 12, 15])
    
    print(f"  {s1} - {s2} = {s1.subtract(s2)}")
    print(f"  {s1} - {s3} = {s1.subtract(s3)}")
    
    # Complex subtraction
    s4 = IndexSet.from_any(slice(0, 100))
    s5 = IndexSet.from_any([10, 20, 30, 40, 50, 60, 70, 80, 90])
    print(f"  {s4} - {s5} = {s4.subtract(s5)[:3]}... (first 3 pieces)")
    
    # Test efficient grouping
    print("\n4. Efficient grouping:")
    indices = [0, 1, 2, 3, 10, 11, 12, 20, 22, 24, 26, 28, 30, 40, 50, 51]
    s = IndexSet.from_any(indices)
    print(f"  Input indices: {indices}")
    print(f"  Grouped as: {s}")
    
    # Now split it
    s_split = s.subtract(IndexSet.from_any([2, 11, 24, 50]))
    print(f"  After removing [2, 11, 24, 50]: {s_split}")


if __name__ == "__main__":
    test_index_set()