"""Tests for distributed utilities."""

import jax
import jax.numpy as jnp
import pytest
from unittest.mock import patch, MagicMock

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ueaj.utils import distutil


class TestThisHostHasFirst:
    """Tests for this_host_has_first function."""
    
    def test_single_device_mesh(self):
        """Test with a single device mesh."""
        devices = jax.local_devices()[:1]
        mesh = jax.sharding.Mesh(devices, ['data'])
        
        # Should return True since there's only one device
        assert distutil.this_host_has_first(mesh, 'data') == True
    
    def test_invalid_axis_name(self):
        """Test with invalid axis name."""
        devices = jax.local_devices()[:1]
        mesh = jax.sharding.Mesh(devices, ['data'])
        
        with pytest.raises(ValueError, match="Axis 'invalid' not found"):
            distutil.this_host_has_first(mesh, 'invalid')
    
    @patch('jax.local_devices')
    @patch('jax.sharding.device_to_index')
    def test_multi_device_mesh_first_position(self, mock_device_to_index, mock_local_devices):
        """Test with multi-device mesh where local device has first position."""
        # Mock devices
        mock_device_0 = MagicMock()
        mock_device_1 = MagicMock()
        mock_local_devices.return_value = [mock_device_0]
        
        # Create a mock mesh
        mock_devices_array = MagicMock()
        mock_devices_array.flat = [mock_device_0, mock_device_1]
        
        mesh = MagicMock()
        mesh.devices = mock_devices_array
        mesh.axis_names = ['data', 'model']
        
        # Mock device_to_index to return position (0, 0) for our local device
        mock_device_to_index.return_value = (0, 0)
        
        # Should return True since local device is at position 0 along 'data' axis
        result = distutil.this_host_has_first(mesh, 'data')
        assert result == True
    
    @patch('jax.local_devices')
    @patch('jax.sharding.device_to_index')
    def test_multi_device_mesh_not_first_position(self, mock_device_to_index, mock_local_devices):
        """Test with multi-device mesh where local device is not at first position."""
        # Mock devices
        mock_device_0 = MagicMock()
        mock_device_1 = MagicMock()
        mock_local_devices.return_value = [mock_device_1]
        
        # Create a mock mesh
        mock_devices_array = MagicMock()
        mock_devices_array.flat = [mock_device_0, mock_device_1]
        
        mesh = MagicMock()
        mesh.devices = mock_devices_array
        mesh.axis_names = ['data', 'model']
        
        # Mock device_to_index to return position (1, 0) for our local device
        mock_device_to_index.return_value = (1, 0)
        
        # Should return False since local device is at position 1 along 'data' axis
        result = distutil.this_host_has_first(mesh, 'data')
        assert result == False


class TestMeshSlice:
    """Tests for MeshSlice functionality."""
    
    def test_slice_creation(self):
        """Test creating a MeshSlice object."""
        devices = jax.local_devices()[:1] 
        mesh = jax.sharding.Mesh(devices, ['data'])
        
        mesh_slice = distutil.slice(mesh)
        assert isinstance(mesh_slice, distutil.MeshSlice)
        assert mesh_slice.mesh == mesh
    
    def test_single_axis_slice(self):
        """Test slicing a single axis mesh."""
        # Skip if we don't have enough devices
        if len(jax.local_devices()) < 2:
            pytest.skip("Not enough devices for multi-device test")
        
        devices = jnp.array(jax.local_devices()[:2]).reshape(2, 1)
        mesh = jax.sharding.Mesh(devices, ['data', 'model'])
        
        # Slice to get first device along data axis
        sliced_mesh = distutil.slice(mesh)[:1, :]
        
        assert sliced_mesh.shape == (1, 1)
        assert sliced_mesh.axis_names == ['data', 'model']
    
    def test_multiple_axis_slice(self):
        """Test slicing multiple axes."""
        # Skip if we don't have enough devices
        if len(jax.local_devices()) < 4:
            pytest.skip("Not enough devices for multi-device test")
        
        devices = jnp.array(jax.local_devices()[:4]).reshape(2, 2)
        mesh = jax.sharding.Mesh(devices, ['data', 'model'])
        
        # Slice to get first device along both axes
        sliced_mesh = distutil.slice(mesh)[:1, :1]
        
        assert sliced_mesh.shape == (1, 1)
        assert sliced_mesh.axis_names == ['data', 'model']
    
    def test_slice_with_integer_index(self):
        """Test slicing with integer indices (removes dimension)."""
        # Skip if we don't have enough devices
        if len(jax.local_devices()) < 2:
            pytest.skip("Not enough devices for multi-device test")
        
        devices = jnp.array(jax.local_devices()[:2]).reshape(2, 1)
        mesh = jax.sharding.Mesh(devices, ['data', 'model'])
        
        # Slice to get single device (removes data dimension)
        sliced_mesh = distutil.slice(mesh)[0, :]
        
        assert sliced_mesh.shape == (1,)
        assert sliced_mesh.axis_names == ['model']
    
    def test_too_many_slice_dimensions(self):
        """Test error when providing too many slice dimensions."""
        devices = jax.local_devices()[:1]
        mesh = jax.sharding.Mesh(devices, ['data'])
        
        with pytest.raises(ValueError, match="Too many slice dimensions"):
            distutil.slice(mesh)[:1, :1, :1]  # 3 dims for 1D mesh
    
    def test_invalid_slice_type(self):
        """Test error with invalid slice type."""
        devices = jax.local_devices()[:1]
        mesh = jax.sharding.Mesh(devices, ['data'])
        
        # This should work fine during creation, error during actual slicing
        mesh_slice = distutil.slice(mesh)
        
        with pytest.raises(ValueError, match="Unsupported slice type"):
            mesh_slice[1.5]  # Float index should fail


def test_integration():
    """Integration test using both functions together."""
    devices = jax.local_devices()[:1]
    mesh = jax.sharding.Mesh(devices, ['data'])
    
    # Check if this host has first
    has_first = distutil.this_host_has_first(mesh, 'data')
    
    if has_first:
        # If we have first, slicing should work
        first_rank_mesh = distutil.slice(mesh)[:1]
        assert first_rank_mesh.shape == (1,)
        assert first_rank_mesh.axis_names == ['data']


if __name__ == "__main__":
    # Basic smoke test
    devices = jax.local_devices()[:1]
    mesh = jax.sharding.Mesh(devices, ['data'])
    
    print(f"Testing with {len(devices)} device(s)")
    print(f"this_host_has_first(mesh, 'data'): {distutil.this_host_has_first(mesh, 'data')}")
    
    sliced_mesh = distutil.slice(mesh)[:1]
    print(f"Sliced mesh shape: {sliced_mesh.shape}")
    print(f"Sliced mesh axis names: {sliced_mesh.axis_names}")
    
    print("Basic tests passed!")