import numpy as np
import pytest
import zarr

from bia_zarr.write import normalize_array_dimensions, write_array_as_ome_zarr

def test_normalize_array_dimensions_yx():
    """Test basic 2D array with YX dimensions"""
    arr = np.zeros((10, 5))
    normalized = normalize_array_dimensions(arr, 'yx')
    assert normalized.shape == (1, 1, 1, 10, 5)

def test_normalize_array_dimensions_cyx():
    """Test 3D array with CYX dimensions"""
    arr = np.zeros((3, 10, 5))
    normalized = normalize_array_dimensions(arr, 'cyx')
    assert normalized.shape == (1, 3, 1, 10, 5)

def test_normalize_array_dimensions_zyx():
    """Test 3D array with ZYX dimensions"""
    arr = np.zeros((4, 10, 5))
    normalized = normalize_array_dimensions(arr, 'zyx')
    assert normalized.shape == (1, 1, 4, 10, 5)

def test_normalize_array_dimensions_czyx():
    """Test 4D array with CZYX dimensions"""
    arr = np.zeros((3, 4, 10, 5))
    normalized = normalize_array_dimensions(arr, 'czyx')
    assert normalized.shape == (1, 3, 4, 10, 5)

def test_normalize_array_dimensions_tczyx():
    """Test full 5D array with TCZYX dimensions"""
    arr = np.zeros((2, 3, 4, 10, 5))
    normalized = normalize_array_dimensions(arr, 'tczyx')
    assert normalized.shape == (2, 3, 4, 10, 5)

def test_normalize_array_dimensions_wrong_rank():
    """Test error when dimension string doesn't match array rank"""
    arr = np.zeros((10, 5))
    with pytest.raises(ValueError, match="does not match array rank"):
        normalize_array_dimensions(arr, 'czyx')

def test_normalize_array_dimensions_wrong_spatial_order():
    """Test error when spatial dimensions are in wrong order"""
    arr = np.zeros((5, 10))
    with pytest.raises(ValueError, match="Spatial dimensions must be in order"):
        normalize_array_dimensions(arr, 'xy')

def test_normalize_array_dimensions_wrong_optional_order():
    """Test error when optional dimensions are in wrong order"""
    arr = np.zeros((4, 3, 10, 5))
    with pytest.raises(ValueError, match="Non-spatial dimensions must be in order"):
        normalize_array_dimensions(arr, 'zcyx')


def test_derive_n_levels():
    """Test calculation of pyramid levels"""
    from bia_zarr.write import derive_n_levels, write_array_as_ome_zarr
    
    # Test cases with expected number of levels (including base level)
    test_cases = [
        ((1, 1, 1, 256, 256), 1),  # Already at target size
        ((1, 1, 1, 512, 512), 2),  # One downsample needed
        ((1, 1, 1, 1024, 1024), 3),  # Two downsamples needed
        ((1, 1, 1, 100, 100), 1),  # Below target size
        ((1, 1, 1, 257, 256), 2),  # Just over target size
        ((1, 1, 1, 256, 512), 2),  # One dimension at target, other larger
    ]
    
    for shape, expected_levels in test_cases:
        assert derive_n_levels(shape) == expected_levels


def test_write_array_as_ome_zarr(tmp_path):
    """Test writing array as OME-ZARR with multiple pyramid levels"""
    # Create test array that will need 3 pyramid levels (1024 -> 512 -> 256)
    arr = np.zeros((2, 3, 4, 1024, 1024))
    output_path = tmp_path / "test.zarr"
    
    # Write array
    write_array_as_ome_zarr(arr, 'tczyx', str(output_path))
    
    # Check that all three levels exist
    assert (output_path / "0").exists()
    assert (output_path / "1").exists()
    assert (output_path / "2").exists()
    
    # Check that level 0 has original dimensions
    zarr_array = zarr.open(str(output_path / "0"))
    assert zarr_array.shape == (2, 3, 4, 1024, 1024)
