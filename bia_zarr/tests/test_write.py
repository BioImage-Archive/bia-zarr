import numpy as np
import pytest

from bia_zarr.write import normalize_array_dimensions

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