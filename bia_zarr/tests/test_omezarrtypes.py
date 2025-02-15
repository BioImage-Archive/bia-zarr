import pytest
from unittest.mock import MagicMock, patch

from bia_zarr.omezarrtypes import get_ome_zarr_type, get_single_image_uri, OMEZarrType


@pytest.mark.parametrize("url,expected_type", [
    (
        "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr/0",
        OMEZarrType.v04image,
    ),
    (
        "https://uk1s3.embassy.ebi.ac.uk/ebi-ngff-challenge-2024/4ffaeed2-fa70-4907-820f-8a96ef683095.zarr",
        OMEZarrType.v05image,
    ),
    (
        "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr",
        OMEZarrType.bf2rawtr,
    ),
    (
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0056B/7361.zarr",
        OMEZarrType.hcs,
    ),
])
def test_get_ome_zarr_type(url, expected_type):
    """Test with real network access to example URLs from the enum docstrings"""
    assert get_ome_zarr_type(url) == expected_type


def test_get_ome_zarr_type_unknown():
    mock_group = MagicMock()
    mock_group.metadata.zarr_format = 2
    mock_group.attrs = {"unknown": "format"}

    with patch('zarr.open_group', return_value=mock_group):
        with pytest.raises(ValueError, match="Unknown OME-Zarr format"):
            get_ome_zarr_type("some_url")


@pytest.mark.parametrize("zarr_format,attrs,base_uri,expected_uri", [
    # v0.4 image - return as-is
    (2, {"multiscales": [{}]}, "http://example.com/image.zarr", "http://example.com/image.zarr"),
    # v0.5 image - return as-is
    (3, {"ome": {"multiscales": [{}]}}, "http://example.com/image.zarr", "http://example.com/image.zarr"),
    # bioformats2raw - append /0
    (2, {"bioformats2raw.layout": 3}, "http://example.com/bf2raw.zarr", "http://example.com/bf2raw.zarr/0"),
    # HCS plate - use first well path
    (2, {
        "plate": {
            "wells": [{"path": "A/1"}]
        }
    }, "http://example.com/plate.zarr", "http://example.com/plate.zarr/A/1/0"),
])
def test_get_single_image_uri(zarr_format, attrs, base_uri, expected_uri):
    mock_group = MagicMock()
    mock_group.metadata.zarr_format = zarr_format
    mock_group.attrs = attrs
    
    assert get_single_image_uri(mock_group, base_uri) == expected_uri


def test_get_single_image_uri_unknown():
    mock_group = MagicMock()
    mock_group.metadata.zarr_format = 2
    mock_group.attrs = {"unknown": "format"}

    with pytest.raises(ValueError, match="Unknown OME-Zarr format"):
        get_single_image_uri(mock_group, "http://example.com/unknown.zarr")
