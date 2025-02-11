import pytest
from unittest.mock import MagicMock, patch

from bia_zarr.omezarrtypes import get_ome_zarr_type, OMEZarrType


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
