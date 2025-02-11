import pytest
from unittest.mock import MagicMock, patch

from bia_zarr.omezarrtypes import get_ome_zarr_type, OMEZarrType


@pytest.mark.parametrize("url,expected_type,zarr_format,attrs", [
    (
        "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr/0",
        OMEZarrType.v04image,
        2,
        {"multiscales": [{}]}  # Simplified valid v0.4 metadata
    ),
    (
        "https://uk1s3.embassy.ebi.ac.uk/ebi-ngff-challenge-2024/4ffaeed2-fa70-4907-820f-8a96ef683095.zarr",
        OMEZarrType.v05image,
        3,
        {"ome": {"multiscales": [{}]}}  # Simplified valid v0.5 metadata
    ),
    (
        "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr",
        OMEZarrType.bf2rawtr,
        2,
        {"bioformats2raw.layout": 3}
    ),
    (
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0056B/7361.zarr",
        OMEZarrType.hcs,
        2,
        {"plate": {}}
    ),
])
def test_get_ome_zarr_type(url, expected_type, zarr_format, attrs):
    mock_group = MagicMock()
    mock_group.metadata.zarr_format = zarr_format
    mock_group.attrs = attrs

    with patch('zarr.open_group', return_value=mock_group):
        assert get_ome_zarr_type(url) == expected_type


def test_get_ome_zarr_type_unknown():
    mock_group = MagicMock()
    mock_group.metadata.zarr_format = 2
    mock_group.attrs = {"unknown": "format"}

    with patch('zarr.open_group', return_value=mock_group):
        with pytest.raises(Exception, match="Unknown"):
            get_ome_zarr_type("some_url")
