import pytest
from pathlib import Path

from ..zarr2zarr import zarr2zarr, ZarrConversionConfig


def test_zarr2zarr_raises_on_bf2rawtr():
    """Test that zarr2zarr raises ValueError when given a bf2rawtr format."""
    with pytest.raises(ValueError, match="must be an OME-Zarr v0.4 or v0.5 image"):
        zarr2zarr(
            "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr",
            Path("output"),
            ZarrConversionConfig()
        )
