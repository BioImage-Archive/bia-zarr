import pytest
from pathlib import Path

from ..zarr2zarr import zarr2zarr, ZarrConversionConfig, generate_write_config_from_input_image
from ..omezarrmeta import OMEZarrMeta, MultiScaleImage, DataSet, CoordinateTransformation, Axis
from ..proxyimage import OMEZarrImage


def test_zarr2zarr_raises_on_bf2rawtr():
    """Test that zarr2zarr raises ValueError when given a bf2rawtr format."""
    with pytest.raises(ValueError, match="must be an OME-Zarr v0.4 or v0.5 image"):
        zarr2zarr(
            "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr",
            Path("output"),
            ZarrConversionConfig()
        )


def create_test_ngff_metadata(dimension_str: str, scales: list, downsample_factor: int = 2):
    """Helper function to create test NGFF metadata."""
    axes = [Axis(name=dim, type='space' if dim in 'xyz' else 'time' if dim == 't' else 'channel') 
            for dim in dimension_str]
    
    datasets = []
    for level in range(len(scales)):
        scale_factors = [1 if dim in 'tc' else downsample_factor**level for dim in dimension_str]
        datasets.append(DataSet(
            path=str(level),
            coordinateTransformations=[
                CoordinateTransformation(
                    type="scale",
                    scale=[s * f for s, f in zip(scales, scale_factors)]
                )
            ]
        ))
    
    return OMEZarrMeta(multiscales=[MultiScaleImage(
        axes=axes,
        datasets=datasets,
        name="test_image"
    )])


def test_generate_write_config_2d():
    """Test generating write config from a 2D image."""
    # Create test metadata for a 2D image (tcyx)
    ngff_metadata = create_test_ngff_metadata(
        dimension_str='tcyx',
        scales=[1, 1, 0.5, 0.5],  # Initial scales
        downsample_factor=2
    )
    
    image = OMEZarrImage(
        sizeX=1024,
        sizeY=1024,
        dimensions='tcyx',
        n_scales=3,
        path_keys=['0', '1', '2'],
        ngff_metadata=ngff_metadata
    )
    
    config = generate_write_config_from_input_image(image)
    
    assert config.chunks == [1, 1, 1, 1024, 1024]
    assert config.coordinate_scales == [1, 1, 0.5, 0.5]
    assert config.downsample_factors == [2, 2]


def test_generate_write_config_3d():
    """Test generating write config from a 3D image."""
    # Create test metadata for a 3D image (tczyx)
    ngff_metadata = create_test_ngff_metadata(
        dimension_str='tczyx',
        scales=[1, 1, 2, 0.5, 0.5],  # Initial scales
        downsample_factor=2
    )
    
    image = OMEZarrImage(
        sizeX=1024,
        sizeY=1024,
        sizeZ=100,
        dimensions='tczyx',
        n_scales=3,
        path_keys=['0', '1', '2'],
        ngff_metadata=ngff_metadata
    )
    
    config = generate_write_config_from_input_image(image)
    
    assert config.chunks == [1, 1, 64, 64, 64]
    assert config.coordinate_scales == [1, 1, 2, 0.5, 0.5]
    assert config.downsample_factors == [2, 2]
