import time
from datetime import timedelta
from pathlib import Path
import itertools
from pathlib import Path
from urllib.parse import urlparse

import rich
import tensorstore as ts
import zarr
from typing import List, Optional
from pydantic import BaseModel, Field

from .proxyimage import open_ome_zarr_image, open_ome_zarr, OMEZarrImage
from .omezarrtypes import get_ome_zarr_type, OMEZarrType
from .write import write_array_as_ome_zarr, ZarrWriteConfig, derive_n_levels


def ensure_uri(uri: str) -> str:
    """Ensure a URI is properly formatted for tensorstore.
    
    Args:
        uri: Input URI string
        
    Returns:
        Properly formatted URI string
    """
    parsed = urlparse(uri)
    if parsed.scheme:
        return uri
    return f"file://{uri}"


def open_zarr_array_with_ts(input_array_uri: str):
    """Open a Zarr array with tensorstore.
    
    Args:
        input_array_uri: URI to the Zarr array
        
    Returns:
        tensorstore.TensorStore: Opened array
    """
    input_array_uri = ensure_uri(input_array_uri)
    
    source = ts.open({ # type: ignore
        'driver': 'zarr',
        'kvstore': input_array_uri,
    }).result()
    
    return source


def generate_write_config_from_input_image(input_image: OMEZarrImage) -> ZarrWriteConfig:
    """Generate ZarrWriteConfig from an input OME-Zarr image.
    
    Args:
        input_image: OMEZarrImage object containing metadata about the input
        
    Returns:
        ZarrWriteConfig configured based on input image properties
    """
    # Set chunks based on whether we have a z dimension
    if input_image.sizeZ > 1:
        target_chunks = [1, 1, 64, 64, 64]
        # TODO - this calculation should probably be smarter (e.g. only downsample in z when z is bigger)
        downsample_factors = [1, 1, 2, 2, 2]
    else:
        target_chunks = [1, 1, 1, 1024, 1024]
        downsample_factors = [1, 1, 1, 2, 2]
    
    # Get coordinate scales from the first dataset's transformations
    first_dataset = input_image.ngff_metadata.multiscales[0].datasets[0]
    scale_transform = next(ct for ct in first_dataset.coordinateTransformations if ct.type == 'scale') # type: ignore
    coordinate_scales = scale_transform.scale

    # Calculate number of pyramid levels from image shape
    shape = (input_image.sizeT, input_image.sizeC, input_image.sizeZ, 
            input_image.sizeY, input_image.sizeX)
    n_pyramid_levels = derive_n_levels(shape)

    return ZarrWriteConfig(
        target_chunks=target_chunks,
        coordinate_scales=coordinate_scales,
        downsample_factors=downsample_factors,
        n_pyramid_levels=n_pyramid_levels
    )


def zarr2zarr(
    ome_zarr_uri: str,
    output_base_dirpath: Path,
    config: Optional[ZarrWriteConfig] = None,
    show_config_only: bool = False
):
    """Convert between OME-Zarr formats with optional transformations.
    
    Args:
        ome_zarr_uri: URI to input OME-Zarr image
        output_base_dirpath: Path where output OME-Zarr will be written
        config: Configuration for the conversion process
        
    Raises:
        ValueError: If the input URI is not an OME-Zarr image
    """
    # Check the type of the input
    zarr_type = get_ome_zarr_type(ome_zarr_uri)
    if zarr_type not in (OMEZarrType.v04image, OMEZarrType.v05image):
        raise ValueError(f"Input URI must be an OME-Zarr v0.4 or v0.5 image, got {zarr_type}")

    # Open the OME-Zarr image
    ome_zarr_image = open_ome_zarr_image(ome_zarr_uri)
    if not ome_zarr_image.path_keys:
        raise ValueError("No arrays found in Zarr group")

    # Generate write config from input image if none provided
    if not config:
        config = generate_write_config_from_input_image(ome_zarr_image)

    if show_config_only:
        rich.print(config)
        return
    
    first_array_uri = f"{ome_zarr_uri}/{ome_zarr_image.path_keys[0]}"
    source_array = open_zarr_array_with_ts(first_array_uri)

    write_array_as_ome_zarr(
        array=source_array,
        dimension_str=ome_zarr_image.dimensions,
        output_path=str(output_base_dirpath),
        write_config=config,
    )
