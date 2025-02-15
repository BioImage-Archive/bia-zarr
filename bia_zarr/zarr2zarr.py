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

from .proxyimage import open_ome_zarr_image, open_ome_zarr
from .omezarrtypes import get_ome_zarr_type, OMEZarrType


class ZarrConversionConfig(BaseModel):
    target_chunks: List[int] = Field(
        default=[1, 1, 64, 64, 64],
        description="Array chunk layout for output zarr"
    )
    downsample_factors: List[int] = Field(
        default=[1, 1, 2, 2, 2],
        description="Factor by which each successive pyramid layer will be downsampled"
    )
    transpose_axes: List[int] = Field(
        default=[0, 1, 2, 3, 4],
        description="Order of axis transpositions to be applied during transformation."
    )
    coordinate_scales: Optional[List[float]] = Field(
        default=None,
        description="Voxel to physical space coordinate scales for pyramid base level. If unset, will be copied from input OME-Zarr"
    )
    n_pyramid_levels: Optional[int] = Field(
        default=None,
        description="Number of downsampled "
    )
    rewrite_omero_block: bool = Field(
        default=False,
        description="Rewrite the OMERO rendering block, guessing parameters. Otherwise will copy from input OME-Zarr."
    )
    zarr_version: int = Field(
        default=2,
        description="Version of Zarr to use for output (2 or 3)"
    )
    shard_size: List[int] = Field(
        default=[1, 1, 128, 128, 128],
        description="Sharding size to use for Zarr v3"
    )

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
    
    source = ts.open({
        'driver': 'zarr',
        'kvstore': input_array_uri,
    }).result()
    
    return source


def zarr2zarr(
    ome_zarr_uri: str,
    output_base_dirpath: Path,
    config: ZarrConversionConfig
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

    # Open the zarr group and get the first array
    zarr_group = zarr.open_group(ome_zarr_uri, mode='r')
    path_keys = [k for k in zarr_group.array_keys()]
    if not path_keys:
        raise ValueError("No arrays found in Zarr group")
    
    first_array_uri = f"{ome_zarr_uri}/{path_keys[0]}"
    source_array = open_zarr_array_with_ts(first_array_uri)
    
    # TODO: Implement conversion logic
    raise NotImplementedError("zarr2zarr conversion not yet implemented")
