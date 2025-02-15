import time
from datetime import timedelta
import itertools

import rich
import tensorstore as ts
from typing import List, Optional
from pydantic import BaseModel, Field


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
    """
    # TODO: Implement conversion logic
    raise NotImplementedError("zarr2zarr conversion not yet implemented")
