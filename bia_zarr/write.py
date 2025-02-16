import time
import itertools
import math
from typing import List, Tuple, Optional
from datetime import timedelta
from pathlib import Path

import numpy as np
import rich
import zarr
import tensorstore as ts
from pydantic import BaseModel, Field

from .genmeta import create_ome_zarr_metadata



class ZarrWriteConfig(BaseModel):
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
    coordinate_scales: List[float] = Field(
        description="Voxel to physical space coordinate scales for pyramid base level. If unset, will be copied from input OME-Zarr"
    )
    n_pyramid_levels: Optional[int] = Field(
        default=None,
        description="Number of downsampled pyramid levels"
    )
    zarr_version: int = Field(
        default=2,
        description="Version of Zarr to use for output (2 or 3)"
    )
    shard_size: List[int] = Field(
        default=[1, 1, 128, 128, 128],
        description="Sharding size to use for Zarr v3"
    )


def derive_n_levels(shape: Tuple[int, ...]) -> int:
    """Calculate number of pyramid levels needed to downsample until largest spatial dimension <= 256.
    
    Args:
        shape: Tuple of array dimensions in TCZYX order
        
    Returns:
        Number of pyramid levels needed
    """
    # Get X and Y dimensions (last two elements of shape)
    max_spatial_dim = max(shape[-2:])  # max of Y and X dimensions
    
    # Calculate how many times we need to divide by 2 to get <= 256
    n_levels = max(0, math.ceil(math.log2(max_spatial_dim / 256)))
    
    return n_levels + 1  # Add 1 to include the base level


def normalize_np_array_dimensions(array, dimension_str: str) -> np.ndarray:
    """Normalize input array to 5D TCZYX format.
    
    Args:
        array: Input array-like object
        dimension_str: String indicating dimension order (e.g. 'tczyx', 'cyx')
        
    Returns:
        5D array with dimensions ordered as TCZYX, with size 1 for missing dimensions
    """
    # # Convert input array to numpy if needed
    # arr = np.asarray(array)
    
    # Verify dimension string matches array rank
    if len(dimension_str) != array.ndim:
        raise ValueError(f"Dimension string {dimension_str} does not match array rank {array.ndim}")
        
    # Convert to lowercase for comparison
    dims = dimension_str.lower()
    
    # Split into spatial ('yx') and optional dimensions ('tcz')
    spatial_dims = ''.join(d for d in dims if d in 'yx')
    optional_dims = ''.join(d for d in dims if d in 'tcz')
    
    # Validate spatial dimensions are in correct order
    if spatial_dims and spatial_dims != 'yx':
        raise ValueError(f"Spatial dimensions must be in order 'yx', got '{spatial_dims}'")
    
    # Validate optional dimensions are in correct order if present
    if optional_dims:
        valid_optional = 'tcz'
        optional_positions = [valid_optional.index(d) for d in optional_dims]
        if not all(a <= b for a, b in zip(optional_positions, optional_positions[1:])):
            raise ValueError(f"Non-spatial dimensions must be in order 'tcz', got '{optional_dims}'")
    
    # Map each dimension to its position in the target 5D array
    dim_to_pos = {'t': 0, 'c': 1, 'z': 2, 'y': 3, 'x': 4}
    
    # Create shape for padding (all 1s initially)
    new_shape = [1, 1, 1, 1, 1]
    
    # Map each input dimension to its correct position
    for i, dim in enumerate(dims):
        if dim not in dim_to_pos:
            raise ValueError(f"Invalid dimension '{dim}', must be one of 'tczyx'")
        pos = dim_to_pos[dim]
        new_shape[pos] = array.shape[i]
    
    # Just reshape - no transpose needed since dimensions are in correct order
    return array.reshape(tuple(new_shape))


# TODO - coordinate scales, omero block
def generate_write_config_from_array(array, coordinate_scales: Optional[List[float]] = None) -> ZarrWriteConfig:
    """Generate ZarrWriteConfig from an array and its dimension string.
    
    Args:
        array: Input array
        
    Returns:
        ZarrWriteConfig configured based on array properties
    """
    
    # Require TCZYX
    assert len(array.shape) == 5

    # Set chunks based on whether we have a z dimension with size > 1
    if array.shape[2] > 1:  # Z dimension is index 2 in TCZYX
        target_chunks = [1, 1, 64, 64, 64]
        downsample_factors = [1, 1, 2, 2, 2]
    else:
        target_chunks = [1, 1, 1, 1024, 1024]
        downsample_factors = [1, 1, 1, 2, 2]
    
    # Default coordinate scales (1.0 for each dimension)
    if not coordinate_scales:
        coordinate_scales = [1.0] * 5
    
    return ZarrWriteConfig(
        target_chunks=target_chunks,
        coordinate_scales=coordinate_scales,
        downsample_factors=downsample_factors
    )


def write_array_as_ome_zarr(
        array,
        dimension_str: str,
        output_path: str,
        write_config: Optional[ZarrWriteConfig] = None,
        channel_labels: Optional[List[str]] = None
):
    """Write an array as OME-ZARR, normalizing dimensions to TCZYX format.
    
    Args:
        array: Input array-like object
        dimension_str: String indicating dimension order (e.g. 'tczyx', 'cyx')
        output_path: Path to write the zarr store
        write_config: Optional ZarrWriteConfig object for controlling output format
        channel_labels: Optional list of strings to label channels. Must match number of channels.
    """

    # TODO - we only handle tensorstore arrays if they're already TCZYX. Fixing the dimensions for TS
    # arrays is fiddlier, but we should do it sometime
    if hasattr(array, 'read'):
        assert (len(array.shape) == 5) and (dimension_str == 'tczyx')
        normalized_array = array
    else:
        # Normalize array to 5D TCZYX
        normalized_array = normalize_np_array_dimensions(array, dimension_str)

    # Use default config if none provided
    if write_config is None:
        write_config = generate_write_config_from_array(normalized_array)
    
    # Validate channel labels if provided
    if channel_labels is not None:
        if 'c' not in dimension_str.lower():
            raise ValueError("Channel labels provided but input has no channel dimension")

    if channel_labels is not None:
        n_channels = normalized_array.shape[1]  # C is second dimension in TCZYX
        if len(channel_labels) != n_channels:
            raise ValueError(f"Number of channel labels ({len(channel_labels)}) does not match number of channels ({n_channels})")
    
    # Create zarr group at output path
    group = zarr.open_group(output_path, mode='w', zarr_format=write_config.zarr_version)

    # Calculate number of pyramid levels needed
    n_levels = write_config.n_pyramid_levels or derive_n_levels(normalized_array.shape)
    
    # Write base level (level 0)
    current_path = f"{output_path}/0"
    write_array_to_disk_chunked(normalized_array, current_path, write_config.target_chunks)
    
    # Generate subsequent pyramid levels
    for level in range(1, n_levels):
        prev_path = f"{output_path}/{level-1}"
        current_path = f"{output_path}/{level}"
        
        downsample_array_and_write_to_dirpath(
            prev_path,
            Path(current_path),
            write_config.downsample_factors,
            write_config.target_chunks
        )

    ome_zarr_metadata = create_ome_zarr_metadata(
        output_path,
        "test_image",
        write_config.coordinate_scales,
        write_config.downsample_factors,
        create_omero_block=True,
        channel_labels=channel_labels
    )

    ome_metadata_dict = ome_zarr_metadata.model_dump(exclude_unset=True)

    if write_config.zarr_version == 3:
        ome_metadata_dict.update(
            {
                "version": "0.5",
                "_creator": {
                    "name": "bia-zarr"
                }
            }
        )
        ome_metadata = {
            "ome": ome_metadata_dict
        }
        group.attrs.update(ome_metadata) # type: ignore
    elif write_config.zarr_version == 2:
        group.attrs.update(ome_zarr_metadata.model_dump(exclude_unset=True)) # type: ignore


def downsample_array_and_write_to_dirpath(
        array_uri: str,
        output_dirpath: Path,
        downsample_factors: List[int],
        output_chunks: List[int],
        downsample_method='mean'
    ):
    """
    Downsample a zarr array and save the result to a new location with specified chunking.

    This function opens a source array, downsamples it using the specified method, and writes
    the result to a new zarr array with specified chunk sizes. The dimension separator
    in the output is set to '/'.

    Args:
        array_uri: URI or path to source zarr array. If a local path is provided,
            it will be converted to a file:// URI.
        output_dirpath: Path where the downsampled array will be written.
        downsample_factors: List of integers specifying the downsample factor for each dimension.
            For example, [2, 2] will reduce the size of a 2D array by half in each dimension.
        output_chunks: List of integers specifying the chunk size for each dimension
            of the output array.
        downsample_method: string description of the downsampling method, must be one of those
            supported by tensorstore's downsampling driver

    Returns:
        None

    Example:
        >>> downsample_array_and_write_to_dirpath(
        ...     "data.zarr",
        ...     Path("downsampled.zarr"),
        ...     downsample_factors=[2, 2],
        ...     output_chunks=[256, 256]
        ... )
    """

    source = ts.open({ # type: ignore
        'driver': 'downsample',
        'downsample_factors': downsample_factors,
        "downsample_method": downsample_method,
        'base': {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': array_uri
            }
        }
    }).result()

    write_array_to_disk_chunked(source, output_dirpath, output_chunks)


def write_array_to_disk_chunked(source_array, output_dirpath, target_chunks):
    """Write the input array to the output path with the target array chunk size.
    The actual read/write from source to output is also chunked, so should handle
    large arrays without memory issues."""

    output_spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': str(output_dirpath)
        },
        'dtype': source_array.dtype.name,
        'metadata': {
            'shape': source_array.shape,
            'chunks': target_chunks,
            'dimension_separator': '/',
        },
    }

    output_array = ts.open(output_spec, create=True, delete_existing=True).result() # type: ignore

    # TODO - should probably make this configurable
    processing_chunk_size = [512, 512, 512, 512, 512]
    
    # Calculate number of chunks needed in each dimension
    num_chunks = tuple(
        (shape + chunk - 1) // chunk 
        for shape, chunk in zip(source_array.shape, processing_chunk_size)
    )

    # Process array in chunks
    idx_list = list(itertools.product(*[range(n) for n in num_chunks]))
    start_time = time.time()
    
    for n, idx in enumerate(idx_list):
        
        # Calculate slice for this chunk
        slices = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, processing_chunk_size, source_array.shape)
        )
        
        # Read the chunk
        # Check if it's a tensorstore array by seeing if it has the read attribute
        # We want to do this lazily if at all possible
        if hasattr(source_array[slices], 'read'):
            chunk_data = source_array[slices].read().result()
        else:
            # For regular numpy arrays, just use the slice directly
            chunk_data = source_array[slices]

        output_array[slices].write(chunk_data).result()
        
        # Calculate progress and timing
        elapsed_time = time.time() - start_time
        chunks_remaining = len(idx_list) - (n + 1)
        
        if n > 0:  # Only calculate average after first chunk
            avg_chunk_time = elapsed_time / (n + 1)
            eta_seconds = avg_chunk_time * chunks_remaining
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "calculating..."
            
        # Progress indication with timing
        rich.print(
            f"Processed chunk [{n + 1}/{len(idx_list)}]{idx} of {tuple(n-1 for n in num_chunks)} | "
            f"Elapsed: {str(timedelta(seconds=int(elapsed_time)))} | "
            f"ETA: {eta}"
        )


