import time
import itertools
from datetime import timedelta
import numpy as np

import rich
import zarr
import tensorstore as ts

def normalize_array_dimensions(array, dimension_str: str) -> np.ndarray:
    """Normalize input array to 5D TCZYX format.
    
    Args:
        array: Input array-like object
        dimension_str: String indicating dimension order (e.g. 'tczyx', 'cyx')
        Must be in order matching suffix of 'tczyx'
        
    Returns:
        5D array with dimensions ordered as TCZYX, with size 1 for missing dimensions
    """
    # Convert input array to numpy if needed
    arr = np.asarray(array)
    
    # Verify dimension string matches array rank
    if len(dimension_str) != arr.ndim:
        raise ValueError(f"Dimension string {dimension_str} does not match array rank {arr.ndim}")
        
    # Check if dimensions are in correct order
    if dimension_str.lower() != 'tczyx'[-len(dimension_str):]:
        raise ValueError(
            f"Dimensions must be provided in order matching suffix of 'tczyx', "
            f"got {dimension_str}"
        )
    
    # Create shape for padding (all 1s initially)
    new_shape = [1, 1, 1, 1, 1]
    
    # Fill in the provided dimensions from the right
    new_shape[-len(dimension_str):] = arr.shape
    
    # Just reshape - no transpose needed since dimensions are in correct order
    return arr.reshape(tuple(new_shape))

def write_array_as_ome_zarr(array, dimension_str: str, output_path: str, chunks=None):
    """Write an array as OME-ZARR, normalizing dimensions to TCZYX format.
    
    Args:
        array: Input array-like object
        dimension_str: String indicating dimension order (e.g. 'tczyx', 'cyx')
        output_path: Path to write the zarr store
        chunks: Optional chunk size (defaults to [1,1,64,64,64])
    """
    # Default chunks if none provided
    if chunks is None:
        chunks = [1, 1, 64, 64, 64]
        
    # Normalize array to 5D TCZYX
    normalized_array = normalize_array_dimensions(array, dimension_str)
    
    # Write the normalized array
    write_array_to_disk_chunked(normalized_array, output_path, chunks)


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
