from typing import List, Optional, Any
from pydantic import BaseModel, Field
import dask.array as da
import zarr

from .omezarrmeta import OMEZarrMeta


# FIXME? - should we allow no unit? Propagate unknowns?
UNIT_LOOKUP = {
    None: 1,
    "meter": 1,
    "micrometer": 1e-6,
    "nanometer": 1e-9,
    "angstrom": 1e-10,
    "femtometer": 1e-15
}

AXIS_NAME_LOOKUP = {
    "x": "PhysicalSizeX",
    "y": "PhysicalSizeY",
    "z": "PhysicalSizeZ"
}


# NOTE - replace with the ome zarr models version once v3 support exists
def open_ome_zarr(zarr_group: zarr.Group) -> OMEZarrMeta:
    attrs_dict = zarr_group.attrs.asdict()

    if "ome" in attrs_dict:
        ome_metadata_dict = attrs_dict["ome"]
    elif "multiscales" in attrs_dict:
        ome_metadata_dict = attrs_dict
    else:
        raise Exception("Error parsing zarr group attributes")
    
    image_metadata = OMEZarrMeta.model_validate(ome_metadata_dict)

    return image_metadata


class OMEZarrImage(BaseModel):

    sizeX: int
    sizeY: int
    sizeZ: int = 1
    sizeC: int = 1
    sizeT: int = 1

    # Valid values for dimensions are {'tczyx', 'zyx', 'tcyx', 'czyx'}
    dimensions: str = "tczyx"

    n_scales: int = 1
    xy_scaling: float = 1.0
    chunk_scheme: List[int] = []
    shard_scheme: List[int] = []
    z_scaling: float = 1.0
    path_keys: List[str]= []
        
    PhysicalSizeX: Optional[float] = None
    PhysicalSizeY: Optional[float] = None
    PhysicalSizeZ: Optional[float] = None

    ngff_metadata: OMEZarrMeta = Field(repr=False)
    zarr_group: Any = None


def sizes_from_array_shape_and_dimension_str(array_shape: tuple, dimension_str: str) -> dict:
    """Convert array shape and dimension string into a dictionary of sizes.
    
    Args:
        array_shape (tuple): Shape of the array
        dimension_str (str): String indicating dimension order (must be one of: 
                           'tczyx', 'zyx', 'tcyx', 'czyx')
        
    Returns:
        dict: Dictionary with keys 'sizeX', 'sizeY', 'sizeZ', 'sizeT', 'sizeC'
        
    Raises:
        ValueError: If array_shape length doesn't match dimension_str length
        ValueError: If dimension_str is not one of the allowed values
    """
    valid_dims = {'tczyx', 'zyx', 'tcyx', 'czyx'}
    if dimension_str not in valid_dims:
        raise ValueError(f"Dimension string '{dimension_str}' must be one of: {', '.join(sorted(valid_dims))}")
        
    if len(array_shape) != len(dimension_str):
        raise ValueError(f"Shape {array_shape} and dimension string '{dimension_str}' must have same length")
        
    # Initialize all sizes to 1
    sizes = {
        'sizeT': 1,
        'sizeC': 1,
        'sizeZ': 1,
        'sizeX': 1,
        'sizeY': 1
    }
    
    # Create mapping from dimension character to size
    dim_to_size = dict(zip(dimension_str, array_shape))
    
    # Update sizes based on dimension string
    dim_to_key = {'t': 'sizeT', 'c': 'sizeC', 'z': 'sizeZ', 'y': 'sizeY', 'x': 'sizeX'}
    
    for dim in dim_to_key:
        if dim in dim_to_size:
            sizes[dim_to_key[dim]] = dim_to_size[dim]
    
    return sizes


def ome_zarr_image_from_zarr_group_and_metadata(
        zarr_group: zarr.Group,
        ome_zarr_metadata: OMEZarrMeta,
        ignore_unit_errors=False
):
    """Generate a OME Zarr image object by reading an OME Zarr and
    parsing the NGFF metadata for properties."""
    
    assert len(ome_zarr_metadata.multiscales) == 1
    
    multiscale = ome_zarr_metadata.multiscales[0]

    dimension_str = ''.join(a.name for a in multiscale.axes).lower() # type: ignore
    base_path_key = multiscale.datasets[0].path
    zarray = zarr_group[base_path_key]

    init_dict = sizes_from_array_shape_and_dimension_str(zarray.shape, dimension_str) # type: ignore
    init_dict['path_keys'] = [ds.path for ds in multiscale.datasets]
    init_dict['dimensions'] = dimension_str
    init_dict['n_scales'] = len(multiscale.datasets)

    scale_ratios = calculate_scale_ratios(multiscale, dimension_str)
    scale_dict = validate_scale_ratios_and_extract_xyz(scale_ratios)
    init_dict.update(scale_dict)

    init_dict['zarr_group'] = zarr_group
    init_dict['ngff_metadata'] = ome_zarr_metadata
    
    # Get chunk and shard information from the base array
    base_array = zarr_group[base_path_key]
    init_dict['chunk_scheme'] = list(base_array.chunks)
    init_dict['shard_scheme'] = list(base_array.shape) if hasattr(base_array, 'shard_shape') else []

    ome_zarr_image = OMEZarrImage(**init_dict)
    
    factors = calculate_voxel_to_physical_factors(ome_zarr_metadata, ignore_unit_errors)
    ome_zarr_image.__dict__.update(factors)
    
    return ome_zarr_image


def get_array_with_min_dimensions(ome_zarr_image: OMEZarrImage, dims: tuple):
    ydim, xdim = dims

    for path_key in reversed(ome_zarr_image.path_keys):
        zarr_array = ome_zarr_image.zarr_group[path_key]
        size_y = zarr_array.shape[-2]
        size_x = zarr_array.shape[-1]

        if (size_y >= ydim) and (size_x >= xdim):
            break
    
    return da.from_zarr(zarr_array)


def open_ome_zarr_image(ome_zarr_image_uri: str):
    import fsspec
    
    # Use fsspec to handle the URI and ensure proper cleanup
    fs = fsspec.filesystem('file')
    mapper = fs.get_mapper(ome_zarr_image_uri)
    
    # Create a DirectoryStore from the path
    store = zarr.DirectoryStore(ome_zarr_image_uri)
    zarr_group = zarr.open_group(store=store)

    ome_zarr_metadata = open_ome_zarr(zarr_group)

    ome_zarr_image = ome_zarr_image_from_zarr_group_and_metadata(zarr_group, ome_zarr_metadata)

    return ome_zarr_image


def calculate_scale_ratios(multiscale_img, dimension_str: str) -> dict[str, list[float]]:
    """Calculate the ratios between consecutive level scales in a MultiScaleImage.
    
    Args:
        multiscale_img: MultiScaleImage object containing datasets with scale transformations
        dimension_str: String indicating dimension order (e.g. 'tcyx')
        
    Returns:
        dict[str, list[float]]: Dictionary mapping dimension labels to lists of scale ratios
                               between consecutive levels
    
    Example:
        For a 4-level pyramid with 2x downscaling in Y,X at each level:
        {
            't': [1.0, 1.0, 1.0],
            'c': [1.0, 1.0, 1.0],
            'y': [2.0, 2.0, 2.0],
            'x': [2.0, 2.0, 2.0]
        }
    """
    # Get scales from each level's coordinate transformations
    scales = [dataset.coordinateTransformations[0].scale for dataset in multiscale_img.datasets]
    
    # Number of dimensions
    n_dims = len(dimension_str)
    
    # Calculate ratios between consecutive levels
    ratios = []
    for i in range(len(scales) - 1):
        current_scale = scales[i]
        next_scale = scales[i + 1]
        level_ratio = [next_scale[j] / current_scale[j] for j in range(n_dims)]
        ratios.append(level_ratio)
        
    # Convert to dictionary with dimension labels as keys
    return {dim: [ratio[i] for ratio in ratios] 
            for i, dim in enumerate(dimension_str)}


def validate_scale_ratios_and_extract_xyz(scale_ratios: dict[str, list[float]]) -> dict[str, float]:
    """Validate scale ratios from a multiscale image and return the scaling factors.
    
    Args:
        scale_ratios: Dictionary mapping dimension labels to lists of scale ratios
        
    Returns:
        dict[str, float]: Dictionary with 'xy_scaling' and 'z_scaling' values
        
    Raises:
        ValueError: If any of the following conditions are not met:
            - Scale ratios vary within an axis
            - Non-spatial dimensions (t, c) have scaling != 1.0
            - X and Y scaling ratios are not equal
            - Any scaling ratios are not positive numbers
    """
    # Check that t and c don't have scaling
    for dim in ['t', 'c']:
        if dim in scale_ratios:
            ratios = scale_ratios[dim]
            if not all(abs(r - 1.0) < 1e-10 for r in ratios):
                raise ValueError(f"Dimension {dim} must not have scaling (all ratios must be 1.0)")
    
    # Check that each axis has consistent ratios
    for dim, ratios in scale_ratios.items():
        if len(ratios) > 0:  # Only check if we have ratios
            first_ratio = ratios[0]
            if not all(abs(r - first_ratio) < 1e-10 for r in ratios):
                raise ValueError(f"Inconsistent scaling ratios for dimension {dim}: {ratios}")
            
            # Check for positive numbers
            if first_ratio <= 0:
                raise ValueError(f"Scale ratio must be positive for dimension {dim}")
    
    # Check that x and y scaling are the same
    xy_scaling = 1.0
    if 'x' in scale_ratios and 'y' in scale_ratios:
        x_ratios = scale_ratios['x']
        y_ratios = scale_ratios['y']
        if len(x_ratios) != len(y_ratios):
            raise ValueError("X and Y dimensions must have the same number of scaling ratios")
        if not all(abs(x - y) < 1e-10 for x, y in zip(x_ratios, y_ratios)):
            raise ValueError(f"X and Y scaling ratios must be equal: x={x_ratios}, y={y_ratios}")
        xy_scaling = x_ratios[0] if x_ratios else 1.0
    
    # Get z scaling (default to 1.0 if not present)
    z_scaling = scale_ratios.get('z', [1.0])[0] if 'z' in scale_ratios else 1.0
    
    return {
        'xy_scaling': xy_scaling,
        'z_scaling': z_scaling
    }     


def calculate_voxel_to_physical_factors(ngff_metadata, ignore_unit_errors=False):
    """Given ngff_metadata, calculate the voxel to physical space scale factors
    in m for each spatial dimension.
    
    NOTE: Makes a lot of assumptions about ordering of multiscales, datasets and transforms."""
    
    scale_transformations = [
        ct
        for ct in ngff_metadata.multiscales[0].datasets[0].coordinateTransformations
        if ct.type == 'scale'
    ]

    factors = {}
    
    for scale, axis in zip(scale_transformations[0].scale, ngff_metadata.multiscales[0].axes):
        if axis.type == 'space':
            attribute_name = AXIS_NAME_LOOKUP[axis.name]
            unit_multiplier = UNIT_LOOKUP.get(axis.unit, None)
            if unit_multiplier is not None:
                attribute_value = scale * unit_multiplier
                factors[attribute_name] = attribute_value
            else:
                if not ignore_unit_errors:
                    raise Exception(f"Don't know unit {axis.unit}")
            
    return factors


def reshape_to_5D(arr, dimension_str: str):
    """Reshape array to 5D (t, c, z, y, x) by adding missing dimensions as needed.
    
    Args:
        arr: dask array to reshape
        dimension_str (str): String indicating dimension order (e.g. 'tcyx', 'zyx')
        
    Returns:
        dask.array: 5D array with shape (t, c, z, y, x)
        
    Example:
        If input is (1024, 1024) with dimension_str 'yx',
        output will be (1, 1, 1, 1024, 1024)
    """
    # Validate dimension string
    valid_dims = {'tczyx', 'zyx', 'tcyx', 'czyx'}
    if dimension_str not in valid_dims:
        raise ValueError(f"Dimension string '{dimension_str}' must be one of: {', '.join(sorted(valid_dims))}")
        
    # Create mapping of dimensions to their current positions
    dim_to_pos = {dim: i for i, dim in enumerate(dimension_str)}
    
    # Define target dimensions and their default sizes
    target_dims = 'tczyx'
    new_shape = [1, 1, 1, 1, 1]  # Default shape if dimension not present
    
    # Fill in actual dimensions from input array
    for i, dim in enumerate(target_dims):
        if dim in dim_to_pos:
            new_shape[i] = arr.shape[dim_to_pos[dim]]
            
    # Reshape array to 5D
    reshaped = arr.reshape(new_shape)
    
    return reshaped
