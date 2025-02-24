from typing import List, Optional

import zarr

from .omezarrmeta import (
    Axis, OMEZarrMeta, DataSet, CoordinateTransformation, MSMetadata,
    MultiScaleImage,
    Omero, Channel, Window, RDefs
)

def create_ome_zarr_metadata(
        zarr_group_uri: str,
        name: str,
        coordinate_scales: List[float],
        downsample_factors: List,
        create_omero_block: bool = False,
        channel_labels: Optional[List[str]] = None
    ) -> OMEZarrMeta:
    """Read a Zarr group and generate the OME-Zarr metadata for that group,
    effectively turning a group of Zarr arrays into an OME-Zarr.
    
    If downsample factors are provided, use those to calculate scale transforms,
    otherwise calculate them from the sizes of the"""

    # Open the group and find the arrays in it
    group = zarr.open_group(zarr_group_uri, mode='r')
    array_keys = list(group.array_keys())

    # Make sure we sort in increasing numerical order - this assumes arrays are "0", "1", etc
    array_keys.sort(key=lambda x: int(x))
    n_pyramid_levels = len(array_keys)

    dim_ratios = [
        [(1.0 / f) ** n for f in downsample_factors]
        for n in range(n_pyramid_levels)
    ]

    # Use these scaling factors together with base coordinate scales to generate DataSet objects
    datasets = generate_dataset_objects(coordinate_scales, dim_ratios, array_keys)
    multiscales = generate_multiscales(datasets, name)
    if create_omero_block:
        omero = create_omero_metadata_object(str(zarr_group_uri), channel_labels)
    else:
        omero = None

    ome_zarr_metadata = OMEZarrMeta(
        multiscales=[multiscales],
        omero=omero
    )

    return ome_zarr_metadata


def round_to_sigfigs(x: float, sigfigs: int = 3) -> float:
    """
    Round a float to a specified number of significant figures.
    
    Args:
        x: Number to round
        sigfigs: Number of significant figures to keep (default 3)
    
    Returns:
        Float rounded to specified significant figures
    """
    if x == 0:
        return 0
    return float(f'{x:.{sigfigs}g}')


def generate_dataset_objects(
    start_scales,
    factors,
    path_keys
):
    datasets = [
        DataSet(
            path=path_label,
            coordinateTransformations=[
                CoordinateTransformation(
                    scale = [
                        round_to_sigfigs(start_scale / factor, 3)
                        for (start_scale, factor) in zip(start_scales, factors[n])
                    ],
                    type="scale"
                )
            ]
        )
        for n, path_label in enumerate(path_keys)
    ]

    return datasets


def generate_axes():
    """Generate the Axis objects we use as standard for all of our OME-Zarr images."""

    initial_axes = [
        Axis(
            name="t",
            type="time",
        ),
        Axis(
            name="c",
            type="channel",
        )
    ]
    spatial_axes = [
        Axis(
            name=name,
            type="space",
            unit='meter'
        )
        for name in ["z", "y", "x"]
    ]

    return initial_axes + spatial_axes


def generate_multiscales(datasets, name):

    multiscales = MultiScaleImage(
        datasets=datasets,
        axes=generate_axes(),
        version="0.4",
        name=name,
        metadata=MSMetadata(
            method="BIA scripts",
            version="0.1"
        )
    )

    return multiscales


def create_omero_metadata_object(zarr_group_uri: str, channel_labels: List[str] = None):
    group = zarr.open_group(zarr_group_uri)
    array_keys = list(group.array_keys())

    smallest_array = group[array_keys[-1]]
    min_val = smallest_array[:].min() # type: ignore
    max_val = smallest_array[:].max() # type: ignore

    largest_array = group[array_keys[0]]
    if len(largest_array.shape) != 5:
        raise ValueError("Input array must be 5-dimensional (t,c,z,y,x)")
        
    tdim, cdim, zdim, _, _ = largest_array.shape

    window = Window(
        min=0.0,
        max=255.0,
        start=min_val,
        end=max_val
    )

    # Define colors for channels
    colors = ["FF0000", "00FF00", "0000FF", "00FFFF", "FFFF00", "FF00FF"]
    
    channels = []
    for c in range(cdim):
        if cdim == 1:
            color = "FFFFFF"  # White for single channel
        else:
            color = colors[c % len(colors)]  # Cycle through colors
            
        channel = Channel(
            color=color,
            coefficient=1,
            active=c < 3,  # First three channels active
            label=channel_labels[c] if channel_labels else f"Channel {c}",
            window=window,
            family="linear",
            inverted=False
        )
        channels.append(channel)

    rdefs = RDefs(
        model="color" if cdim > 1 else "greyscale",
        defaultT=tdim//2,
        defaultZ=zdim//2
    )
    
    omero = Omero(
        rdefs=rdefs,
        channels=channels
    )

    return omero
