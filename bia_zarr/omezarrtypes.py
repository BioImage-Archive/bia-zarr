from enum import Enum

import rich
import zarr

from .omezarrmeta import OMEZarrMeta


class OMEZarrType(str, Enum):
    # https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr/0
    v04image = 'v04image'
    # https://uk1s3.embassy.ebi.ac.uk/ebi-ngff-challenge-2024/4ffaeed2-fa70-4907-820f-8a96ef683095.zarr
    v05image = 'v05image'
    # https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr
    bf2rawtr = 'bf2rawtr'
    # https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0056B/7361.zarr
    hcs = 'hcs'


def get_ome_zarr_type(ome_zarr_url: str) -> OMEZarrType:
    
    zarr_group = zarr.open_group(ome_zarr_url, mode='r')
    ome_zarr_type = determine_ome_zarr_type(zarr_group)

    return ome_zarr_type


def determine_ome_zarr_type(zarr_group):

    if zarr_group.metadata.zarr_format == 2:
        if 'plate' in zarr_group.attrs:
            return OMEZarrType.hcs
        if dict(zarr_group.attrs) == {'bioformats2raw.layout': 3}:
            return OMEZarrType.bf2rawtr
        else:
            try:
                OMEZarrMeta.model_validate(zarr_group.attrs)
                return OMEZarrType.v04image
            except IOError:
                raise
    elif zarr_group.metadata.zarr_format == 3:
        try:
            OMEZarrMeta.model_validate(zarr_group.attrs['ome'])
            return OMEZarrType.v05image
        except IOError:
            raise

    raise Exception("Unknown")