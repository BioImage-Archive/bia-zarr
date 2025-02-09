from enum import Enum

import rich
import zarr
from pydantic import BaseModel

from .omezarrmeta import OMEZarrMeta


class OMEZarrThing(str, Enum):
    # https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr/0
    v04image = 'v04image'
    # https://uk1s3.embassy.ebi.ac.uk/ebi-ngff-challenge-2024/4ffaeed2-fa70-4907-820f-8a96ef683095.zarr
    v05image = 'v05image'
    # https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BIAD144/IM1.zarr
    bf2rawtr = 'bf2rawtr'
    # https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0056B/7361.zarr
    hcs = 'hcs'


def open_ome_zarr_thing(url: str):
    zarr_group = zarr.open_group(url, mode='r')

    thing_type = determine_thing(zarr_group)

    rich.print(thing_type)


def determine_thing(zarr_group):

    if zarr_group.metadata.zarr_format == 2:
        if 'plate' in zarr_group.attrs:
            return OMEZarrThing.hcs
        if dict(zarr_group.attrs) == {'bioformats2raw.layout': 3}:
            return OMEZarrThing.bf2rawtr
        else:
            try:
                OMEZarrMeta.model_validate(zarr_group.attrs)
                return OMEZarrThing.v04image
            except IOError:
                raise
    elif zarr_group.metadata.zarr_format == 3:
        try:
            OMEZarrMeta.model_validate(zarr_group.attrs['ome'])
            return OMEZarrThing.v05image
        except IOError:
            raise

    raise Exception("Unknown")