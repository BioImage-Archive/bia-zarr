from typing import List
from pathlib import Path

import zarr

def rechunk_and_write_array(
        input_array: zarr.Array,
        output_dirpath: Path,
        target_chunks: List[int]
):

    source = ts.open({
        'driver': 'zarr',
        'kvstore': input_array_uri,
    }).result()