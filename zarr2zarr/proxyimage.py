import zarr
from ome_zarr_models import open_ome_zarr # type: ignore



def open_ome_zarr_image(ome_zarr_image_uri: str):
    zarr_group = zarr.open_group(ome_zarr_image_uri)

    image_metadata = open_ome_zarr(zarr_group).attributes

    input_array = zarr_group[image_metadata.multiscales[0].datasets[0].path]
    import rich
    rich.print(input_array)

