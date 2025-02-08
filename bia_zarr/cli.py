import rich
import typer
from typing import Tuple

from .proxyimage import open_ome_zarr_image
from .rendering import generate_padded_thumbnail_from_ngff_uri


app = typer.Typer()


@app.command()
def spike_write(url: str):
    ome_zarr_image = open_ome_zarr_image(url)
    from .zarr2zarr import write_array_to_disk_chunked
    import tensorstore as ts

    input_array_url = url + '/' + ome_zarr_image.path_keys[0]
    # array = ome_zarr_image.zarr_group[ome_zarr_image.path_keys[0]]
    source = ts.open({
        'driver': 'zarr',
        'kvstore': input_array_url,
    }).result()

    target_chunks = [1, 1, 64, 64, 64]

    write_array_to_disk_chunked(source,  'goo', target_chunks)


@app.command()
def validate_ome_zarr_thing(url: str):
    from .thing import open_ome_zarr_thing

    open_ome_zarr_thing(url)


@app.command()
def validate_ome_zarr_image(ome_zarr_url: str):
    ome_zarr_image = open_ome_zarr_image(ome_zarr_url)

    rich.print(ome_zarr_image)


@app.command()
def thumbnail(
    ome_zarr_url: str,
    output: str = typer.Option(..., help="Output filename"),
    dimensions: Tuple[int, int] = typer.Option((256, 256), help="Output dimensions (width, height)")
):
    """Generate a thumbnail from an OME-ZARR image with specified dimensions."""
    im = generate_padded_thumbnail_from_ngff_uri(ome_zarr_url, dims=dimensions)
    im.save(output)


if __name__ == "__main__":
    app()
