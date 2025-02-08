import rich
import typer
from typing import Tuple

from .proxyimage import open_ome_zarr_image
from .rendering import generate_padded_thumbnail_from_ngff_uri


app = typer.Typer()


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
    dimensions: Tuple[int, int] = typer.Option((256, 256), help="Output dimensions (width, height)"),
    channels: str = typer.Option(None, help="Comma-separated list of channel indices to include")
):
    """Generate a thumbnail from an OME-ZARR image with specified dimensions."""
    channel_indices = None
    if channels:
        channel_indices = [int(c) for c in channels.split(',')]
    im = generate_padded_thumbnail_from_ngff_uri(ome_zarr_url, dims=dimensions, channels=channel_indices)
    im.save(output)


if __name__ == "__main__":
    app()
