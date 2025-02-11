import rich
import typer
from typing import Tuple

from .proxyimage import open_ome_zarr_image
from .rendering import generate_padded_thumbnail_from_ngff_uri
from .omezarrtypes import get_ome_zarr_type


app = typer.Typer()


@app.command('get-type')
def determine_ome_zarr_type(url: str):
    ome_zarr_type = get_ome_zarr_type(url)

    rich.print(f"Determined type as: {ome_zarr_type}")


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
