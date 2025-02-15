import rich
import typer
import json
from typing import Tuple, Optional, Annotated
from pathlib import Path

from .proxyimage import open_ome_zarr_image
from .rendering import generate_padded_thumbnail_from_ngff_uri
from .omezarrtypes import get_ome_zarr_type


app = typer.Typer()


@app.command('get-type')
def determine_ome_zarr_type(url: str):
    ome_zarr_type = get_ome_zarr_type(url)

    rich.print(f"Determined type as: {ome_zarr_type}")


@app.command()
def get_image_uri(url: str):
    """Get a single image URI from any OME-NGFF container."""
    from .omezarrtypes import get_single_image_uri_from_url
    image_uri = get_single_image_uri_from_url(url)
    rich.print(image_uri)


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
    from .omezarrtypes import get_single_image_uri_from_url
    
    # First get the single image URL from any container type
    image_uri = get_single_image_uri_from_url(ome_zarr_url)
    
    # Then generate the thumbnail from that image
    channel_indices = None
    if channels:
        channel_indices = [int(c) for c in channels.split(',')]
    im = generate_padded_thumbnail_from_ngff_uri(image_uri, dims=dimensions, channels=channel_indices)
    im.save(output)


@app.command()
def zarr2zarr(
    ome_zarr_uri: str, 
    output_base_dirpath: Path,
    conversion_config: Annotated[Optional[str], typer.Argument()] = "{}"
):
    """Convert between OME-Zarr formats with optional transformations."""
    from .zarr2zarr import zarr2zarr as _zarr2zarr, ZarrConversionConfig
    
    config = ZarrConversionConfig(**json.loads(conversion_config))
    _zarr2zarr(ome_zarr_uri, output_base_dirpath, config)


if __name__ == "__main__":
    app()
