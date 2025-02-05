import rich
import typer
from typing import Tuple

from .proxyimage import open_ome_zarr_image
from .rendering import generate_padded_thumbnail_from_ngff_uri


app = typer.Typer()


@app.command()
def validate_ome_zarr(ome_zarr_uri: str):
    ome_zarr_image = open_ome_zarr_image(ome_zarr_uri)

    rich.print(ome_zarr_image)


@app.command()
def thumbnail(
    ome_zarr_uri: str,
    output: str = typer.Option(..., help="Output filename"),
    dimensions: Tuple[int, int] = typer.Option((256, 256), help="Output dimensions (width, height)")
):
    """Generate a thumbnail from an OME-ZARR image with specified dimensions."""
    im = generate_padded_thumbnail_from_ngff_uri(ome_zarr_uri, dims=dimensions)
    im.save(output)


if __name__ == "__main__":
    app()
