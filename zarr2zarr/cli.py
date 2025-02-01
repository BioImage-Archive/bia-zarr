import rich
import typer

from .proxyimage import open_ome_zarr_image
from .rendering import generate_padded_thumbnail_from_ngff_uri


app = typer.Typer()


@app.command()
def info(ome_zarr_uri: str):
    im = generate_padded_thumbnail_from_ngff_uri(ome_zarr_uri)
    im.save('foo.png')


if __name__ == "__main__":
    app()