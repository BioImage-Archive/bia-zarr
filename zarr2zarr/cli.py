import typer

from .proxyimage import open_ome_zarr_image


app = typer.Typer()


@app.command()
def info(ome_zarr_uri: str):
    open_ome_zarr_image(ome_zarr_uri)


if __name__ == "__main__":
    app()