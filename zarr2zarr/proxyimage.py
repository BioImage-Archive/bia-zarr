import zarr

from .omezarrmeta import OMEZarrMeta

# NOTE - replace with the ome zarr models version once v3 support exists
def open_ome_zarr(zarr_group: zarr.Group):
    attrs_dict = zarr_group.attrs.asdict()

    image_metadata = OMEZarrMeta.parse_obj(attrs_dict["ome"])

    import rich
    rich.print(image_metadata)


class OMEZarrImage(BaseModel):

    sizeX: int
    sizeY: int
    sizeZ: int = 1
    sizeC: int = 1
    sizeT: int = 1

    # Valid values for dimensions are {'tczyx', 'zyx', 'tcyx', 'czyx'}
    dimensions: str = "tczyx"
    zgroup: zarr.Group

    n_scales: int = 1
    xy_scaling: float = 1.0
    z_scaling: float = 1.0
    path_keys: List[str]= []
        
    PhysicalSizeX: Optional[float] = None
    PhysicalSizeY: Optional[float] = None
    PhysicalSizeZ: Optional[float] = None

    ngff_metadata: ZMeta | None = None

    class Config:
        arbitrary_types_allowed=True


def open_ome_zarr_image(ome_zarr_image_uri: str):
    zarr_group = zarr.open_group(ome_zarr_image_uri)

    open_ome_zarr(zarr_group)
    # image_metadata = open_ome_zarr(zarr_group).attributes

    # input_array = zarr_group[image_metadata.multiscales[0].datasets[0].path]
    # import rich
    # rich.print(input_array)

