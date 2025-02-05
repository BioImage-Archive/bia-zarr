import pytest
from ..proxyimage import open_ome_zarr_image

def test_open_ome_zarr_image_chunks():
    """Test that opening a known zarr image returns expected chunk sizes and version."""
    uri = "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/postprocess/rechunking/F107_A3.zarr"
    image = open_ome_zarr_image(uri)
    
    assert image.zarr_version == 2
    assert image.chunk_scheme == [1, 1, 64, 64, 64]
    assert image.shard_scheme == None


    uri = "https://bioimaging-01-pub.livingobjects.ebi.ac.uk/phase1test/platy-sbem-6dpf-1-whole-raw-z3.ome.zarr"
    image = open_ome_zarr_image(uri)
    
    assert image.zarr_version == 3
    assert image.chunk_scheme == [1, 1, 64, 64, 64]
    assert image.shard_scheme == [1, 1, 256, 256, 256]
