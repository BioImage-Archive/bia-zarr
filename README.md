# bia-zarr Module Overview

This module provides utilities for working with Zarr and OME-Zarr datasets, primarily focused on reading, writing, and manipulating image data in these formats.  It includes functionality for generating OME-Zarr metadata, creating thumbnails, and converting between different Zarr formats, with a particular emphasis on handling BioImaging Archive (BIA) data. It is designed for compatibility with both Zarr v2 and v3.

## Key Components

The module is organized into several submodules, each with specific responsibilities:

*   **`bia_zarr.arrays`**:  Provides functions related to array manipulation, particularly rechunking and writing Zarr arrays.

*   **`bia_zarr.cli`**: Implements a command-line interface (CLI) using `typer` for tasks like validating OME-Zarr structures and generating thumbnails.  This is the main entry point for users interacting with the library from the command line.

*   **`bia_zarr.genmeta`**: Contains functions for generating OME-Zarr metadata from existing Zarr groups, essentially adding the necessary metadata to make a standard Zarr group compliant with the OME-Zarr specification. It includes utilities for creating `DataSet`, `Axis`, `CoordinateTransformation`, and `MultiScaleImage` objects as defined by the OME-NGFF specification.

*   **`bia_zarr.omezarrmeta`**: Defines Pydantic models representing the OME-Zarr metadata structure (v0.4).  This includes classes like `OMEZarrMeta`, `MultiScaleImage`, `DataSet`, `Axis`, `CoordinateTransformation`, `Omero`, `Channel`, `Window`, and `RDefs`.  These models are used for validation and programmatic manipulation of OME-Zarr metadata.

*   **`bia_zarr.proxyimage`**:  Provides a higher-level abstraction for interacting with OME-Zarr images. Key classes and functions include:
    *   `OMEZarrImage`:  A Pydantic model representing an OME-Zarr image, encapsulating its dimensions, scaling, chunking information, and underlying Zarr group and NGFF metadata.  This is a central class for interacting with image data.
    *   `open_ome_zarr_image()`:  Opens an OME-Zarr image from a URI and returns an `OMEZarrImage` instance.  This is the primary function for loading OME-Zarr images.
    *   `open_ome_zarr()`: Parses a Zarr group's attributes and returns an `OMEZarrMeta` object, validating the metadata against the OME-Zarr specification.
    *   `get_array_with_min_dimensions()`: Efficiently retrieves the smallest resolution level Dask array that meets specified minimum dimensions. This is important for operations like thumbnail generation where you don't need the full resolution image.
    *   `reshape_to_5D()`, `sizes_from_array_shape_and_dimension_str()`:  Utilities for handling array dimensions, converting between different representations, and ensuring consistency.
    *   `calculate_scale_ratios()`. `validate_scale_ratios_and_extract_xyz()`, `calculate_voxel_to_physical_factors()`:  Functions to calculate and validate scale factors between pyramid levels, ensuring consistent downsampling ratios and extracting XY and Z scaling information.
    *   `generate_dataset_objects()` and `generate_dataset_objects_from_scaling()`: Creates dataset objects with scale transformations.

*   **`bia_zarr.rendering`**:  Handles the generation of 2D image renderings from multi-dimensional OME-Zarr data. Key components include:
    *   `ChannelRenderingSettings`, `RenderingInfo`, `BoundingBox2DRel`, `BoundingBox2DAbs`, `PlaneRegionSelection`, `RenderingView`: Pydantic models to define rendering parameters, including bounding boxes, channel settings, and region selections.
    *   `render_proxy_image()`:  The core function for rendering a 2D plane from an `OMEZarrImage` object, allowing for channel selection, colormap application, windowing, and region of interest cropping. It uses `microfilm` for applying colormaps.
    *   `generate_padded_thumbnail_from_ngff_uri()`: Generates a thumbnail image from an OME-Zarr URI, handling padding, scaling, and basic color adjustments.  This function uses `render_proxy_image` internally.
    *   `scale_to_uint8()`, `apply_window()`: Utility functions for image processing tasks like scaling to 8-bit and applying windowing functions.
    *   `pad_to_target_dims()`: Pads an image to specified dimensions.

*   **`bia_zarr.thing`**:  Determines the "type" of an OME-Zarr based on its metadata, identifying whether it's a standard image (v0.4 or v0.5), an HCS plate, or a bioformats2raw transformed dataset.

*   **`bia_zarr.write`**: Provides functionality for writing arrays to disk in OME-Zarr format. Key functions include:
    *   `write_array_as_ome_zarr()`:  Writes a NumPy array to disk as an OME-Zarr, handling dimension normalization to TCZYX, pyramid level generation, and metadata creation (using `create_ome_zarr_metadata` from `genmeta.py`). It supports both Zarr v2 and v3.
    *   `downsample_array_and_write_to_dirpath()`: Downsamples a Zarr array using TensorStore and writes the result to a new Zarr array, handling chunking configurations.
    *   `write_array_to_disk_chunked()`: Writes an array to disk with specified chunking, processing large arrays in smaller chunks to manage memory usage.
    * `derive_n_levels():` Calculates the require number of resolution levels for a given level 0 shape.
     * `normalize_array_dimensions()` Normalizes potentially-irregular input arrays into the standard 5D TCZYX format used by OME-Zarr.

*   **`bia_zarr.zarr2zarr`**: *[This submodule appears to be under development, but contains the beginnings of functionality for converting between Zarr datasets, likely involving rechunking, downsampling, and metadata updates.]* Contains `ZarrConversionConfig`, a Pydantic model to specify conversion parameters.

*   **`bia_zarr.tests`**: Contains unit tests, particularly for `proxyimage.py` and `write.py`.

## Usage Examples

*   **Loading an OME-Zarr image and accessing its metadata:**

    ```python
    from bia_zarr.proxyimage import open_ome_zarr_image

    image = open_ome_zarr_image("path/to/image.zarr")
    print(image.sizeX, image.sizeY)  # Accessing dimensions
    print(image.ngff_metadata)      # Accessing OME-Zarr metadata
    ```

*   **Working with different OME-Zarr types and generating thumbnails:**

    ```python
    from bia_zarr.omezarrtypes import get_ome_zarr_type, get_single_image_uri_from_url
    from bia_zarr.rendering import generate_padded_thumbnail_from_ngff_uri

    # First determine the type of OME-Zarr container
    zarr_type = get_ome_zarr_type("path/to/container.zarr")
    print(f"Container type: {zarr_type}")  # Could be v04image, v05image, bf2rawtr, or hcs

    # Get a single image URI regardless of container type
    # For HCS plates, this will return the first image in the first well
    image_uri = get_single_image_uri_from_url("path/to/container.zarr")

    # Generate a thumbnail from the image
    im = generate_padded_thumbnail_from_ngff_uri(image_uri, dims=(128, 128))
    im.save("thumbnail.png")
    ```

    The module supports several OME-Zarr container types:
    * `v04image`: A standard OME-NGFF v0.4 image
    * `v05image`: A standard OME-NGFF v0.5 image
    * `bf2rawtr`: A Bio-Formats2Raw transformed dataset
    * `hcs`: A High Content Screening plate, where `get_single_image_uri_from_url()` will return the path to the first image in the first well

*   **Writing a NumPy array as an OME-Zarr:**

    ```python
    import numpy as np
    from bia_zarr.write import write_array_as_ome_zarr

    array = np.random.rand(1, 3, 1024, 1024)  # Example 4D array (TCYX)
    write_array_as_ome_zarr(array, "tcyx", "output.zarr")
    ```

*   **Running from the command line (thumbnail generation):**

    ```bash
    python bia_zarr/cli.py thumbnail --ome-zarr-url path/to/image.zarr --output thumbnail.png --dimensions 256 256 --channels 0,1,2
    ```

## Dependencies

*   `zarr`: For working with Zarr arrays.
*   `fsspec`: For file system access (used in the initial commented-out code, may be used more extensively in the future).
*   `aiohttp`: For asynchronous HTTP requests (used in the initial commented-out code).
*   `rich`:  For rich text formatting and console output.
*   `typer`: For building the command-line interface.
*   `pydantic`: For data validation and defining data models (especially for OME-Zarr metadata).
*   `dask`: For working with lazy arrays and parallel computation, particularly for handling large datasets.
*   `PIL` (Pillow): For image processing (resizing, padding, format conversion).
*   `numpy`: For numerical operations and array manipulation.
*   `tensorstore`: For efficient storage and retrieval of multi-dimensional array data (used for downsampling and rechunking).
* `microfilm`: Used in rendering.
*  `pytest`: Used in testing.

## Notes
The initial commented-out code snippets in `minzarr.py` suggest an early intention to use asynchronous HTTP requests for accessing remote Zarr datasets. While this isn't fully implemented in the provided code, it indicates a potential future direction for the module. The `proxyimage` and `rendering` submodules are key for accessing and displaying image data. The use of Pydantic models throughout the module provides strong type checking and validation, making the code more robust and easier to understand. The Zarr v3 support is present.
