from pathlib import Path
import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from asdf.resource import DirectoryResourceMapping

from asdf.extension import ManifestExtension

import asdf_zarr
from .converter import ChunkedNdarrayConverter


def get_resource_mappings():
    resources_root = importlib_resources.files(asdf_zarr) / "resources"
    if not resources_root.is_dir():
        raise RuntimeError("Missing resources directory")

    return [
        DirectoryResourceMapping(
            resources_root / "schemas",
            "asdf://asdf-format.org/chunked_ndarray/schemas/",
        ),
        DirectoryResourceMapping(
            resources_root / "manifests",
            "asdf://asdf-format.org/chunked_ndarray/manifests/",
        ),
    ]


def get_extensions():
    return [
        ManifestExtension.from_uri(
            "asdf://asdf-format.org/chunked_ndarray/manifests/chunked_ndarray-0.1.0",
            converters=[ChunkedNdarrayConverter()]
        )
    ]
