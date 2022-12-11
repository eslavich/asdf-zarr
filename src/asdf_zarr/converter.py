import math

from asdf.extension import Converter
from asdf.tags.core import ndarray
import numpy as np

from .storage import AsdfStorage


def _set_at_index(sequence, key, value):
    indices = [int(k) for k in key.split(".")]
    for idx in indices[:-1]:
        sequence = sequence[idx]
    sequence[indices[-1]] = value


def _create_sources(shape, chunk_shape):
    num_chunks = math.ceil(shape[0] / chunk_shape[0])
    if len(shape) == 1:
        return [None] * num_chunks
    else:
        result = []
        for _ in range(num_chunks):
            result.append(_create_sources(shape[1:], chunk_shape[1:]))
        return result


def _create_data_callback(obj, array_key):
    # TODO: Support Fortran array layout.
    return lambda: np.ascontiguousarray(obj[tuple(array_key)])


class ChunkedNdarrayConverter(Converter):
    tags = ["asdf://asdf-format.org/chunked_ndarray/tags/chunked_ndarray-*"]
    types = ["zarr.core.Array"]

    def to_yaml_tree(self, obj, tag, ctx):
        shape = list(obj.shape)
        chunk_shape = list(obj.chunks)
        datatype, byteorder = ndarray.numpy_dtype_to_asdf_datatype(obj.dtype)

        # TODO: We'd eventually like to store this in an additional block
        # instead of in the YAML.
        sources = _create_sources(shape, chunk_shape)

        for key in obj.store.keys():
            if key != ".zarray":
                indices = [int(k) for k in key.split(".")]
                array_key = []
                for i, idx in enumerate(indices):
                    array_key.append(slice(idx * chunk_shape[i], (idx + 1) * chunk_shape[i]))
                source = ctx.reserve_block((id(obj), key), _create_data_callback(obj, array_key))
                _set_at_index(sources, key, source)

        return {
            "shape": shape,
            "chunk_shape": chunk_shape,
            "datatype": datatype,
            "byteorder": byteorder,
            "sources": sources
        }

    def from_yaml_tree(self, node, tag, ctx):
        from zarr.core import Array

        storage = AsdfStorage(
            node["shape"],
            node["chunk_shape"],
            ndarray.asdf_datatype_to_numpy_dtype(node["datatype"], node["byteorder"]),
            node["sources"],
            ctx.block_manager,
        )

        return Array(storage)
