import math

import numpy as np
from asdf.extension import Converter
from asdf.tags.core import ndarray

from .storage import AsdfStorage
from .util import get_at_index, iter_keys, set_at_index


class ChunkedNdarrayConverter(Converter):
    tags = ["asdf://asdf-format.org/chunked_ndarray/tags/chunked_ndarray-*"]
    types = ["zarr.core.Array"]

    def to_yaml_tree(self, obj, tag, ctx):
        shape = list(obj.shape)
        chunk_shape = list(obj.chunks)
        datatype, byteorder = ndarray.numpy_dtype_to_asdf_datatype(obj.dtype)
        fill_value = obj.fill_value

        # TODO: We'd eventually like to store this in an additional block
        # instead of in the YAML.
        sources = _create_sources(shape, chunk_shape)
        chunk_size = np.prod(chunk_shape) * obj.dtype.itemsize

        for key in obj.store.keys():
            if key != ".zarray":
                source = ctx.reserve_block(
                    (id(obj), key), _create_data_callback(obj, key, chunk_shape), data_size=chunk_size
                )
                set_at_index(sources, key, source)

        return {
            "shape": shape,
            "chunk_shape": chunk_shape,
            "datatype": datatype,
            "byteorder": byteorder,
            "fill_value": fill_value,
            "sources": sources,
        }

    def from_yaml_tree(self, node, tag, ctx):
        from zarr.core import Array

        ndarray.asdf_datatype_to_numpy_dtype(node["datatype"], node["byteorder"]),

        storage = AsdfStorage(
            node["shape"],
            node["chunk_shape"],
            ndarray.asdf_datatype_to_numpy_dtype(node["datatype"], node["byteorder"]),
            node.get("fill_value", 0),
            node["sources"],
            ctx.block_manager,
        )

        array = Array(storage)
        storage._array_id = id(array)

        for key in iter_keys(node["shape"], node["chunk_shape"]):
            source = get_at_index(node["sources"], key)
            if source is not None:
                ctx.identify_block(source, (id(array), key))

        return array


def _create_sources(shape, chunk_shape):
    num_chunks = math.ceil(shape[0] / chunk_shape[0])
    if len(shape) == 1:
        return [None] * num_chunks
    else:
        result = []
        for _ in range(num_chunks):
            result.append(_create_sources(shape[1:], chunk_shape[1:]))
        return result


def _create_data_callback(obj, key, chunk_shape):
    if isinstance(obj.store._mutable_mapping, AsdfStorage):
        return lambda: obj.store._mutable_mapping.get_ndarray(key)
    else:
        indices = [int(k) for k in key.split(".")]
        array_key = []
        for i, idx in enumerate(indices):
            array_key.append(slice(idx * chunk_shape[i], (idx + 1) * chunk_shape[i]))
        # TODO: Support Fortran array layout.
        return lambda: np.frombuffer(obj[tuple(array_key)], dtype=np.uint8)
