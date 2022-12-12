import json
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

import numpy as np

from .util import get_at_index, iter_keys


class AsdfStorage(MutableMapping):
    def __init__(self, shape, chunk_shape, dtype, fill_value, sources, block_manager):
        self._config = {
            "zarr_format": 2,
            "shape": shape,
            "chunks": chunk_shape,
            "dtype": str(dtype),
            "compressor": None,
            "fill_value": fill_value,
            "order": "C",
            "filters": None,
        }

        self._sources = sources
        self._block_manager = block_manager
        self.__temp_dir = None
        self._temp_file_map = {}
        self._deleted_keys = set()
        self._array_id = None

    def __getitem__(self, key):
        if key == ".zarray":
            return json.dumps(self._config)

        if key in self._deleted_keys:
            raise KeyError(key)

        if key in self._temp_file_map:
            return self._temp_file_map[key].read_bytes()

        source = get_at_index(self._sources, key)
        if source is None:
            raise KeyError(key)

        return self._block_manager.get_block_by_key((self._array_id, key)).data.tobytes()

    def get_ndarray(self, key):
        if key in self._deleted_keys:
            raise KeyError(key)

        if key in self._temp_file_map:
            # TODO: Support Fortran array layout.
            return np.frombuffer(self._temp_file_map[key].read_bytes(), dtype=np.uint8)

        return self._block_manager.get_block_by_key((self._array_id, key)).data

    def __setitem__(self, key, value):
        if key == ".zarray":
            raise ValueError("Cannot set zarr config on AsdfStorage class")

        # This library doesn't support flushing changes to disk until
        # AsdfFile.write_to/update is called, but at the same time
        # we don't want to hold all modified chunks in memory.
        # The solution for now is to store them in temporary files.
        path = Path(self._temp_dir.name) / key
        path.write_bytes(value)
        self._temp_file_map[key] = path
        if key in self._deleted_keys:
            self._deleted_keys.remove(key)

    def __delitem__(self, key):
        if key == ".zarray":
            raise ValueError("Cannot delete zarr config on AsdfStorage class")

        self._deleted_keys.add(key)
        if key in self._temp_file_map:
            self._temp_file_map.pop(key)

    def __iter__(self):
        for key in iter_keys(self._config["shape"], self._config["chunks"]):
            if key not in self._deleted_keys and (
                key in self._temp_file_map or get_at_index(self._sources, key) is not None
            ):
                yield key

    def __len__(self):
        return sum(1 for _ in self)

    @property
    def _temp_dir(self):
        if self.__temp_dir is None:
            self.__temp_dir = tempfile.TemporaryDirectory()

        return self.__temp_dir
