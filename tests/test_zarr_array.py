import asdf
import numpy as np
import psutil
import pytest
import zarr
from numpy.testing import assert_array_equal
import shutil


def create_chunked_array():
    arr = zarr.creation.create((6, 9), chunks=(3, 3))
    arr[0:3, 0:3] = 1
    arr[0:3, 3:6] = 2
    arr[0:3, 6:9] = 3
    arr[3:6, 0:3] = 4
    arr[3:6, 3:6] = 5
    arr[3:6, 6:9] = 6
    return arr


def assert_chunked_array_equal(arr1, arr2):
    assert_array_equal(arr1[:], arr2[:])


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("compression", ["input", "zlib"])
def test_round_trip(tmp_path, copy_arrays, lazy_load, compression):
    arr = create_chunked_array()
    file_path = tmp_path / "test.asdf"

    with asdf.AsdfFile() as af:
        af["arr"] = arr
        af.write_to(file_path, all_array_compression=compression)

    with asdf.open(file_path, copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        assert_chunked_array_equal(arr, af["arr"])


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("compression", ["input", "zlib"])
def test_update(tmp_path, copy_arrays, lazy_load, compression):
    arr = create_chunked_array()
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["foo"] = "bar"
        af["arr"] = arr
        af.write_to(file_path, all_array_compression=compression)

    # Update an unrelated key
    with asdf.open(file_path, mode="rw", copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        assert_chunked_array_equal(arr, af["arr"])
        assert len(af.blocks._internal_blocks) == 6
        af["foo"] = "barbaz"
        af.update()
        # Ensure that we're not experiencing any block weirdness
        assert len(af.blocks._internal_blocks) == 6
        assert_chunked_array_equal(arr, af["arr"])

    # Now update a chunk
    with asdf.open(file_path, mode="rw", copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        assert len(af.blocks._internal_blocks) == 6
        assert_chunked_array_equal(arr, af["arr"])
        assert af["foo"] == "barbaz"
        af["arr"][0:3, 0:3] = 0
        af.update()
        assert len(af.blocks._internal_blocks) == 6
        assert_array_equal(af["arr"][0:3, 0:3], np.full((3, 3), 0))


def test_sparse_array(tmp_path):
    arr = zarr.creation.create((6, 9), chunks=(3, 3), dtype=np.int64)
    arr[0:3, 0:3] = 1
    arr[3:6, 6:9] = 6

    file_path = tmp_path / "test.asdf"

    with asdf.AsdfFile() as af:
        af["arr"] = arr
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert len(af.blocks._internal_blocks) == 2
        assert_chunked_array_equal(af["arr"], arr)


@pytest.mark.parametrize("copy_arrays", [True, False])
def test_large_array(tmp_path, copy_arrays):
    total_memory_bytes = psutil.virtual_memory().total
    chunk_size = 1024 ** 3
    array_size = ((total_memory_bytes * 2) // chunk_size) * chunk_size

    arr = zarr.creation.create((array_size,), chunks=(chunk_size,), dtype=np.uint8, store=zarr.storage.TempStore())
    for i in range(0, array_size, chunk_size):
        arr[i:i + chunk_size] = np.random.randint(0, 255, size=chunk_size, dtype=np.uint8)

    file_path = tmp_path / "test.asdf"

    with asdf.AsdfFile() as af:
        af["arr"] = arr
        af.write_to(file_path)

    with asdf.open(file_path, copy_arrays=copy_arrays, lazy_load=True) as af:
        for i in range(0, array_size, chunk_size):
            assert_array_equal(af["arr"][i:i + chunk_size], arr[i:i + chunk_size])

    file_path.unlink()
    shutil.rmtree(arr.store.path)
