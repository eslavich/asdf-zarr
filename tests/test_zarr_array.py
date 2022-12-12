import asdf
import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal


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
        af["foo"] = "baz"
        af.update()
        # Ensure that we're not experiencing any block weirdness
        assert len(af.blocks._internal_blocks) == 6
        assert_chunked_array_equal(arr, af["arr"])

    # Now update a chunk
    with asdf.open(file_path, mode="rw", copy_arrays=copy_arrays, lazy_load=lazy_load) as af:
        assert len(af.blocks._internal_blocks) == 6
        assert_chunked_array_equal(arr, af["arr"])
        assert af["foo"] == "baz"
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
