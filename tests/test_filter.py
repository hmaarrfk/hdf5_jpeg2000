import os
import subprocess
import sys

import h5py
import numpy as np
import ojph
import pytest

FILTER_ID = 32031
CODESTREAM_START = bytes.fromhex("ff4fff51")
PLANAR = 0x80000000


@pytest.fixture(scope="session", autouse=True)
def plugin_available():
    if not os.environ.get("HDF5_PLUGIN_PATH"):
        pytest.fail("set HDF5_PLUGIN_PATH to the directory that holds the built plugin")
    if not h5py.h5z.filter_avail(FILTER_ID):
        pytest.fail(f"HDF5 cannot load filter {FILTER_ID} from {os.environ['HDF5_PLUGIN_PATH']}")


def random_samples(dtype, shape, seed=0):
    information = np.iinfo(dtype)
    samples = np.random.default_rng(seed).integers(
        information.min, information.max, size=shape, endpoint=True, dtype=np.int64
    )
    samples.flat[0] = information.min
    samples.flat[-1] = information.max
    return samples.astype(dtype)


def stored_filter_values(dataset):
    return dataset.id.get_create_plist().get_filter_by_id(FILTER_ID)[1]


def write_without_plugin(tmp_path, images, chunks, filter_values):
    np.save(tmp_path / "images.npy", images)
    writer = f"""
import h5py, numpy as np, ojph
assert not h5py.h5z.filter_avail({FILTER_ID})
images = np.load({str(tmp_path / "images.npy")!r})
with h5py.File({str(tmp_path / "data.h5")!r}, "w") as file:
    dataset = file.create_dataset(
        "data", shape=images.shape, dtype=images.dtype, chunks={chunks!r},
        compression={FILTER_ID}, compression_opts={filter_values!r}, allow_unknown_filter=True,
    )
    for index, image in enumerate(images):
        dataset.id.write_direct_chunk((index, 0, 0), bytes(ojph.imwrite_to_memory(image)))
"""
    environment = dict(os.environ, HDF5_PLUGIN_PATH=str(tmp_path / "no_plugins"))
    subprocess.run([sys.executable, "-c", writer], env=environment, check=True)


@pytest.mark.parametrize(
    "dtype, element_type",
    [("u1", 0x01), ("i1", 0x81), ("<u2", 0x02), ("<i2", 0x82), ("<u4", 0x04), ("<i4", 0x84), (">u2", 0x102), (">i4", 0x184)],
)
@pytest.mark.parametrize(
    "shape, chunks, components, width, height",
    [
        ((2, 256, 320), (1, 256, 320), None, 320, 256),
        ((2, 100, 130), (1, 64, 64), None, 64, 64),
        ((2, 3, 37, 41), (1, 1, 37, 41), None, 41, 37),
        ((9, 5), (4, 3), None, 3, 4),
        ((2, 30, 44, 3), (1, 30, 44, 3), None, 44, 30),
        ((2, 30, 44, 1), (1, 30, 44, 1), 1, 44, 30),
        ((2, 3, 30, 44), (1, 3, 30, 44), PLANAR | 3, 44, 30),
        ((2, 1, 30, 44), (1, 1, 30, 44), PLANAR | 1, 44, 30),
    ],
)
def test_round_trip_through_filter(tmp_path, dtype, element_type, shape, chunks, components, width, height):
    data = random_samples(np.dtype(dtype), shape)
    compression_opts = None if components is None else (1, element_type, width, height, components)
    with h5py.File(tmp_path / "data.h5", "w") as file:
        file.create_dataset(
            "data", data=data, chunks=chunks, compression=FILTER_ID, compression_opts=compression_opts
        )
    with h5py.File(tmp_path / "data.h5", "r") as file:
        dataset = file["data"]
        expected_components = chunks[-1] if sum(size > 1 for size in chunks) == 3 else 0
        if components is not None:
            expected_components = components
        assert stored_filter_values(dataset) == (1, element_type, width, height, expected_components)
        np.testing.assert_array_equal(dataset[...], data)
        np.testing.assert_array_equal(dataset[..., 1::3], data[..., 1::3])
        filter_mask, chunk = dataset.id.read_direct_chunk((0,) * len(shape))
        assert filter_mask == 0
        assert chunk[:4] == CODESTREAM_START

    first_chunk = data[tuple(slice(0, size) for size in chunks)]
    if expected_components & PLANAR:
        first_chunk = np.moveaxis(first_chunk.squeeze(0), 0, -1)
    decoded = ojph.imread_from_memory(chunk)
    np.testing.assert_array_equal(decoded.reshape(first_chunk.shape), first_chunk.astype(first_chunk.dtype.newbyteorder("=")))


@pytest.mark.parametrize("wavelet", ["rev12", "rev13", "rev53"])
def test_read_chunks_written_directly(tmp_path, wavelet):
    images = random_samples(np.uint16, (3, 64, 80), seed=1) >> 4
    color_images = random_samples(np.uint8, (2, 48, 64, 3), seed=2)
    with h5py.File(tmp_path / "data.h5", "w") as file:
        gray = file.create_dataset(
            "gray", shape=images.shape, dtype=images.dtype, chunks=(1, 64, 80), compression=FILTER_ID
        )
        interleaved = file.create_dataset(
            "interleaved", shape=color_images.shape, dtype=color_images.dtype, chunks=(1, 48, 64, 3), compression=FILTER_ID
        )
        planar = file.create_dataset(
            "planar", shape=(2, 3, 48, 64), dtype=color_images.dtype, chunks=(1, 3, 48, 64),
            compression=FILTER_ID, compression_opts=(1, 0x01, 64, 48, PLANAR | 3),
        )
        for index, image in enumerate(images):
            codestream = ojph.imwrite_to_memory(image, wavelet=wavelet, tlm_marker=True, tileparts_at_resolutions=True)
            gray.id.write_direct_chunk((index, 0, 0), bytes(codestream))
        for index, image in enumerate(color_images):
            codestream = bytes(ojph.imwrite_to_memory(image, wavelet=wavelet, tlm_marker=True, tileparts_at_resolutions=True))
            interleaved.id.write_direct_chunk((index, 0, 0, 0), codestream)
            planar.id.write_direct_chunk((index, 0, 0, 0), codestream)
    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert stored_filter_values(file["gray"]) == (1, 0x02, 80, 64, 0)
        assert stored_filter_values(file["interleaved"]) == (1, 0x01, 64, 48, 3)
        assert stored_filter_values(file["planar"]) == (1, 0x01, 64, 48, PLANAR | 3)
        np.testing.assert_array_equal(file["gray"][...], images)
        np.testing.assert_array_equal(file["interleaved"][...], color_images)
        np.testing.assert_array_equal(file["planar"][...], np.moveaxis(color_images, -1, 1))


@pytest.mark.parametrize("filter_values", [None, (1, 0x0002), (1, 0x0002, 320, 256, 0)])
def test_read_chunks_from_a_writer_without_the_plugin(tmp_path, filter_values):
    images = random_samples(np.uint16, (2, 256, 320), seed=3)
    write_without_plugin(tmp_path, images, (1, 256, 320), filter_values)
    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert stored_filter_values(file["data"]) == (filter_values or ())
        np.testing.assert_array_equal(file["data"][...], images)


def test_virtual_dataset_of_compressed_sources(tmp_path, monkeypatch):
    frames = random_samples(np.uint16, (6, 72, 90), seed=4)
    (tmp_path / "sources").mkdir()
    layout = h5py.VirtualLayout(shape=frames.shape, dtype=frames.dtype)
    for index, frame in enumerate(frames):
        name = f"sources/frame_{index}.h5"
        with h5py.File(tmp_path / name, "w") as file:
            file.create_dataset("image", data=frame, chunks=frame.shape, compression=FILTER_ID)
        layout[index] = h5py.VirtualSource(name, "image", shape=frame.shape)
    with h5py.File(tmp_path / "stack.h5", "w") as file:
        file.create_virtual_dataset("stack", layout, fillvalue=0)

    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    with h5py.File(tmp_path / "stack.h5", "r") as file:
        stack = file["stack"]
        assert stack.is_virtual
        np.testing.assert_array_equal(stack[...], frames)
        np.testing.assert_array_equal(stack[1::2, 3:70:5, ::7], frames[1::2, 3:70:5, ::7])
        np.testing.assert_array_equal(stack[4, 10], frames[4, 10])


@pytest.mark.parametrize(
    "dtype, chunks, compression_opts, message",
    [
        ("f4", (1, 16, 16), None, "must be an integer type"),
        ("i8", (1, 16, 16), None, "it must be 1, 2 or 4 bytes"),
        ("u2", (2, 16, 16, 1), None, "more than two chunk axes are larger than 1"),
        ("u2", (1, 16, 16, 1), (1, 0x82), "does not match the dataset's type"),
        ("u2", (1, 16, 16, 1), (2,), "this filter writes version 1"),
        ("u2", (1, 16, 24, 1), (1, 0x02, 16, 16, 0), "does not match the chunk's image width"),
        ("u2", (1, 16, 16, 3), (1, 0x02, 16, 16, 4), "the last chunk axis is not the component count"),
        ("u2", (1, 3, 16, 16), (1, 0x02, 16, 16, PLANAR), "the planar flag needs a component count"),
    ],
)
def test_dataset_creation_errors(tmp_path, dtype, chunks, compression_opts, message):
    with h5py.File(tmp_path / "data.h5", "w") as file:
        with pytest.raises(Exception, match="jpeg2000 filter") as error:
            file.create_dataset(
                "data", shape=chunks, dtype=dtype, chunks=chunks, compression=FILTER_ID, compression_opts=compression_opts
            )
        assert message in str(error.value)


@pytest.mark.parametrize("damage", ["garbage", "truncated"])
def test_corrupt_codestream_error(tmp_path, damage):
    codestream = bytes(ojph.imwrite_to_memory(random_samples(np.uint16, (64, 64), seed=5)))
    chunk = b"\x12" * 200 if damage == "garbage" else codestream[:100]
    with h5py.File(tmp_path / "data.h5", "w") as file:
        dataset = file.create_dataset("data", shape=(1, 64, 64), dtype="u2", chunks=(1, 64, 64), compression=FILTER_ID)
        dataset.id.write_direct_chunk((0, 0, 0), chunk)
    with h5py.File(tmp_path / "data.h5", "r") as file:
        with pytest.raises(OSError, match="cannot decode chunk"):
            file["data"][...]


@pytest.mark.parametrize(
    "filter_values, message",
    [
        ((1, 0x0002, 320), "version 1 has 0, 2 or 5 cd_values, the dataset has 3"),
        ((2, 0x0002), "this filter reads version 1"),
        ((1, 0x0002, 300, 256, 0), "the codestream is 320 wide and 256 high"),
        ((1, 0x0002, 320, 256, PLANAR), "the planar flag needs a component count"),
        ((1, 0x0002, 320, 256, 3), "the codestream has 1 components, but the dataset stores 3"),
    ],
)
def test_filter_values_that_do_not_match_the_chunks(tmp_path, filter_values, message):
    write_without_plugin(tmp_path, random_samples(np.uint16, (1, 256, 320), seed=6), (1, 256, 320), filter_values)
    with h5py.File(tmp_path / "data.h5", "r") as file:
        with pytest.raises(OSError, match="cannot decode chunk") as error:
            file["data"][...]
        assert message in str(error.value)
