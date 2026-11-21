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
KERNELS = {"irv97": 0, "rev53": 1, "rev13": 2, "rev12": 3}


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


def write_without_plugin(tmp_path, images, chunks, filter_values, dtype=None, encoder_settings=None):
    np.save(tmp_path / "images.npy", images)
    writer = f"""
import h5py, numpy as np, ojph
assert not h5py.h5z.filter_avail({FILTER_ID})
images = np.load({str(tmp_path / "images.npy")!r})
with h5py.File({str(tmp_path / "data.h5")!r}, "w") as file:
    dataset = file.create_dataset(
        "data", shape=images.shape, dtype={str(dtype or images.dtype)!r}, chunks={chunks!r},
        compression={FILTER_ID}, compression_opts={filter_values!r}, allow_unknown_filter=True,
    )
    for index, image in enumerate(images):
        codestream = ojph.imwrite_to_memory(image, **{encoder_settings or {}!r})
        dataset.id.write_direct_chunk((index, 0, 0), bytes(codestream))
"""
    environment = dict(os.environ, HDF5_PLUGIN_PATH=str(tmp_path / "no_plugins"))
    subprocess.run([sys.executable, "-c", writer], env=environment, check=True)


@pytest.mark.parametrize(
    "dtype, element_type",
    [
        ("u1", 0x01), ("i1", 0x81), ("<u2", 0x02), ("<i2", 0x82), ("<u4", 0x04), ("<i4", 0x84),
        (">u2", 0x102), (">i2", 0x182), (">u4", 0x104), (">i4", 0x184),
    ],
)
@pytest.mark.parametrize(
    "shape, chunks, components, width, height, decompositions",
    [
        ((2, 256, 320), (1, 256, 320), None, 320, 256, 2),
        ((2, 100, 130), (1, 64, 64), None, 64, 64, 1),
        ((2, 3, 37, 41), (1, 1, 37, 41), None, 41, 37, 1),
        ((9, 5), (4, 3), None, 3, 4, 1),
        ((2, 30, 44, 3), (1, 30, 44, 3), None, 44, 30, 1),
        ((2, 30, 44, 1), (1, 30, 44, 1), 1, 44, 30, 1),
        ((2, 3, 30, 44), (1, 3, 30, 44), PLANAR | 3, 44, 30, 1),
        ((2, 1, 30, 44), (1, 1, 30, 44), PLANAR | 1, 44, 30, 1),
    ],
)
def test_round_trip_through_filter(tmp_path, dtype, element_type, shape, chunks, components, width, height, decompositions):
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
        assert stored_filter_values(dataset) == (1, element_type, width, height, expected_components, 3, 1, decompositions)
        np.testing.assert_array_equal(dataset[...], data)
        np.testing.assert_array_equal(dataset[..., 1::3], data[..., 1::3])
        skipped_filters, chunk = dataset.id.read_direct_chunk((0,) * len(shape))
        assert skipped_filters == 0
        assert chunk[:4] == CODESTREAM_START

    first_chunk = data[tuple(slice(0, size) for size in chunks)]
    if expected_components & PLANAR:
        first_chunk = np.moveaxis(first_chunk.squeeze(0), 0, -1)
    decoded = ojph.imread_from_memory(chunk)
    np.testing.assert_array_equal(decoded.reshape(first_chunk.shape), first_chunk.astype(first_chunk.dtype.newbyteorder("=")))


@pytest.mark.parametrize("kernel", ["irv97", "rev53", "rev13", "rev12"])
@pytest.mark.parametrize("decompositions", [0, 3, None])
def test_encode_with_each_kernel(request, tmp_path, kernel, decompositions):
    if decompositions == 0 and kernel != "irv97":
        request.applymarker(
            pytest.mark.xfail(strict=True, reason="libojph decodes a 0 sample as 32768 with no decompositions")
        )
    images = random_samples(np.uint16, (2, 256, 320), seed=7) >> 4
    compression_opts = (1, 0x02, 320, 256, 0, 3, KERNELS[kernel])
    if decompositions is not None:
        compression_opts += (decompositions,)
    with h5py.File(tmp_path / "data.h5", "w") as file:
        file.create_dataset("data", data=images, chunks=(1, 256, 320), compression=FILTER_ID, compression_opts=compression_opts)
    with h5py.File(tmp_path / "data.h5", "r") as file:
        dataset = file["data"]
        expected_decompositions = 2 if decompositions is None else decompositions
        assert stored_filter_values(dataset) == (1, 0x02, 320, 256, 0, 3, KERNELS[kernel], expected_decompositions)
        decoded = dataset[...]
        _, chunk = dataset.id.read_direct_chunk((0, 0, 0))
    if kernel == "irv97":
        np.testing.assert_allclose(decoded[0], ojph.imread_from_memory(chunk), rtol=0, atol=1)
        assert np.abs(decoded.astype(np.int64) - images).max() < 64
    else:
        np.testing.assert_array_equal(decoded, images)


@pytest.mark.parametrize("kernel", ["irv97", "rev53", "rev13", "rev12"])
def test_read_chunks_written_directly(tmp_path, kernel):
    gray_images = random_samples(np.uint16, (3, 64, 80), seed=1) >> 4
    color_images = random_samples(np.uint8, (2, 48, 64, 3), seed=2)
    encoder_settings = {"wavelet": kernel, "num_decompositions": 2, "tlm_marker": True, "tileparts_at_resolutions": True}
    if kernel == "irv97":
        encoder_settings["qstep"] = 0.01
    expected_gray = []
    expected_color = []
    with h5py.File(tmp_path / "data.h5", "w") as file:
        gray = file.create_dataset(
            "gray", shape=gray_images.shape, dtype=gray_images.dtype, chunks=(1, 64, 80),
            compression=FILTER_ID, compression_opts=(1, 0x02, 80, 64, 0, 3, KERNELS[kernel], 2),
        )
        interleaved = file.create_dataset(
            "interleaved", shape=color_images.shape, dtype=color_images.dtype, chunks=(1, 48, 64, 3),
            compression=FILTER_ID, compression_opts=(1, 0x01, 64, 48, 3, 3, KERNELS[kernel], 2),
        )
        planar = file.create_dataset(
            "planar", shape=(2, 3, 48, 64), dtype=color_images.dtype, chunks=(1, 3, 48, 64),
            compression=FILTER_ID, compression_opts=(1, 0x01, 64, 48, PLANAR | 3, 3, KERNELS[kernel], 2),
        )
        for index, image in enumerate(gray_images):
            codestream = bytes(ojph.imwrite_to_memory(image, **encoder_settings))
            gray.id.write_direct_chunk((index, 0, 0), codestream)
            expected_gray.append(ojph.imread_from_memory(codestream))
        for index, image in enumerate(color_images):
            codestream = bytes(ojph.imwrite_to_memory(image, **encoder_settings))
            interleaved.id.write_direct_chunk((index, 0, 0, 0), codestream)
            planar.id.write_direct_chunk((index, 0, 0, 0), codestream)
            expected_color.append(ojph.imread_from_memory(codestream))
    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert stored_filter_values(file["planar"]) == (1, 0x01, 64, 48, PLANAR | 3, 3, KERNELS[kernel], 2)
        tolerance = 1 if kernel == "irv97" else 0
        np.testing.assert_allclose(file["gray"][...], np.stack(expected_gray), rtol=0, atol=tolerance)
        np.testing.assert_allclose(file["interleaved"][...], np.stack(expected_color), rtol=0, atol=tolerance)
        np.testing.assert_allclose(file["planar"][...], np.moveaxis(np.stack(expected_color), -1, 1), rtol=0, atol=tolerance)
    if kernel != "irv97":
        np.testing.assert_array_equal(np.stack(expected_gray), gray_images)
        np.testing.assert_array_equal(np.stack(expected_color), color_images)


PROGRESSIVE = {"wavelet": "rev12", "num_decompositions": 2, "tlm_marker": True, "tileparts_at_resolutions": True}


@pytest.mark.parametrize(
    "dtype, filter_values, encoder_settings",
    [
        ("u1", None, None),
        ("i2", None, None),
        ("u2", None, None),
        ("u4", None, None),
        ("u2", (1, 0x0002), None),
        ("i4", (1, 0x0084), None),
        ("u2", (1, 0x0002, 320, 256, 0), None),
        ("u2", (1, 0x0002, 320, 256, 0, 3, 3, 2), PROGRESSIVE),
        ("u2", (1, 0x0002, 320, 256, 0, 1, 3, 2), dict(PROGRESSIVE, tileparts_at_resolutions=False)),
        ("u2", (1, 0x0002, 320, 256, 0, 2, 3, 2), dict(PROGRESSIVE, tlm_marker=False)),
        ("u2", (1, 0x0002, 320, 256, 0, 0, 3, 2), dict(PROGRESSIVE, tlm_marker=False, tileparts_at_resolutions=False)),
    ],
)
def test_read_chunks_from_a_writer_without_the_plugin(tmp_path, dtype, filter_values, encoder_settings):
    images = random_samples(np.dtype(dtype), (2, 256, 320), seed=3)
    write_without_plugin(tmp_path, images, (1, 256, 320), filter_values, encoder_settings=encoder_settings)
    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert stored_filter_values(file["data"]) == (filter_values or ())
        np.testing.assert_array_equal(file["data"][...], images)


@pytest.mark.parametrize(
    "filter_values, skipped_filters",
    [
        ((1, 0x0002), 1),
        ((1, 0x0002, 320, 256, 0), 0),
        ((1, 0x0002, 320, 256, 0, 3, 3, 2), 0),
        ((1, 0x0002, 300, 256, 0), 1),
        ((1, 0x0002, 320, 256, PLANAR), 1),
        ((1, 0x0002, 320, 256, 0, 3, 5, 2), 1),
    ],
)
def test_write_into_a_dataset_created_without_the_plugin(tmp_path, filter_values, skipped_filters):
    images = random_samples(np.uint16, (1, 256, 320), seed=8)
    write_without_plugin(tmp_path, np.zeros_like(images), (1, 256, 320), filter_values, encoder_settings=PROGRESSIVE)
    with h5py.File(tmp_path / "data.h5", "a") as file:
        file["data"][...] = images
    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert file["data"].id.read_direct_chunk((0, 0, 0))[0] == skipped_filters
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
        ("u2", (1, 20, 16, 1), (1, 0x02, 16, 16, 0), "does not match the chunk's image height"),
        ("u2", (1, 16, 16, 3), (1, 0x02, 16, 16, 0), "more than two chunk axes are larger than 1"),
        ("u2", (1, 16, 16, 3), (1, 0x02, 16, 16, 4), "the last chunk axis is not the component count"),
        ("u2", (1, 3, 16, 16), (1, 0x02, 16, 16, PLANAR | 4), "is not the planar component count"),
        ("u2", (1, 3, 16, 16), (1, 0x02, 16, 16, PLANAR), "the planar flag needs a component count"),
        ("u2", (1, 16, 16, 1), (1, 0x02, 16, 16, 20000), "at most 16384 components"),
        ("u2", (1, 16, 16, 1), (1, 0x02, 16, 16, 0, 4), "cd_values[5] has reserved bits set"),
        ("u2", (1, 16, 16, 1), (1, 0x02, 16, 16, 0, 3, 4), "cd_values[6] must be 0 (irv97), 1 (rev53)"),
        ("u2", (1, 16, 16, 1), (1, 0x02, 16, 16, 0, 3, 1, 33), "must be at most 32"),
        ("u2", (1, 16, 16, 1), (1, 0x02, 16, 16, 0, 3, 1, 2, 0), "version 1 has 8 cd_values, 9 were given"),
    ],
)
def test_dataset_creation_errors(tmp_path, dtype, chunks, compression_opts, message):
    with h5py.File(tmp_path / "data.h5", "w") as file:
        with pytest.raises(Exception, match="jpeg2000 filter") as error:
            file.create_dataset(
                "data", shape=chunks, dtype=dtype, chunks=chunks, compression=FILTER_ID, compression_opts=compression_opts
            )
        assert message in str(error.value)


def damaged_codestream(damage):
    gray = bytes(ojph.imwrite_to_memory(random_samples(np.uint16, (64, 64), seed=5)))
    if damage == "garbage":
        return b"\x12" * 200, (1, 64, 64), 0
    if damage == "truncated in the header":
        return gray[:100], (1, 64, 64), 0
    if damage == "truncated in the data":
        return gray[: len(gray) // 2], (1, 64, 64), 0
    if damage == "missing its end marker":
        return gray[:-2], (1, 64, 64), 0
    color = bytearray(ojph.imwrite_to_memory(random_samples(np.uint8, (64, 64, 3), seed=5)))
    first_component_size = 42
    if damage == "subsampled":
        color[first_component_size + 4] = 2
    elif damage == "mixed bit depths":
        color[first_component_size + 3] = 6
    elif damage == "38-bit samples":
        color[first_component_size : first_component_size + 9 : 3] = bytes([37, 37, 37])
    return bytes(color), (1, 64, 64, 3), 3


@pytest.mark.parametrize(
    "damage, message",
    [
        ("garbage", "cannot decode chunk"),
        ("truncated in the header", "File ended before finding a tile"),
        ("truncated in the data", "codestream problem"),
        ("missing its end marker", "codestream problem"),
        ("subsampled", "component 1 is subsampled"),
        ("mixed bit depths", "component 1 has a different bit depth or signedness"),
        ("38-bit samples", "the codestream's bit depth is 38"),
    ],
)
def test_corrupt_codestream_error(tmp_path, damage, message):
    chunk, shape, components = damaged_codestream(damage)
    with h5py.File(tmp_path / "data.h5", "w") as file:
        dataset = file.create_dataset(
            "data", shape=shape, dtype="u4", chunks=shape, compression=FILTER_ID,
            compression_opts=(1, 0x04, 64, 64, components, 0, 1, 5),
        )
        dataset.id.write_direct_chunk((0,) * len(shape), chunk)
    with h5py.File(tmp_path / "data.h5", "r") as file:
        with pytest.raises(OSError, match="cannot decode chunk") as error:
            file["data"][...]
        assert message in str(error.value)


@pytest.mark.parametrize(
    "filter_values, encoder_settings, message",
    [
        ((1, 0x0002, 320), None, "version 1 has 0, 2, 5 or 8 cd_values, the dataset has 3"),
        ((1, 0x0002, 320, 256, 0, 0), None, "version 1 has 0, 2, 5 or 8 cd_values, the dataset has 6"),
        ((1, 0x0002, 320, 256, 0, 3, 3, 2, 0), PROGRESSIVE, "the dataset has 9"),
        ((2, 0x0002), None, "this filter reads version 1"),
        ((1, 0x0002, 300, 256, 0), None, "the codestream is 320 wide and 256 high"),
        ((1, 0x0002, 320, 200, 0), None, "the codestream is 320 wide and 256 high"),
        ((1, 0x0002, 320, 256, PLANAR), None, "the planar flag needs a component count"),
        ((1, 0x0002, 320, 256, 3), None, "the codestream has 1 components, but the dataset stores 3"),
        ((1, 0x0001), None, "does not fit the dataset's 1-byte elements"),
        ((1, 0x0008), None, "element size 8 is not 1, 2 or 4 bytes"),
        ((1, 0x0202), None, "has unknown bits set"),
        ((1, 0x0002, 320, 256, 0, 3, 1, 2), PROGRESSIVE, "uses the rev12 kernel, but cd_values[6] says rev53"),
        ((1, 0x0002, 320, 256, 0, 3, 3, 4), PROGRESSIVE, "has 2 decompositions, but cd_values[7] says 4"),
        ((1, 0x0002, 320, 256, 0, 3, 5, 2), PROGRESSIVE, "cd_values[6] must be 0 (irv97)"),
        ((1, 0x0002, 320, 256, 0, 3, 3, 40), PROGRESSIVE, "must be at most 32"),
        ((1, 0x0002, 320, 256, 0, 7, 3, 2), PROGRESSIVE, "cd_values[5] has reserved bits set"),
        ((1, 0x0002, 320, 256, 0, 1, 3, 2), dict(PROGRESSIVE, tlm_marker=False), "promises a TLM marker"),
        (
            (1, 0x0002, 320, 256, 0, 2, 3, 2),
            dict(PROGRESSIVE, tileparts_at_resolutions=False),
            "promises one tile-part per resolution (3), but the codestream has 1",
        ),
        (
            (1, 0x0002, 320, 256, 0, 2, 3, 2),
            dict(PROGRESSIVE, progression_order="LRCP"),
            "progression order is LRCP",
        ),
    ],
)
def test_filter_values_that_do_not_match_the_chunks(tmp_path, filter_values, encoder_settings, message):
    dtype = "u1" if filter_values == (1, 0x0001) else "u2"
    images = random_samples(np.uint16, (1, 256, 320), seed=6)
    write_without_plugin(tmp_path, images, (1, 256, 320), filter_values, dtype, encoder_settings)
    with h5py.File(tmp_path / "data.h5", "r") as file:
        with pytest.raises(OSError, match="cannot decode chunk") as error:
            file["data"][...]
        assert message in str(error.value)


def test_values_set_local_writes_for_an_interleaved_image(tmp_path):
    images = random_samples(np.uint8, (2, 480, 640, 3), seed=9)
    with h5py.File(tmp_path / "data.h5", "w") as file:
        file.create_dataset("data", data=images, chunks=(1, 480, 640, 3), compression=FILTER_ID)
    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert stored_filter_values(file["data"]) == (1, 0x01, 640, 480, 3, 3, 1, 3)
        np.testing.assert_array_equal(file["data"][...], images)
