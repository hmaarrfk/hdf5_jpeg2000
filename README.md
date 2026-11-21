# hdf5_jpeg2000

An HDF5 filter plugin that stores each chunk of a dataset as a JPEG 2000
codestream. It is the reference implementation of the registered HDF5 filter
**32031** (JPEG 2000, including HTJ2K, ISO/IEC 15444-15).

Once the plugin is on HDF5's plugin path, HDF5 decodes these datasets itself:
`H5Dread`, h5py indexing, and reads through virtual datasets whose sources use
the filter all return the decoded samples.

The codec is [OpenJPH](https://github.com/aous72/OpenJPH), through the
`libojph` build of its fork with the reversible predict-only kernels.

## The format

### Chunks

* **Each chunk is one bare JPEG 2000 codestream**: a `.j2c` stream that
  starts with the SOC and SIZ markers (`FF 4F FF 51`), with no JP2 boxes
  around it.
* The codestream describes itself: wavelet kernel, number of resolution
  levels, reversibility, bit depth, signedness, number of components, and
  which part of the standard it uses. A decoder reads all of these from the
  codestream, not from the dataset.
* **One image per chunk.** Every axis of the chunk but the image's height,
  width and (optionally) component axis has size 1. For example a
  `(T, Z, Y, X)` dataset is chunked `(1, 1, Y, X)` or `(1, 1, tile_y, tile_x)`,
  and a `(T, Y, X, C)` colour dataset is chunked `(1, Y, X, C)`.
* Every component of a codestream has the full image size (no subsampling)
  and the same bit depth and signedness.
* Edge chunks are whole chunks: HDF5 pads them to the chunk shape before it
  calls the filter, so their codestreams have the chunk's full height and
  width.

### `cd_values`

A filter callback sees the chunk's bytes and the `cd_values`, never the
dataset's type or shape. The `cd_values` record what a decoder cannot read
from the codestream (the element type and where the components are) and the
image shape the filter's encoder needs.

Version 1:

| Index   | Meaning | Values |
|---------|---------|--------|
| `cd[0]` | Format version | `1` |
| `cd[1]` | Element type: `size_in_bytes`, `\| 0x80` if signed, `\| 0x100` if big-endian | see below |
| `cd[2]` | Width of the image a chunk holds | samples |
| `cd[3]` | Height of the image a chunk holds | samples |
| `cd[4]` | Components: the low 31 bits are the component count; bit 31 (`0x80000000`) set means planar | see below |

Element types:

| Type   | Little-endian | Big-endian | | Type  | Little-endian | Big-endian |
|--------|---------------|------------|-|-------|---------------|------------|
| uint8  | `0x01` | | | int8  | `0x81` | |
| uint16 | `0x02` | `0x102` | | int16 | `0x82` | `0x182` |
| uint32 | `0x04` | `0x104` | | int32 | `0x84` | `0x184` |

One-byte types never set `0x100`.

Components, for a codestream with `C` components of `H` rows and `W` columns:

| `cd[4]` | Where the components are | Sample `(row, column, component)` is element |
|---------|--------------------------|----------------------------------------------|
| `0` | No component axis, one component | `row * W + column` |
| `C` (`1` to `16384`) | On the last axis, pixel-interleaved (`H W C`) | `(row * W + column) * C + component` |
| `0x80000000 \| C` | On the first axis that is not 1, planar (`C H W`) | `(component * H + row) * W + column` |

`0x80000000` with a count of 0 is invalid. h5dump and h5py print `cd[4]` as
an unsigned integer: three planar components, `0x80000003`, appear as
`2147483651`.

`0` and `1` are the same bytes: `H * W` samples in C order, with no component
axis or a trailing axis of length 1. `0x80000001`, a leading axis of length 1,
is also the same bytes.

For example, a `(T, 256, 320)` uint16 dataset chunked `(1, 256, 320)` stores
`(1, 2, 320, 256, 0)`, and a `(T, 480, 640, 3)` uint8 dataset chunked
`(1, 480, 640, 3)` stores `(1, 1, 640, 480, 3)`.

Readers:

* need only `cd[0]`, `cd[1]` and `cd[4]`; the image shape comes from the
  codestream. When `cd[2]` and `cd[3]` are present they are checked against
  the codestream, and a mismatch fails;
* accept `cd_values` that stop after `cd[1]` and treat them as one component
  with no component axis;
* when the `cd_values` are missing (an older writer), infer the element size
  from the codestream's bit depth, rounded up to 1, 2 or 4 bytes, take the
  signedness from the codestream, and assume little-endian samples and one
  component with no component axis;
* reject any other number of `cd_values` and any version but 1;
* clamp decoded samples to the range of both the codestream's bit depth and
  the element type, which matters only for lossy codestreams;
* reject a codestream whose bit depth does not fit the element type, or whose
  component count does not match `cd[4]`.

#### Compared with h5z-htj2k

The `cd_values` use the same order and the same element-type coding as the
h5z-htj2k filter, so files that fit both read the same way. This filter also
allows 4-byte types (`0x04`, `0x84`, `0x104`, `0x184`), and `cd[4]` adds `0`
for no component axis and the `0x80000000` flag for planar.

### Encoder settings

Encoder settings are **not** in the `cd_values`. JPEG 2000 has far more
options (kernel, quantisation, number of decompositions, code-block size,
progression order, rate control) than fit in a list of integers, and every
codec spells them differently. The codestream records what a decoder needs.

Applications that need specific settings encode the codestreams themselves
and store them with direct chunk writes (`H5Dwrite_chunk`, or
`dataset.id.write_direct_chunk` in h5py). They may record their settings in a
dataset attribute; that is outside the filter.

The filter's own encoder is lossless, with fixed settings:

| Setting | Value |
|---------|-------|
| Wavelet kernel | reversible 5/3 (`rev53`) |
| Decompositions | `max(1, min(31, floor(log2(max(H, W)))) - 6)` |
| Progression order | RLCP (resolution first) |
| TLM marker | on |
| Tile-parts | one per resolution |
| Tiles | one tile, the whole image |
| Code blocks | 64 x 64 |
| Multi-component transform | none |
| Bit depth | the element size in bits, signedness from the type |

The resolution-first progression, the TLM marker and one tile-part per
resolution let a reader find the bytes of each resolution level, which is
what makes resolution-progressive reads of a chunk possible.

### `set_local`

When a dataset is created, `set_local` fills all five `cd_values` from the
dataset's type and chunk dimensions:

* Element type: integer types of 1, 2 or 4 bytes. Other types (floating
  point, 8-byte integers, compound types) fail with an error.
* Width and height: the two chunk axes larger than 1 besides the component
  axis (1 when there are fewer).
* Components: `0` when at most two chunk axes are larger than 1. When three
  are larger than 1, the last one must be the component axis and `cd[4]` is
  its length; otherwise dataset creation fails. The planar flag is used only
  when the user passes it.
* Values the user passed are kept when they are consistent with the dataset
  (version 1, the dataset's element type, the chunk's image width and height,
  and a component axis that fits the chunk); inconsistent values fail with an
  error.

### Partial and reduced-resolution reads

A filter cannot read part of a chunk or a reduced resolution: HDF5 always
hands a filter the whole stored chunk and expects the whole decoded chunk
back. Readers that want one resolution level, or a region, read the
codestream with a direct chunk read (`H5Dread_chunk`, or
`dataset.id.read_direct_chunk` in h5py) and decode it with a JPEG 2000
library.

## Building

Requirements: a C++17 compiler, CMake 3.20 or later, HDF5 2.x, and the
`libojph` library (`find_package(ojph CONFIG)` providing `ojph::ojph`).

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/prefix
cmake --build build --config Release
cmake --install build --config Release --prefix /path/to/prefix
```

The plugin is one shared library: `libh5jpeg2000.so` on Linux,
`libh5jpeg2000.dylib` on macOS and `h5jpeg2000.dll` on Windows. It installs to
`<prefix>/lib/plugin`, the directory HDF5 searches by default when it is
installed in the same prefix. Set `HDF5_PLUGIN_INSTALL_DIR` to install it
elsewhere. Linking `libojph` statically (`-DBUILD_SHARED_LIBS=OFF
-DOJPH_ENABLE_HWY=OFF` when building it) gives a plugin that depends only on
HDF5 and the C++ runtime.

To build `libojph`, check out
[`ramonaoptics/ojph`](https://github.com/ramonaoptics/ojph) branch `libojph`
with its submodules and install it with CMake. The
[CI workflow](.github/workflows/ci.yml) shows the full sequence.

## Using it

Point HDF5 at the directory that holds the plugin, unless it is already in
HDF5's default plugin directory:

```
export HDF5_PLUGIN_PATH=/path/to/prefix/lib/plugin
```

Then, from h5py:

```python
import h5py
import numpy as np

images = np.random.randint(0, 4096, size=(10, 512, 640), dtype=np.uint16)
with h5py.File("images.h5", "w") as file:
    file.create_dataset("images", data=images, chunks=(1, 512, 640), compression=32031)
with h5py.File("images.h5", "r") as file:
    assert (file["images"][...] == images).all()
```

From C, `H5Pset_filter(dcpl, 32031, H5Z_FLAG_MANDATORY, 0, NULL)` on a chunked
dataset creation property list.

To store codestreams encoded with other settings, create the dataset with the
filter and write each chunk directly:

```python
import ojph

with h5py.File("images.h5", "w") as file:
    dataset = file.create_dataset(
        "images", shape=images.shape, dtype=images.dtype, chunks=(1, 512, 640), compression=32031
    )
    for index, image in enumerate(images):
        codestream = ojph.imwrite_to_memory(image, wavelet="rev12", tlm_marker=True, tileparts_at_resolutions=True)
        dataset.id.write_direct_chunk((index, 0, 0), bytes(codestream))
```

## Testing

```
HDF5_PLUGIN_PATH=/path/to/prefix/lib/plugin python -m pytest tests
```

The tests need h5py and the `ojph` Python package.

## Learn more

* https://github.com/chafey/HTJ2KResources
* https://github.com/aous72/OpenJPH
* https://www.openjpeg.org/

## License

BSD 3-Clause; see [LICENSE.txt](LICENSE.txt). The plugin links libojph, which is
built from a fork of [OpenJPH](https://github.com/aous72/OpenJPH) (BSD 2-Clause).
