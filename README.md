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
from the codestream (the element type and where the components are), the
image shape the filter's encoder needs, and what a reader can rely on to plan
partial reads without opening a chunk.

Version 1:

| Index   | Meaning | Values |
|---------|---------|--------|
| `cd[0]` | Format version | `1` |
| `cd[1]` | Element type: `size_in_bytes`, `\| 0x80` if signed, `\| 0x100` if big-endian | see [element types](#element-types) |
| `cd[2]` | Width of the image a chunk holds | samples |
| `cd[3]` | Height of the image a chunk holds | samples |
| `cd[4]` | Components: the low 31 bits are the component count; bit 31 (`0x80000000`) set means planar | see [components](#components) |
| `cd[5]` | Read flags, a promise about every chunk in the dataset | bit 0: each codestream has a TLM marker; bit 1: tile-parts are split at resolutions, in RLCP order; `0` promises nothing; higher bits are reserved and must be 0 |
| `cd[6]` | Wavelet kernel, with OpenJPH's numbers | `0` = 9/7 irreversible (`irv97`), `1` = 5/3 reversible (`rev53`), `2` = `rev13`, `3` = `rev12` |
| `cd[7]` | Number of decompositions (resolution levels minus one) | `0` to `32` |

A list of `cd_values` has 0, 2, 5 or 8 entries; any other length fails.

* **0** (no `cd_values`, an older writer): the element size is inferred from
  the codestream's bit depth, rounded up to 1, 2 or 4 bytes, the signedness
  comes from the codestream, and the samples are little-endian with one
  component and no component axis.
* **2** (`cd[0..1]`): the given element type, one component and no
  component axis.
* **5** (`cd[0..4]`): adds the image shape and the components.
* **8** (`cd[0..7]`): adds what a reader needs to plan partial reads.

Lists of 2 and 5 values match the h5z-htj2k filter and promise nothing about
reading in part.

For example, a 256x320 uint16 image encoded `rev12` with 2 decompositions, a
TLM marker and one tile-part per resolution stores
`(1, 2, 320, 256, 0, 3, 3, 2)`. A `(T, 480, 640, 3)` uint8 dataset chunked
`(1, 480, 640, 3)` and written by the filter's encoder stores
`(1, 1, 640, 480, 3, 3, 1, 3)`.

#### Element types

| Type   | Little-endian | Big-endian | | Type  | Little-endian | Big-endian |
|--------|---------------|------------|-|-------|---------------|------------|
| uint8  | `0x01` | | | int8  | `0x81` | |
| uint16 | `0x02` | `0x102` | | int16 | `0x82` | `0x182` |
| uint32 | `0x04` | `0x104` | | int32 | `0x84` | `0x184` |

One-byte types never set `0x100`. Any other bit, or a size other than 1, 2
or 4, fails.

#### Components

For a codestream with `C` components of `H` rows and `W` columns:

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

#### Read flags, kernel and decompositions

`cd[5]`, `cd[6]` and `cd[7]` let a reader plan a partial or reduced-resolution
read of any chunk, and know how many levels there are and whether they are
exact, without opening a chunk. The filter itself does not need them to
decode: it always decodes whole chunks.

* `cd[6]` uses OpenJPH's kernel numbers, the values `set_wavelet_kern` takes.
  `0` and `1` are JPEG 2000's own COD transform codes. `2` (`rev13`) and `3`
  (`rev12`) are the reversible predict-only kernels, signalled with an ATK
  marker; their reduced resolutions equal `image[::2**r, ::2**r]` exactly.
  Other kernels' reduced resolutions are filtered images.
* `cd[7]` is the number of decompositions, so a chunk holds `cd[7] + 1`
  resolutions.
* `cd[5]` bit 0, the TLM marker, is what makes fast partial reads possible:
  it lists the length of every tile-part in the main header. With bit 1 also
  set, tile-part `r` of each tile carries resolution `r`, smallest first.

A reader that wants resolution `r` of a chunk, counting from the smallest:

1. reads the first bytes of the chunk, up to the first SOT marker (the main
   header), with a direct chunk read limited to a byte range (HDF5's
   sub-chunk reads) or a whole direct chunk read;
2. takes the tile-part lengths from the TLM marker;
3. reads only the prefix of the chunk that ends with tile-part `r`;
4. decodes that prefix with a JPEG 2000 library, skipping
   `cd[7] - r` resolutions.

#### Readers

The filter, and any other reader:

* needs only `cd[0]`, `cd[1]` and `cd[4]` to decode; the image shape comes
  from the codestream;
* checks `cd[2]` and `cd[3]` against the codestream when they are present,
  and fails on a mismatch;
* checks `cd[6]` and `cd[7]` against the codestream when they are present
  (the COD transform and decomposition count, and the ATK marker for `rev13`
  and `rev12`), and fails on a mismatch;
* fails when `cd[5]` bit 0 is set and the codestream has no TLM marker, or
  bit 1 is set and the codestream is not in RLCP order with one tile-part
  per resolution: a broken promise would make other readers read the wrong
  bytes;
* rejects any version but 1;
* clamps decoded samples to the range of both the codestream's bit depth and
  the element type, which matters only for lossy codestreams;
* rejects a codestream whose bit depth does not fit the element type, whose
  component count does not match `cd[4]`, whose components are subsampled or
  differ in bit depth or signedness, or that OpenJPH reports as damaged (for
  example, one that ends early).

#### Compared with h5z-htj2k

The `cd_values` use the same order and the same element-type coding as the
h5z-htj2k filter, so files that fit both read the same way. This filter also
allows 4-byte types (`0x04`, `0x84`, `0x104`, `0x184`), `cd[4]` adds `0` for
no component axis and the `0x80000000` flag for planar, and `cd[5..7]` are
added for partial reads.

### The filter's encoder

The filter's encoder writes each chunk with these settings:

| Setting | Value |
|---------|-------|
| Wavelet kernel | `cd[6]`; `rev53` (lossless) unless the user passes another |
| Decompositions | `cd[7]`; `max(1, min(31, floor(log2(max(H, W)))) - 6)` unless the user passes another |
| Quantisation for `irv97` | OpenJPH's default step size (the filter has no `qstep` value) |
| Progression order | RLCP (resolution first) |
| TLM marker | on |
| Tile-parts | one per resolution |
| Tiles | one tile, the whole image |
| Code blocks | 64 x 64 |
| Multi-component transform | none |
| Bit depth | the element size in bits, signedness from the type |

It needs 5 or 8 `cd_values`. With 5 it uses `rev53` and the default number
of decompositions. With any other number, or values that do not describe the
chunk, encoding fails; HDF5 then stores the chunk unfiltered when the filter
is optional (h5py's default) and fails the write when it is mandatory.

Other encoder settings (quantisation, code-block size, rate control) are not
in the `cd_values`. Writers that need them encode the codestreams themselves
and store them with direct chunk writes (`H5Dwrite_chunk`, or
`dataset.id.write_direct_chunk` in h5py), with `cd[5..7]` describing what they
wrote. They may record their settings in a dataset attribute; that is
outside the filter.

### `set_local`

When a dataset is created, `set_local` fills all eight `cd_values` from the
dataset's type and chunk dimensions:

* Element type: integer types of 1, 2 or 4 bytes. Other types (floating
  point, 8-byte integers, compound types) fail with an error.
* Width and height: the two chunk axes larger than 1 besides the component
  axis (1 when there are fewer).
* Components: `0` when at most two chunk axes are larger than 1. When three
  are larger than 1, the last one must be the component axis and `cd[4]` is
  its length; otherwise dataset creation fails. The planar flag is used only
  when the user passes it.
* `cd[5] = 3`, since the filter's encoder writes a TLM marker and one
  tile-part per resolution; `cd[6] = 1` (`rev53`); `cd[7]` from the default
  formula.
* Values the user passed are kept when they are consistent with the dataset
  (version 1, the dataset's element type, the chunk's image width and height,
  a component axis that fits the chunk, and valid read flags, kernel and
  decomposition count); inconsistent values fail with an error.

### What the filter cannot do

A filter cannot read part of a chunk or a reduced resolution: HDF5 always
hands a filter the whole stored chunk and expects the whole decoded chunk
back. Readers that want one resolution level, or a region, use direct chunk
reads as described under
[read flags, kernel and decompositions](#read-flags-kernel-and-decompositions).

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
elsewhere. Linking `libojph` statically (`-DBUILD_SHARED_LIBS=OFF` when
building it) gives a plugin that depends only on HDF5 and the C++ runtime.

On x86-64, `libojph` has Google Highway SIMD kernels (decoding, the wavelet
transforms, the colour transforms and the HT block encoder), compiled for
SSE4 to AVX-512 and chosen at run time. The CI and release builds for Linux
x86-64 and Windows x86-64 include them: Highway is built as a static,
position independent library with `libojph`'s `tools/build_hwy.sh` and
linked in with `-DOJPH_REQUIRE_HWY=ON`, so the plugin has no run time
dependency on a Highway library (CI checks this). Linux aarch64 and macOS
arm64 builds use OpenJPH's other code paths; the Highway kernels are
x86-only.

To build `libojph`, check out its sources with their submodules and install
them with CMake. The [build action](.github/actions/build-plugin/action.yml)
names the repository and branch, and shows the full sequence.

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

To store codestreams encoded with other settings, create the dataset with
`cd_values` that describe them and write each chunk directly:

```python
import ojph

with h5py.File("images.h5", "w") as file:
    dataset = file.create_dataset(
        "images", shape=images.shape, dtype=images.dtype, chunks=(1, 512, 640),
        compression=32031, compression_opts=(1, 2, 640, 512, 0, 3, 3, 2),
    )
    for index, image in enumerate(images):
        codestream = ojph.imwrite_to_memory(
            image, wavelet="rev12", num_decompositions=2, tlm_marker=True, tileparts_at_resolutions=True
        )
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
