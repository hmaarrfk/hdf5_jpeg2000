# HDF5 JPEG2000 Filter

An HDF5 filter plugin for compressing and decompressing data using JPEG2000 codestreams.

## Overview

This project provides an HDF5 filter that enables JPEG2000 compression for chunked datasets in HDF5 files. JPEG2000 is a sophisticated image compression standard that supports both lossless and lossy compression modes, making it particularly well-suited for scientific and medical imaging applications where high-quality compression is essential.

## What is HDF5?

HDF5 (Hierarchical Data Format version 5) is a versatile data model, library, and file format for storing and managing large and complex datasets. It is widely used in scientific computing, engineering, and data-intensive applications.

## What are HDF5 Filters?

HDF5 filters are plugin mechanisms that allow custom processing of data chunks during read and write operations. Filters can provide:
- **Compression** (gzip, SZIP, JPEG2000, etc.)
- **Encryption**
- **Checksum verification**
- **Custom data transformations**

Filters operate on chunked datasets, processing each chunk independently as data flows to/from storage.

## What is JPEG2000?

JPEG2000 is an advanced image compression standard (ISO/IEC 15444) that offers:
- **Lossless and lossy compression** - Choose between perfect reconstruction or higher compression ratios
- **Progressive decoding** - Access lower resolution versions without full decompression
- **Region of interest** - Specify areas that should receive higher quality encoding
- **Error resilience** - Better handling of transmission errors compared to JPEG
- **Flexible compression** - Ideal for medical imaging, satellite imagery, and scientific data

## Features

- **Efficient compression** - Achieve significant storage savings for image-like datasets
- **Flexible encoding** - Support for both lossless and lossy compression modes
- **HDF5 integration** - Seamless integration with the HDF5 ecosystem
- **Chunked processing** - Efficient handling of large datasets through chunk-based compression
- **Cross-platform** - Works across different operating systems and architectures

## Requirements

- HDF5 library (version 1.8.0 or later)
- JPEG2000 codec library (e.g., OpenJPEG, Kakadu, or Grok)
- C/C++ compiler with support for the platform
- CMake (for building from source)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/hmaarrfk/hdf5_jpeg2000.git
cd hdf5_jpeg2000

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make

# Install (may require sudo)
make install
```

### Using Package Managers

Instructions for conda, pip, or other package managers will be added as they become available.

## Usage

### C/C++ Example

```c
#include "hdf5.h"

// Create a chunked dataset with JPEG2000 compression
hid_t file_id = H5Fcreate("example.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
hid_t space_id = H5Screate_simple(2, dims, NULL);

// Create dataset creation property list and set chunking
hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
H5Pset_chunk(dcpl_id, 2, chunk_dims);

// Set JPEG2000 filter
unsigned int cd_values[2] = {0, 5}; // Example: lossless mode, quality factor 5
H5Pset_filter(dcpl_id, JPEG2000_FILTER_ID, H5Z_FLAG_MANDATORY, 2, cd_values);

// Create and write dataset
hid_t dset_id = H5Dcreate(file_id, "compressed_data", H5T_NATIVE_FLOAT, 
                          space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

// Cleanup
H5Dclose(dset_id);
H5Pclose(dcpl_id);
H5Sclose(space_id);
H5Fclose(file_id);
```

### Python Example (with h5py)

```python
import h5py
import numpy as np

# Ensure the JPEG2000 filter plugin is loaded
import hdf5plugin

# Create sample data
data = np.random.rand(1000, 1000).astype(np.float32)

# Create HDF5 file with JPEG2000 compression
with h5py.File('example.h5', 'w') as f:
    # Create dataset with JPEG2000 compression
    # Note: Actual filter parameters depend on implementation
    f.create_dataset('compressed_data', 
                     data=data,
                     chunks=(100, 100),
                     compression=JPEG2000_FILTER_ID,
                     compression_opts=(0, 5))  # lossless, quality 5
```

## Filter Parameters

The JPEG2000 filter accepts the following parameters:

- **Compression mode** - Lossless (0) or Lossy (1)
- **Quality factor** - Controls compression ratio in lossy mode (typically 1-100)
- **Code block size** - Size of code blocks for encoding
- **Progression order** - Order in which the compressed data is organized
- **Number of decomposition levels** - Wavelet decomposition depth

Detailed parameter documentation is available in the [wiki](https://github.com/hmaarrfk/hdf5_jpeg2000/wiki).

## Plugin Architecture

This filter can be compiled as a plugin and installed in the HDF5 plugin directory, making it available to all HDF5 applications without recompilation:

```bash
# Default plugin directories:
# Linux: /usr/local/hdf5/lib/plugin
# macOS: /usr/local/hdf5/lib/plugin
# Windows: C:\Program Files\HDF Group\HDF5\lib\plugin

# Set custom plugin directory
export HDF5_PLUGIN_PATH=/path/to/plugin/directory
```

## Performance Considerations

- **Chunk size** - Choose chunk dimensions that match your access patterns
- **Compression settings** - Balance between compression ratio and speed
- **Data characteristics** - JPEG2000 works best with image-like data (spatial correlation)
- **I/O patterns** - Consider read vs. write performance trade-offs

## Limitations

- Filters only operate on **chunked datasets**, not contiguous or compact layouts
- The plugin must be **installed on all systems** that need to read/write filtered files
- Performance depends on the underlying JPEG2000 codec library
- Some data types may be more suitable than others for JPEG2000 compression

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- The HDF5 Group for the HDF5 library and filter architecture
- OpenJPEG, Kakadu, and Grok projects for JPEG2000 codec implementations
- Contributors to the HDF5 ecosystem

## References

- [HDF5 Filters Documentation](https://portal.hdfgroup.org/documentation/hdf5/latest/_f_i_l_t_e_r.html)
- [HDF5 Plugin Architecture](https://portal.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html)
- [JPEG2000 Standard (ISO/IEC 15444)](https://jpeg.org/jpeg2000/)
- [OpenJPEG Library](https://www.openjpeg.org/)

## Support

For questions, issues, or support:
- Open an issue on [GitHub Issues](https://github.com/hmaarrfk/hdf5_jpeg2000/issues)
- Consult the [HDF5 documentation](https://portal.hdfgroup.org/documentation/index.html)
- Check the [wiki](https://github.com/hmaarrfk/hdf5_jpeg2000/wiki) for additional examples
