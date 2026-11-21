// HDF5 filter 32031: chunks stored as bare JPEG 2000 / HTJ2K codestreams.
// The format is specified in README.md.

#include <H5PLextern.h>
#include <hdf5.h>

#include <ojph/ojph_arch.h>
#include <ojph/ojph_codestream.h>
#include <ojph/ojph_file.h>
#include <ojph/ojph_mem.h>
#include <ojph/ojph_message.h>
#include <ojph/ojph_params.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr H5Z_filter_t jpeg2000_filter_id = 32031;
constexpr unsigned format_version = 1;
constexpr unsigned type_size_bits = 0xFF;
constexpr unsigned type_signed = 0x80;
constexpr unsigned type_big_endian = 0x100;
constexpr uint32_t components_planar = 0x80000000u;
constexpr uint32_t components_count_bits = 0x7FFFFFFFu;
constexpr size_t shape_value_count = 5;
constexpr size_t value_count = 8;
constexpr uint32_t read_flag_tlm = 0x1;
constexpr uint32_t read_flag_resolution_tile_parts = 0x2;
constexpr uint32_t read_flags_known = read_flag_tlm | read_flag_resolution_tile_parts;
constexpr uint32_t maximum_kernel = ojph::param_cod::OJPH_WAVELET_REV12;
constexpr uint32_t maximum_decompositions = 32;
constexpr uint64_t maximum_components = 16384;
constexpr uint64_t maximum_image_extent = 0xFFFFFFFFu;

void push_error(const char *function, unsigned line, hid_t minor, const char *format, ...)
{
    char message[512];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    H5Epush2(H5E_DEFAULT, __FILE__, function, line, H5E_ERR_CLS, H5E_PLINE, minor, "jpeg2000 filter: %s",
             message);
}

#define FILTER_ERROR(minor, ...) push_error(__func__, __LINE__, minor, __VA_ARGS__)

class codec_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const char *format, ...)
{
    char message[512];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    throw codec_error(message);
}

// OpenJPH reports errors through a process-wide handler that must throw.
// This one keeps the message instead of printing it to stderr.
class throwing_error_handler : public ojph::message_error {
  public:
    void operator()(int code, const char *file_name, int line_number, const char *format, ...) override
    {
        char detail[400];
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(detail, sizeof(detail), format, arguments);
        va_end(arguments);
        char message[512];
        std::snprintf(message, sizeof(message), "codestream error 0x%08X (%s:%d): %s", code, file_name,
                      line_number, detail);
        throw codec_error(message);
    }
};

// OpenJPH reports damaged codestreams it can still partly decode, such as one
// that ends early, as info messages. They are kept so the filter can fail.
thread_local std::string codec_report;

class recording_info_handler : public ojph::message_info {
  public:
    void operator()(int code, const char *, int, const char *format, ...) override
    {
        if (!codec_report.empty())
            return;
        char detail[400];
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(detail, sizeof(detail), format, arguments);
        va_end(arguments);
        char message[512];
        std::snprintf(message, sizeof(message), "codestream problem 0x%08X: %s", code, detail);
        codec_report = message;
    }
};

throwing_error_handler error_handler;
recording_info_handler info_handler;
std::once_flag handlers_configured;

void configure_codec()
{
    std::call_once(handlers_configured, [] {
        ojph::configure_error(&error_handler);
        ojph::configure_info(&info_handler);
    });
    codec_report.clear();
}

struct element_type {
    unsigned size;
    bool is_signed;
    bool big_endian;
};

element_type parse_element_type(unsigned code)
{
    if (code & ~(type_size_bits | type_big_endian))
        fail("cd_values[1] = 0x%X has unknown bits set", code);
    element_type type{code & type_size_bits & ~type_signed, (code & type_signed) != 0,
                      (code & type_big_endian) != 0};
    if (type.size != 1 && type.size != 2 && type.size != 4)
        fail("cd_values[1] = 0x%X: element size %u is not 1, 2 or 4 bytes", code, type.size);
    return type;
}

struct image_geometry {
    uint64_t height;
    uint64_t width;
    uint64_t components;
    bool planar;
};

struct component_word {
    uint64_t count;
    bool planar;
};

component_word parse_component_word(uint32_t word, const char *&reason)
{
    component_word components{word & components_count_bits, (word & components_planar) != 0};
    reason = nullptr;
    if (components.planar && components.count == 0)
        reason = "cd_values[4] = 0x80000000: the planar flag needs a component count";
    else if (components.count > maximum_components)
        reason = "cd_values[4]: a JPEG 2000 codestream holds at most 16384 components";
    return components;
}

bool image_geometry_from_chunk(const hsize_t *dims, int rank, uint32_t word, image_geometry &geometry,
                               const char *&reason)
{
    component_word components = parse_component_word(word, reason);
    if (reason != nullptr)
        return false;
    geometry.components = 1;
    geometry.planar = components.planar;
    int first_image_axis = 0;
    int last_image_axis = rank;
    if (components.count > 1 && !components.planar) {
        if (rank < 1 || dims[rank - 1] != components.count) {
            reason = "the last chunk axis is not the component count in cd_values[4]";
            return false;
        }
        geometry.components = components.count;
        last_image_axis = rank - 1;
    }
    else if (components.count > 1) {
        while (first_image_axis < rank && dims[first_image_axis] == 1)
            ++first_image_axis;
        if (first_image_axis == rank || dims[first_image_axis] != components.count) {
            reason = "the first chunk axis that is not 1 is not the planar component count in cd_values[4]";
            return false;
        }
        geometry.components = components.count;
        ++first_image_axis;
    }
    std::vector<uint64_t> image_dims;
    for (int axis = first_image_axis; axis < last_image_axis; ++axis)
        if (dims[axis] > 1)
            image_dims.push_back(dims[axis]);
    if (image_dims.size() > 2) {
        reason = components.count > 1
                     ? "more than two chunk axes besides the component axis are larger than 1; chunk every "
                       "other axis with size 1"
                     : "more than two chunk axes are larger than 1; chunk every axis but the image's height, "
                       "width and components with size 1, and put the components on the last axis";
        return false;
    }
    geometry.height = image_dims.size() == 2 ? image_dims[0] : 1;
    geometry.width = image_dims.empty() ? 1 : image_dims.back();
    if (geometry.height > maximum_image_extent || geometry.width > maximum_image_extent) {
        reason = "the image is larger than 2^32 - 1 samples along an axis";
        return false;
    }
    return true;
}

const char *read_values_problem(uint32_t read_flags, uint32_t kernel, uint32_t decompositions)
{
    if (read_flags & ~read_flags_known)
        return "cd_values[5] has reserved bits set; only bits 0 (TLM) and 1 (tile-parts at resolutions) "
               "are defined";
    if (kernel > maximum_kernel)
        return "cd_values[6] must be 0 (irv97), 1 (rev53), 2 (rev13) or 3 (rev12)";
    if (decompositions > maximum_decompositions)
        return "cd_values[7], the number of decompositions, must be at most 32";
    return nullptr;
}

ojph::ui32 default_decomposition_count(uint64_t height, uint64_t width)
{
    uint64_t extent = std::max(height, width);
    int log2_extent = 0;
    while ((extent >> (log2_extent + 1)) != 0)
        ++log2_extent;
    return static_cast<ojph::ui32>(std::max(1, std::min(31, log2_extent) - 6));
}

bool element_type_code(hid_t type_id, unsigned &code)
{
    if (H5Tget_class(type_id) != H5T_INTEGER) {
        FILTER_ERROR(H5E_BADTYPE, "the dataset's type must be an integer type of 1, 2 or 4 bytes");
        return false;
    }
    size_t size = H5Tget_size(type_id);
    if (size != 1 && size != 2 && size != 4) {
        FILTER_ERROR(H5E_BADTYPE, "the dataset's integer type is %zu bytes; it must be 1, 2 or 4 bytes", size);
        return false;
    }
    H5T_sign_t sign = H5Tget_sign(type_id);
    H5T_order_t order = H5Tget_order(type_id);
    if (sign < 0 || order < 0) {
        FILTER_ERROR(H5E_BADTYPE, "cannot read the sign or byte order of the dataset's type");
        return false;
    }
    code = static_cast<unsigned>(size);
    if (sign == H5T_SGN_2)
        code |= type_signed;
    if (size > 1 && order == H5T_ORDER_BE)
        code |= type_big_endian;
    else if (size > 1 && order != H5T_ORDER_LE) {
        FILTER_ERROR(H5E_BADTYPE, "the dataset's type must be little- or big-endian");
        return false;
    }
    return true;
}

herr_t set_local(hid_t dcpl_id, hid_t type_id, hid_t space_id)
{
    unsigned flags = 0;
    size_t given_count = 8;
    unsigned given[8] = {};
    if (H5Pget_filter_by_id2(dcpl_id, jpeg2000_filter_id, &flags, &given_count, given, 0, nullptr, nullptr) <
        0) {
        FILTER_ERROR(H5E_CANTGET, "cannot read the filter's cd_values");
        return -1;
    }
    unsigned code = 0;
    if (!element_type_code(type_id, code))
        return -1;
    if (given_count >= 1 && given[0] != format_version) {
        FILTER_ERROR(H5E_BADVALUE, "cd_values[0] = %u; this filter writes version %u", given[0],
                     format_version);
        return -1;
    }
    if (given_count >= 2 && given[1] != code) {
        FILTER_ERROR(H5E_BADVALUE, "cd_values[1] = 0x%X does not match the dataset's type, 0x%X", given[1],
                     code);
        return -1;
    }
    if (given_count > value_count) {
        FILTER_ERROR(H5E_BADVALUE, "version 1 has %zu cd_values, %zu were given", value_count, given_count);
        return -1;
    }

    int rank = H5Sget_simple_extent_ndims(space_id);
    if (rank < 0 || rank > H5S_MAX_RANK) {
        FILTER_ERROR(H5E_CANTGET, "cannot read the chunk dimensions");
        return -1;
    }
    hsize_t dims[H5S_MAX_RANK] = {};
    if (H5Sget_simple_extent_dims(space_id, dims, nullptr) < 0) {
        FILTER_ERROR(H5E_CANTGET, "cannot read the chunk dimensions");
        return -1;
    }

    uint32_t word = 0;
    if (given_count >= 5)
        word = given[4];
    else {
        int larger_than_one = 0;
        for (int axis = 0; axis < rank; ++axis)
            larger_than_one += dims[axis] > 1;
        if (larger_than_one == 3 && dims[rank - 1] > 1)
            word = static_cast<uint32_t>(dims[rank - 1]);
    }

    image_geometry geometry{};
    const char *reason = "";
    if (!image_geometry_from_chunk(dims, rank, word, geometry, reason)) {
        FILTER_ERROR(H5E_BADVALUE, "cannot store chunks with cd_values[4] = 0x%X: %s", word, reason);
        return -1;
    }
    if (given_count >= 3 && given[2] != geometry.width) {
        FILTER_ERROR(H5E_BADVALUE, "cd_values[2] = %u does not match the chunk's image width, %llu", given[2],
                     static_cast<unsigned long long>(geometry.width));
        return -1;
    }
    if (given_count >= 4 && given[3] != geometry.height) {
        FILTER_ERROR(H5E_BADVALUE, "cd_values[3] = %u does not match the chunk's image height, %llu", given[3],
                     static_cast<unsigned long long>(geometry.height));
        return -1;
    }

    uint32_t read_flags = given_count >= 6 ? given[5] : read_flags_known;
    uint32_t kernel = given_count >= 7 ? given[6] : static_cast<uint32_t>(ojph::param_cod::OJPH_WAVELET_REV53);
    uint32_t decompositions =
        given_count >= 8 ? given[7] : default_decomposition_count(geometry.height, geometry.width);
    if (const char *problem = read_values_problem(read_flags, kernel, decompositions)) {
        FILTER_ERROR(H5E_BADVALUE, "%s", problem);
        return -1;
    }

    unsigned values[value_count] = {format_version,
                                    code,
                                    static_cast<unsigned>(geometry.width),
                                    static_cast<unsigned>(geometry.height),
                                    word,
                                    read_flags,
                                    kernel,
                                    decompositions};
    if (H5Pmodify_filter(dcpl_id, jpeg2000_filter_id, flags, value_count, values) < 0) {
        FILTER_ERROR(H5E_CANTSET, "cannot store the filter's cd_values");
        return -1;
    }
    return 0;
}

struct hdf5_memory_deleter {
    void operator()(void *memory) const { H5free_memory(memory); }
};

using hdf5_buffer = std::unique_ptr<unsigned char, hdf5_memory_deleter>;

hdf5_buffer allocate(size_t size)
{
    hdf5_buffer buffer(static_cast<unsigned char *>(H5allocate_memory(size, false)));
    if (!buffer)
        fail("cannot allocate %zu bytes", size);
    return buffer;
}

uint32_t load_bits(const unsigned char *source, const element_type &type)
{
    uint32_t bits = 0;
    for (unsigned byte = 0; byte < type.size; ++byte) {
        unsigned shift = type.big_endian ? 8 * (type.size - 1 - byte) : 8 * byte;
        bits |= static_cast<uint32_t>(source[byte]) << shift;
    }
    return bits;
}

void store_bits(unsigned char *destination, uint32_t bits, const element_type &type)
{
    for (unsigned byte = 0; byte < type.size; ++byte) {
        unsigned shift = type.big_endian ? 8 * (type.size - 1 - byte) : 8 * byte;
        destination[byte] = static_cast<unsigned char>(bits >> shift);
    }
}

int64_t load_sample(const unsigned char *source, const element_type &type)
{
    uint32_t bits = load_bits(source, type);
    if (!type.is_signed)
        return bits;
    unsigned unused = 32 - 8 * type.size;
    return static_cast<int32_t>(bits << unused) >> unused;
}

void minimum_and_maximum(unsigned bit_depth, bool is_signed, int64_t &minimum, int64_t &maximum)
{
    if (is_signed) {
        minimum = -(int64_t{1} << (bit_depth - 1));
        maximum = (int64_t{1} << (bit_depth - 1)) - 1;
    }
    else {
        minimum = 0;
        maximum = (int64_t{1} << bit_depth) - 1;
    }
}

size_t sample_offset(uint64_t row, uint64_t column, uint64_t component, const image_geometry &geometry)
{
    if (geometry.planar)
        return static_cast<size_t>((component * geometry.height + row) * geometry.width + column);
    return static_cast<size_t>((row * geometry.width + column) * geometry.components + component);
}

uint64_t checked_product(uint64_t a, uint64_t b)
{
    if (a != 0 && b > SIZE_MAX / a)
        fail("the decoded chunk would not fit in memory");
    return a * b;
}

const char *kernel_name(uint32_t kernel)
{
    static const char *const names[] = {"irv97", "rev53", "rev13", "rev12"};
    return kernel <= maximum_kernel ? names[kernel] : "unknown";
}

uint32_t read_big_endian(const unsigned char *bytes, size_t size)
{
    uint32_t value = 0;
    for (size_t index = 0; index < size; ++index)
        value = (value << 8) | bytes[index];
    return value;
}

struct codestream_layout {
    bool has_tlm;
    uint64_t tile_part_count;
};

codestream_layout scan_codestream(const unsigned char *bytes, size_t size)
{
    constexpr uint32_t marker_tlm = 0xFF55;
    constexpr uint32_t marker_sot = 0xFF90;
    codestream_layout layout{false, 0};
    size_t position = 2;
    while (position + 4 <= size) {
        uint32_t marker = read_big_endian(bytes + position, 2);
        if (marker == marker_sot)
            break;
        layout.has_tlm = layout.has_tlm || marker == marker_tlm;
        position += 2 + read_big_endian(bytes + position + 2, 2);
    }
    while (position + 12 <= size && read_big_endian(bytes + position, 2) == marker_sot) {
        ++layout.tile_part_count;
        uint32_t tile_part_length = read_big_endian(bytes + position + 6, 4);
        if (tile_part_length == 0)
            break;
        position += tile_part_length;
    }
    return layout;
}

void check_read_values(const unsigned cd_values[], ojph::codestream &codestream, const unsigned char *bytes,
                       size_t size)
{
    if (const char *problem = read_values_problem(cd_values[5], cd_values[6], cd_values[7]))
        fail("%s", problem);
    ojph::param_cod cod = codestream.access_cod();
    uint32_t kernel = cod.get_wavelet_kern();
    if (kernel != cd_values[6])
        fail("the codestream uses the %s kernel, but cd_values[6] says %s", kernel_name(kernel),
             kernel_name(cd_values[6]));
    uint32_t decompositions = cod.get_num_decompositions();
    if (decompositions != cd_values[7])
        fail("the codestream has %u decompositions, but cd_values[7] says %u", decompositions, cd_values[7]);
    if (cd_values[5] == 0)
        return;
    codestream_layout layout = scan_codestream(bytes, size);
    if ((cd_values[5] & read_flag_tlm) && !layout.has_tlm)
        fail("cd_values[5] promises a TLM marker, but the codestream has none");
    if (cd_values[5] & read_flag_resolution_tile_parts) {
        if (std::strcmp(cod.get_progression_order_as_string(), "RLCP") != 0)
            fail("cd_values[5] promises tile-parts at resolutions in RLCP order, but the codestream's "
                 "progression order is %s",
                 cod.get_progression_order_as_string());
        ojph::param_siz siz = codestream.access_siz();
        ojph::point extent = siz.get_image_extent();
        ojph::point tile_offset = siz.get_tile_offset();
        ojph::size tile_size = siz.get_tile_size();
        uint64_t tiles_across = (extent.x - tile_offset.x + tile_size.w - 1) / tile_size.w;
        uint64_t tiles_down = (extent.y - tile_offset.y + tile_size.h - 1) / tile_size.h;
        uint64_t expected = tiles_across * tiles_down * (decompositions + 1);
        if (layout.tile_part_count != expected)
            fail("cd_values[5] promises one tile-part per resolution (%llu), but the codestream has %llu",
                 static_cast<unsigned long long>(expected),
                 static_cast<unsigned long long>(layout.tile_part_count));
    }
}

size_t decode(size_t cd_nelmts, const unsigned cd_values[], size_t nbytes, size_t *buf_size, void **buf)
{
    if (cd_nelmts != 0 && cd_nelmts != 2 && cd_nelmts != shape_value_count && cd_nelmts != value_count)
        fail("version 1 has 0, 2, 5 or 8 cd_values, the dataset has %zu", cd_nelmts);
    if (cd_nelmts != 0 && cd_values[0] != format_version)
        fail("cd_values[0] = %u; this filter reads version %u", cd_values[0], format_version);

    ojph::mem_infile file;
    file.open(static_cast<const ojph::ui8 *>(*buf), nbytes);
    ojph::codestream codestream;
    codestream.read_headers(&file);
    ojph::param_siz siz = codestream.access_siz();

    image_geometry geometry{siz.get_recon_height(0), siz.get_recon_width(0), siz.get_num_components(), false};
    unsigned bit_depth = siz.get_bit_depth(0);
    bool codestream_signed = siz.is_signed(0);
    for (ojph::ui32 component = 0; component < geometry.components; ++component) {
        ojph::point downsampling = siz.get_downsampling(component);
        if (downsampling.x != 1 || downsampling.y != 1)
            fail("component %u is subsampled; every component must have the full image size", component);
        if (siz.get_bit_depth(component) != bit_depth || siz.is_signed(component) != codestream_signed)
            fail("component %u has a different bit depth or signedness from component 0", component);
    }
    if (bit_depth < 1 || bit_depth > 32)
        fail("the codestream's bit depth is %u; this filter decodes 1 to 32 bits", bit_depth);

    element_type type{bit_depth <= 8 ? 1u : bit_depth <= 16 ? 2u : 4u, codestream_signed, false};
    uint64_t expected_components = 1;
    if (cd_nelmts >= 2)
        type = parse_element_type(cd_values[1]);
    if (cd_nelmts >= shape_value_count) {
        if (cd_values[2] != geometry.width || cd_values[3] != geometry.height)
            fail("the codestream is %llu wide and %llu high, but cd_values[2] and cd_values[3] say %u by %u",
                 static_cast<unsigned long long>(geometry.width),
                 static_cast<unsigned long long>(geometry.height), cd_values[2], cd_values[3]);
        const char *reason = nullptr;
        component_word components = parse_component_word(cd_values[4], reason);
        if (reason != nullptr)
            fail("%s", reason);
        expected_components = std::max<uint64_t>(components.count, 1);
        geometry.planar = components.planar;
    }
    if (geometry.components != expected_components)
        fail("the codestream has %llu components, but the dataset stores %llu",
             static_cast<unsigned long long>(geometry.components),
             static_cast<unsigned long long>(expected_components));
    if (bit_depth > 8 * type.size)
        fail("the codestream's bit depth, %u, does not fit the dataset's %u-byte elements", bit_depth,
             type.size);
    if (cd_nelmts == value_count)
        check_read_values(cd_values, codestream, static_cast<const unsigned char *>(*buf), nbytes);

    uint64_t sample_count = checked_product(checked_product(geometry.height, geometry.width), geometry.components);
    size_t output_size = static_cast<size_t>(checked_product(sample_count, type.size));
    hdf5_buffer output = allocate(output_size);

    int64_t codestream_minimum, codestream_maximum, type_minimum, type_maximum;
    minimum_and_maximum(bit_depth, codestream_signed, codestream_minimum, codestream_maximum);
    minimum_and_maximum(8 * type.size, type.is_signed, type_minimum, type_maximum);
    int64_t minimum = std::max(codestream_minimum, type_minimum);
    int64_t maximum = std::min(codestream_maximum, type_maximum);
    bool unsigned_32_bit = !codestream_signed && bit_depth == 32;

    codestream.create();
    std::vector<uint64_t> rows_done(geometry.components, 0);
    for (uint64_t line_index = 0; line_index < geometry.height * geometry.components; ++line_index) {
        ojph::ui32 component = 0;
        ojph::line_buf *line = codestream.pull(component);
        if (line == nullptr || component >= geometry.components || rows_done[component] >= geometry.height)
            fail("the codestream returned an unexpected line");
        if (!(line->flags & ojph::line_buf::LFT_32BIT) || !(line->flags & ojph::line_buf::LFT_INTEGER))
            fail("the codestream returned lines that are not 32-bit integers");
        uint64_t row = rows_done[component]++;
        const ojph::si32 *samples = line->i32;
        for (uint64_t column = 0; column < geometry.width; ++column) {
            int64_t value = unsigned_32_bit ? static_cast<int64_t>(static_cast<uint32_t>(samples[column]))
                                            : static_cast<int64_t>(samples[column]);
            value = std::min(std::max(value, minimum), maximum);
            size_t offset = sample_offset(row, column, component, geometry);
            store_bits(output.get() + offset * type.size, static_cast<uint32_t>(value), type);
        }
    }
    codestream.close();
    if (!codec_report.empty())
        fail("%s", codec_report.c_str());

    H5free_memory(*buf);
    *buf = output.release();
    *buf_size = output_size;
    return output_size;
}

size_t encode(size_t cd_nelmts, const unsigned cd_values[], size_t nbytes, size_t *buf_size, void **buf)
{
    if ((cd_nelmts != shape_value_count && cd_nelmts != value_count) || cd_values[0] != format_version)
        fail("encoding needs 5 or 8 version-1 cd_values, as the filter's set_local writes; the dataset has %zu",
             cd_nelmts);
    uint32_t kernel = ojph::param_cod::OJPH_WAVELET_REV53;
    uint32_t decompositions = default_decomposition_count(cd_values[3], cd_values[2]);
    if (cd_nelmts == value_count) {
        if (const char *problem = read_values_problem(cd_values[5], cd_values[6], cd_values[7]))
            fail("%s", problem);
        kernel = cd_values[6];
        decompositions = cd_values[7];
    }
    element_type type = parse_element_type(cd_values[1]);
    const char *reason = nullptr;
    component_word components = parse_component_word(cd_values[4], reason);
    if (reason != nullptr)
        fail("%s", reason);
    image_geometry geometry{cd_values[3], cd_values[2], std::max<uint64_t>(components.count, 1), components.planar};
    uint64_t expected_bytes = checked_product(
        checked_product(checked_product(geometry.height, geometry.width), geometry.components), type.size);
    if (expected_bytes == 0 || nbytes != expected_bytes)
        fail("the chunk has %zu bytes, but cd_values describe a %llu-byte image", nbytes,
             static_cast<unsigned long long>(expected_bytes));

    ojph::codestream codestream;
    ojph::param_siz siz = codestream.access_siz();
    siz.set_image_extent(ojph::point(static_cast<ojph::ui32>(geometry.width),
                                     static_cast<ojph::ui32>(geometry.height)));
    siz.set_image_offset(ojph::point(0, 0));
    siz.set_tile_size(ojph::size(static_cast<ojph::ui32>(geometry.width),
                                 static_cast<ojph::ui32>(geometry.height)));
    siz.set_tile_offset(ojph::point(0, 0));
    siz.set_num_components(static_cast<ojph::ui32>(geometry.components));
    for (ojph::ui32 component = 0; component < geometry.components; ++component)
        siz.set_component(component, ojph::point(1, 1), 8 * type.size, type.is_signed);

    ojph::param_cod cod = codestream.access_cod();
    cod.set_reversible(kernel != ojph::param_cod::OJPH_WAVELET_IRV97);
    cod.set_wavelet_kern(kernel);
    cod.set_num_decomposition(decompositions);
    cod.set_block_dims(64, 64);
    cod.set_progression_order("RLCP");
    cod.set_color_transform(false);

    codestream.set_planar(true);
    codestream.request_tlm_marker(true);
    codestream.set_tilepart_divisions(true, false);

    ojph::mem_outfile file;
    file.open();
    codestream.write_headers(&file);

    const unsigned char *input = static_cast<const unsigned char *>(*buf);
    std::vector<uint64_t> rows_done(geometry.components, 0);
    ojph::ui32 component = 0;
    ojph::line_buf *line = codestream.exchange(nullptr, component);
    for (uint64_t line_index = 0; line_index < geometry.height * geometry.components; ++line_index) {
        if (line == nullptr || component >= geometry.components || rows_done[component] >= geometry.height)
            fail("the encoder asked for an unexpected line");
        uint64_t row = rows_done[component]++;
        ojph::si32 *samples = line->i32;
        for (uint64_t column = 0; column < geometry.width; ++column) {
            size_t offset = sample_offset(row, column, component, geometry);
            samples[column] = static_cast<ojph::si32>(load_sample(input + offset * type.size, type));
        }
        line = codestream.exchange(line, component);
    }
    codestream.flush();

    size_t output_size = static_cast<size_t>(file.tell());
    hdf5_buffer output = allocate(output_size);
    std::memcpy(output.get(), file.get_data(), output_size);
    codestream.close();

    H5free_memory(*buf);
    *buf = output.release();
    *buf_size = output_size;
    return output_size;
}

size_t jpeg2000_filter(unsigned flags, size_t cd_nelmts, const unsigned cd_values[], size_t nbytes,
                       size_t *buf_size, void **buf)
{
    configure_codec();
    bool decoding = (flags & H5Z_FLAG_REVERSE) != 0;
    try {
        if (decoding)
            return decode(cd_nelmts, cd_values, nbytes, buf_size, buf);
        return encode(cd_nelmts, cd_values, nbytes, buf_size, buf);
    }
    catch (const std::exception &error) {
        FILTER_ERROR(H5E_CANTFILTER, "%s: %s", decoding ? "cannot decode chunk" : "cannot encode chunk",
                     error.what());
    }
    catch (...) {
        FILTER_ERROR(H5E_CANTFILTER, "%s", decoding ? "cannot decode chunk" : "cannot encode chunk");
    }
    return 0;
}

const H5Z_class2_t jpeg2000_filter_class = {
    H5Z_CLASS_T_VERS,
    jpeg2000_filter_id,
    1,
    1,
    "jpeg2000: JPEG 2000 / HTJ2K codestream per chunk",
    nullptr,
    set_local,
    jpeg2000_filter,
};

} // namespace

extern "C" {

H5PL_type_t H5PLget_plugin_type(void)
{
    return H5PL_TYPE_FILTER;
}

const void *H5PLget_plugin_info(void)
{
    return &jpeg2000_filter_class;
}
}
