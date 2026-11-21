cmake_minimum_required(VERSION 3.20)

project(h5jpeg2000 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(GNUInstallDirs)

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(ojph CONFIG REQUIRED)

set(HDF5_PLUGIN_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/plugin"
    CACHE STRING "Directory, absolute or relative to the install prefix, that receives the filter plugin")

if(IS_ABSOLUTE "${HDF5_PLUGIN_INSTALL_DIR}")
  set(_h5jpeg2000_plugin_dir "${HDF5_PLUGIN_INSTALL_DIR}")
else()
  set(_h5jpeg2000_plugin_dir "${CMAKE_INSTALL_PREFIX}/${HDF5_PLUGIN_INSTALL_DIR}")
endif()
# The run-time search path from the installed plugin to the libraries it links.
file(RELATIVE_PATH _h5jpeg2000_libdir_from_plugin
  "${_h5jpeg2000_plugin_dir}" "${CMAKE_INSTALL_FULL_LIBDIR}")

add_library(h5jpeg2000 MODULE src/h5jpeg2000.cpp)
target_link_libraries(h5jpeg2000 PRIVATE HDF5::HDF5 ojph::ojph)
if(WIN32)
  target_compile_definitions(h5jpeg2000 PRIVATE H5_BUILT_AS_DYNAMIC_LIB)
endif()
if(MSVC)
  target_compile_options(h5jpeg2000 PRIVATE /W4 /EHsc)
else()
  target_compile_options(h5jpeg2000 PRIVATE -Wall -Wextra)
endif()
if(APPLE)
  set_target_properties(h5jpeg2000 PROPERTIES SUFFIX ".dylib" INSTALL_RPATH "@loader_path/${_h5jpeg2000_libdir_from_plugin}")
  target_link_options(h5jpeg2000 PRIVATE
    "LINKER:-exported_symbol,_H5PLget_plugin_type"
    "LINKER:-exported_symbol,_H5PLget_plugin_info")
elseif(UNIX)
  set_target_properties(h5jpeg2000 PROPERTIES INSTALL_RPATH "$ORIGIN/${_h5jpeg2000_libdir_from_plugin}")
  target_link_options(h5jpeg2000 PRIVATE "LINKER:--exclude-libs,ALL")
endif()

install(TARGETS h5jpeg2000
  LIBRARY DESTINATION "${HDF5_PLUGIN_INSTALL_DIR}"
  RUNTIME DESTINATION "${HDF5_PLUGIN_INSTALL_DIR}"
)

install(FILES LICENSE.txt DESTINATION share/doc/hdf5_jpeg2000)
