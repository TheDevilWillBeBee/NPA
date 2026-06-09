cmake_minimum_required(VERSION 3.18...3.27)  # Increased minimum for CUDA support

set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

project(my_project LANGUAGES CXX CUDA)  # Added CUDA language

# Enable CUDA
enable_language(CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -g -lineinfo -src-in-ptx")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -lineinfo -src-in-ptx")

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)


nanobind_add_module(sph_cuda sph_cuda.cu)

set_property(TARGET sph_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET sph_cuda PROPERTY CUDA_ARCHITECTURES "75;80;86")

target_compile_options(sph_cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-g>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-src-in-ptx>
)


target_link_libraries(sph_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


