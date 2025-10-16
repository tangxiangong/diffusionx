cmake_minimum_required(VERSION 3.18)
project(diffusionx_cuda_kernels CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA architectures (adjust based on your GPU)
# You can override this with: cmake -DCMAKE_CUDA_ARCHITECTURES="75;80;86" ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Compilation flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3
    -use_fast_math
    -Xcompiler -fPIC
    --expt-relaxed-constexpr
)

# Add cuRAND library
find_library(CURAND_LIBRARY curand HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

if(NOT CURAND_LIBRARY)
    message(WARNING "cuRAND library not found. Some features may not work.")
endif()

# List all CUDA source files (matching .rs files in src/simulation/continuous)
set(CUDA_SOURCES
    bm.cu
    ou.cu
    geometric_bm.cu
    fbm.cu
    cauchy.cu
    levy.cu
    langevin.cu
    gamma.cu
    bng.cu
    brownian_bridge.cu
    brownian_excursion.cu
    brownian_meander.cu
    levy_walk.cu
)

# Generate PTX for all sources
add_custom_target(ptx ALL)

foreach(CUDA_SOURCE ${CUDA_SOURCES})
    get_filename_component(KERNEL_NAME ${CUDA_SOURCE} NAME_WE)
    set(PTX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ptx/${KERNEL_NAME}.ptx)

    # Create PTX compilation target
    add_custom_command(
        OUTPUT ${PTX_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/ptx
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            -ptx
            -O3
            -use_fast_math
            $<$<CONFIG:Debug>:-g>
            $<$<CONFIG:Debug>:-G>
            --gpu-architecture=compute_75
            --gpu-code=sm_75,sm_80,sm_86
            -I${CUDA_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}
            -o ${PTX_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}
        COMMENT "Compiling ${CUDA_SOURCE} to PTX"
        VERBATIM
    )

    add_custom_target(${KERNEL_NAME}_ptx
        DEPENDS ${PTX_FILE}
    )

    add_dependencies(ptx ${KERNEL_NAME}_ptx)
endforeach()

# Optional: Compile to shared library for testing
option(BUILD_CUDA_LIBRARY "Build CUDA kernels as shared library" ON)

if(BUILD_CUDA_LIBRARY AND CURAND_LIBRARY)
    add_library(diffusionx_cuda_kernels SHARED ${CUDA_SOURCES})

    set_target_properties(diffusionx_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(diffusionx_cuda_kernels
        ${CUDA_LIBRARIES}
        ${CURAND_LIBRARY}
    )

    # Installation
    install(TARGETS diffusionx_cuda_kernels
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

# Install PTX files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ptx/
    DESTINATION lib/diffusionx/kernels
    FILES_MATCHING PATTERN "*.ptx"
)

# Create a manifest file
set(MANIFEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ptx/manifest.txt)
file(WRITE ${MANIFEST_FILE} "DiffusionX CUDA Kernels\n")
file(APPEND ${MANIFEST_FILE} "Generated: ${CMAKE_BUILD_TYPE}\n")
file(APPEND ${MANIFEST_FILE} "CUDA Version: ${CUDA_VERSION}\n")
file(APPEND ${MANIFEST_FILE} "Architectures: ${CMAKE_CUDA_ARCHITECTURES}\n\n")
file(APPEND ${MANIFEST_FILE} "Kernels:\n")
foreach(CUDA_SOURCE ${CUDA_SOURCES})
    file(APPEND ${MANIFEST_FILE} "  - ${CUDA_SOURCE}\n")
endforeach()

# Print configuration summary
message(STATUS "===========================================")
message(STATUS "DiffusionX CUDA Kernels Configuration")
message(STATUS "===========================================")
message(STATUS "CUDA Compiler:      ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version:       ${CUDA_VERSION}")
message(STATUS "CUDA Toolkit:       ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "cuRAND Library:     ${CURAND_LIBRARY}")
message(STATUS "Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "PTX Output:         ${CMAKE_CURRENT_SOURCE_DIR}/ptx/")
message(STATUS "Build Library:      ${BUILD_CUDA_LIBRARY}")
message(STATUS "===========================================")

# Add test target (optional)
option(BUILD_TESTS "Build CUDA kernel tests" OFF)

if(BUILD_TESTS AND BUILD_CUDA_LIBRARY)
    enable_testing()

    # Simple test executable
    add_executable(test_kernels test_kernels.cu)
    target_link_libraries(test_kernels
        diffusionx_cuda_kernels
        ${CUDA_LIBRARIES}
        ${CURAND_LIBRARY}
    )

    add_test(NAME KernelTests COMMAND test_kernels)
endif()
