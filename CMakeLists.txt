cmake_minimum_required(VERSION 3.14)
project(llm_lv1_cpp VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard (C++17 by default, supports C++11/14/17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Allow user to select C++ standard
if(NOT DEFINED CPP_STANDARD)
    set(CPP_STANDARD 17)
endif()

if(CPP_STANDARD EQUAL 11)
    set(CMAKE_CXX_STANDARD 11)
    message(STATUS "Building with C++11")
elseif(CPP_STANDARD EQUAL 14)
    set(CMAKE_CXX_STANDARD 14)
    message(STATUS "Building with C++14")
elseif(CPP_STANDARD EQUAL 17)
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "Building with C++17")
else()
    message(FATAL_ERROR "Unsupported C++ standard: ${CPP_STANDARD}. Use 11, 14, or 17.")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_OPENMP "Enable OpenMP support" ON)

# Find dependencies
find_package(Threads REQUIRED)

if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found")
        add_compile_definitions(USE_OPENMP)
    endif()
endif()

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        add_compile_definitions(USE_CUDA)
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/cpp_include)

# Source files
set(CORE_SOURCES
    cpp_src/core/config.cpp
    cpp_src/core/engine.cpp
    cpp_src/core/state.cpp
    cpp_src/core/tokenizer.cpp
)

set(KERNEL_SOURCES
    cpp_src/kernel/attention.cpp
    cpp_src/kernel/activation.cpp
    cpp_src/kernel/normalization.cpp
)

set(RUNTIME_SOURCES
    cpp_src/runtime/memory_manager.cpp
    cpp_src/runtime/scheduler.cpp
    cpp_src/runtime/context.cpp
)

set(MODEL_SOURCES
    cpp_src/model/model_loader.cpp
    cpp_src/model/model_registry.cpp
)

set(UTILS_SOURCES
    cpp_src/utils/logging.cpp
    cpp_src/utils/math_utils.cpp
)

# Create library
add_library(llm_lv1_lib STATIC
    ${CORE_SOURCES}
    ${KERNEL_SOURCES}
    ${RUNTIME_SOURCES}
    ${MODEL_SOURCES}
    ${UTILS_SOURCES}
)

target_include_directories(llm_lv1_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/cpp_include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(llm_lv1_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(llm_lv1_lib PUBLIC Threads::Threads)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(llm_lv1_lib PRIVATE NOMINMAX)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    set(TEST_SOURCES
        cpp_tests/test_attention.cpp
        cpp_tests/test_memory_manager.cpp
        cpp_tests/test_scheduler.cpp
        cpp_tests/test_engine.cpp
    )
    
    add_executable(run_tests ${TEST_SOURCES})
    target_link_libraries(run_tests llm_lv1_lib)
    
    add_test(NAME RunTests COMMAND run_tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_executable(simple_inference cpp_examples/simple_inference.cpp)
    target_link_libraries(simple_inference llm_lv1_lib)
    
    add_executable(streaming_example cpp_examples/streaming_example.cpp)
    target_link_libraries(streaming_example llm_lv1_lib)
endif()

# Installation
install(TARGETS llm_lv1_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY cpp_include/
    DESTINATION include/llm_lv1
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== LLM-LV1 C++ Configuration Summary ===")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "CUDA Support: ${USE_CUDA}")
message(STATUS "OpenMP Support: ${USE_OPENMP}")
message(STATUS "==========================================")
message(STATUS "")
