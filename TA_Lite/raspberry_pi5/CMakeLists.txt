cmake_minimum_required(VERSION 3.10)
project(program_lite_rpi5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Optimization compiler flags specifically for Raspberry Pi 5 (Cortex-A76 / ARMv8.2-A)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3)
    add_compile_options(-ffast-math)
    
    # Check if compiling on ARM (e.g. Raspberry Pi 5)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        message(STATUS "Configuring build with ARM Cortex-A76 (Raspberry Pi 5) optimizations.")
        add_compile_options(-mcpu=cortex-a76)
        add_compile_options(-march=armv8.2-a+fp16+simd)
    else()
        message(STATUS "Cross-compiling or building on x86/other architecture. Using -march=native.")
        add_compile_options(-march=native)
    endif()
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Try to find OpenMP for parallel thread scheduling
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found. Parallel acceleration enabled.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Executable for camera real-time detection
add_executable(program_optimized
    main_optimized.cpp
    object_detector.cpp
)
target_link_libraries(program_optimized ${OpenCV_LIBS})
if(OpenMP_FOUND)
    target_link_libraries(program_optimized OpenMP::OpenMP_CXX)
endif()

# Executable for video benchmarking
add_executable(program_video
    main_video.cpp
    object_detector.cpp
)
target_link_libraries(program_video ${OpenCV_LIBS})
if(OpenMP_FOUND)
    target_link_libraries(program_video OpenMP::OpenMP_CXX)
endif()

# Executable for interactive launcher menu
add_executable(program_menu
    menu.cpp
)
target_link_libraries(program_menu ${OpenCV_LIBS})
