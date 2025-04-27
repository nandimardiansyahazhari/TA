# Minimum versi CMake yang diperlukan
cmake_minimum_required(VERSION 3.10)

# Nama proyek
project(program)

# Set standar C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Cari pustaka OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV tidak ditemukan. Periksa instalasi OpenCV Anda.")
endif()

include_directories("usr/local/include")

# Cari pustaka pthreadpool
find_library(PTHREADPOOL_LIB pthreadpool REQUIRED)
if(NOT PTHREADPOOL_LIB)
    message(FATAL_ERROR "Library pthreadpool tidak ditemukan.")
endif()

# Tambahkan executable proyek
add_executable(program coba.cpp)

# Tautkan pustaka OpenCV dan pthreadpool
target_link_libraries(program
    ${OpenCV_LIBS}
    ${PTHREADPOOL_LIB}
)

# Tambahkan include directory
target_include_directories(program
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)