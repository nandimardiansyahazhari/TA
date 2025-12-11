cmake_minimum_required(VERSION 3.10)
project(program_lite)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories("/usr/local/include")

add_executable(program_lite main.cpp)
target_link_libraries(program_lite ${OpenCV_LIBS})
