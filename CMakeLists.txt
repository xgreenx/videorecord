cmake_minimum_required(VERSION 3.7)
project(VideoRecord)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp Component.cpp Component.h)
add_executable(VideoRecord ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(VideoRecord ${OpenCV_LIBS})