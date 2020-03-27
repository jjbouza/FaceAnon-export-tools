cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(processing)

find_package(Torch REQUIRED)

add_library(processing SHARED src/processing.cpp src/processing.h src/processing_wrapper.cpp src/processing_wrapper.h)
add_executable(processing_host src/host.cpp src/processing.cpp src/processing.h)
target_link_libraries(processing "${TORCH_LIBRARIES}")
target_link_libraries(processing_host "${TORCH_LIBRARIES}")
set_property(TARGET processing PROPERTY CXX_STANDARD 14)
set_property(TARGET processing_host PROPERTY CXX_STANDARD 14)
