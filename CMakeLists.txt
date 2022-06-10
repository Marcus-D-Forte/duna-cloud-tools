cmake_minimum_required(VERSION 3.10)
project(duna-cloud-tools)

find_package(PCL 1.10 REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})

link_libraries( ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

add_library(global_registrator
lib/global_registration)

add_library(asl_parser 
lib/asl_parser.cpp)


add_executable(keypoint src/keypoint.cpp)
add_executable(keypoint_registration src/keypoint_registration.cpp)

target_link_libraries(keypoint asl_parser global_registrator)
target_link_libraries(keypoint_registration asl_parser global_registrator )


# Batch processors
add_executable(batch_voxel src/batch_voxel.cpp)
add_executable(batch_sor src/batch_sor.cpp )
add_executable(batch_ror src/batch_ror.cpp )





