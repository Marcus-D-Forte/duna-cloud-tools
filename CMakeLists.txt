cmake_minimum_required(VERSION 3.10)
project(duna-cloud-tools)

find_package(PCL 1.10 REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

add_library(global_registrator
lib/global_registration)

add_library(asl_parser 
lib/asl_parser.cpp)



add_executable(keypoint src/keypoint.cpp)
add_executable(keypoint_registration src/keypoint_registration.cpp)

target_link_libraries(keypoint asl_parser ${PCL_LIBRARIES} global_registrator)
target_link_libraries(keypoint_registration asl_parser ${PCL_LIBRARIES} global_registrator OpenMP::OpenMP_CXX)








