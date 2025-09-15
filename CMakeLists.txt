cmake_minimum_required(VERSION 3.16)
project(EdgesCGALWeightedDelaunay2D LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CGAL REQUIRED)

add_executable(EdgesCGALWeightedDelaunay2D src/EdgesCGALWeightedDelaunay2D.cpp)
target_link_libraries(EdgesCGALWeightedDelaunay2D PRIVATE CGAL::CGAL)
target_compile_options(EdgesCGALWeightedDelaunay2D PRIVATE -O3 -DNDEBUG)