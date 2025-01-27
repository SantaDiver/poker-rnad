cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
project(poker-rnad VERSION 0.0.0)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_subdirectory(contrib)
add_subdirectory(cli)
add_subdirectory(lib/cpp)