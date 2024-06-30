cmake_minimum_required(VERSION 3.10)

project(main VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CTest)
enable_testing()

#list(APPEND CMAKE_MODULE_PATH "deps/tbb/cmake/")
#find_package(TBB REQUIRED)

#include(mingw64-toolchain.cmake)

add_executable(main main.cpp mnist.cpp network.cpp)

#target_link_libraries(main PUBLIC TBB::tbb)
#target_link_libraries(main -static)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DPARALLEL")
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
