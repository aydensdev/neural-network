# mingw-w64-toolchain.cmake

# The name of the target operating system
set(CMAKE_SYSTEM_NAME Windows)

# Which compilers to use for C and C++
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)

# Specify the target architecture (optional, but recommended)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")