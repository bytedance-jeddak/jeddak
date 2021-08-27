cmake_minimum_required(VERSION 3.7)
project(oblivious_transfer)
#set( CMAKE_BUILD_TYPE "Release" )
#set(CMAKE_CXX_FLAGS_Release "${CMAKE_CXX_FLAGS_Release}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_DEBUG")
aux_source_directory(./src DIRSRCS)

include_directories(./include)
add_executable(main test/main.cpp ${DIRSRCS} src/iknpOTeSender.cpp)
add_executable(test test/test.cpp ${DIRSRCS} src/iknpOTeSender.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-g -O0 -Wall  -msse2 -msse -march=native -maes")