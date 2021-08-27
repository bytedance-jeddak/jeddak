cmake_minimum_required(VERSION 3.7)
project(BLAKE)


set(CMAKE_CXX_STANDARD 14)


aux_source_directory(./ DIRSRCS)

ADD_LIBRARY (blake STATIC ${DIRSRCS})