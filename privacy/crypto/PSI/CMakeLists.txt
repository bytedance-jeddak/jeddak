cmake_minimum_required(VERSION 3.7)
project(FL_PSI)


set(CMAKE_CXX_STANDARD 14)

aux_source_directory(./src DIRSRCS)
aux_source_directory(./blake2 BLAKESRCS)
include_directories(./include)
include_directories(./blake2)
add_library(blake STATIC ${BLAKESRCS} include/RandomOracle.h)

add_executable(FL_PSI main.cpp ${DIRSRCS} ${BLAKESRCS} ./src/PsiReceiver.cpp include/AES.h src/AES.cpp include/PRNG.h src/PRNG.cpp include/Blake.h src/Blake.cpp src/RandomOracle.cpp)

target_link_libraries(FL_PSI blake)


set(CMAKE_CXX_FLAGS "-g -O0 -Wall  -msse2 -msse -march=native -maes")