cmake_minimum_required(VERSION 3.17)
project(PSI)

set(CMAKE_C_STANDARD 99)

include_directories(NP01OT)

AUX_SOURCE_DIRECTORY(NP01OT DIR_SRCS)

add_executable(untitled test.c ecp_ED25519.c randapi.c big_256_56.c fp_25519.c oct.c rand.c hash.c rom_curve_ED25519.c
        ecdh_ED25519.c aes.c ecdh_support.c rom_field_25519.c)