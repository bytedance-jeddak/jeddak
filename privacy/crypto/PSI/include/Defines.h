#pragma once

#include <vector>
#include <emmintrin.h>



typedef __m128i block;
typedef unsigned char u8;
typedef unsigned long int u32;
typedef unsigned long long u64;

typedef long int i32;
typedef struct {
    unsigned char d[16];
}DATA;

typedef struct {
    unsigned long int width;
    unsigned long int logHeight;
    unsigned long int senderSize;
    unsigned long int receiverSize;
    unsigned long int h1LengthInBytes;
    unsigned long int hashLengthInBytes;
}PARAMETERS;
