#pragma once

#include "utils.h"
#include <cassert>
void
sse_trans(uint8_t const *inp, uint8_t *out, int nrows, int ncols)
{
#   define INP(x,y) inp[(x)*ncols/8 + (y)/8]
#   define OUT(x,y) out[(y)*nrows/8 + (x)/8]
    int rr, cc, i, h;
    union { __m128i x; uint8_t b[16]; } tmp;
    assert(nrows % 8 == 0 && ncols % 8 == 0);

    // Do the main body in 16x8 blocks:
    for (rr = 0; rr <= nrows - 16; rr += 16) {
        for (cc = 0; cc < ncols; cc += 8) {
            for (i = 0; i < 16; ++i)
                tmp.b[i] = INP(rr + i, cc);
            for (i = 8; --i >= 0; tmp.x = _mm_slli_epi64(tmp.x, 1))
                *(uint16_t*)&OUT(rr,cc+i)= _mm_movemask_epi8(tmp.x);
        }
    }
    if (rr == nrows) return;

    // The remainder is a block of 8x(16n+8) bits (n may be 0).
    //  Do a PAIR of 8x8 blocks in each step:
    for (cc = 0; cc <= ncols - 16; cc += 16) {
        for (i = 0; i < 8; ++i) {
            tmp.b[i] = h = *(uint16_t const*)&INP(rr + i, cc);
            tmp.b[i + 8] = h >> 8;
        }
        for (i = 8; --i >= 0; tmp.x = _mm_slli_epi64(tmp.x, 1)) {
            OUT(rr, cc + i) = h = _mm_movemask_epi8(tmp.x);
            OUT(rr, cc + i + 8) = h >> 8;
        }
    }
    if (cc == ncols) return;

    //  Do the remaining 8x8 block:
    for (i = 0; i < 8; ++i)
        tmp.b[i] = INP(rr + i, cc);
    for (i = 8; --i >= 0; tmp.x = _mm_slli_epi64(tmp.x, 1))
        OUT(rr, cc + i) = _mm_movemask_epi8(tmp.x);
}

block toBlock(u64 low_u64) {
    return _mm_set_epi64x(0, low_u64);
}

block toBlock(u64 high_u64, u64 low_u64) {
    return _mm_set_epi64x(high_u64, low_u64);
}

block toBlock(u8*data) {
    return _mm_set_epi64x(((u64*)data)[1], ((u64*)data)[0]);
}

std::vector<u8> fromBlock(const block &block) {
    u8* start = (u8*) &block;
    return std::vector<u8>(start, start + sizeof(block));
}


block *initBlockArray(int length){
    auto *block_arr = new block [length];
    return block_arr;
}

block **initBlock2DArray(int length1, int length2){
    auto **block_arr = new block*[length1];
    for (auto i=0; i<length1; i++)
        block_arr[i] = new block[length2];
    return block_arr;
}


void get2DArrayRow(std::vector<u64> &high, std::vector<u64> &low, block **arr, int row, int length){
    for (auto i=0; i<length; i++){
        low[i] = *((u64 *)(arr[row]+i));
        high[i] = *((u64 *)(arr[row]+i) + 1);
    }
}


void deleteBlockArray(block *arr){
    delete [] arr;
}

void deleteBlock2DArray(block **arr, int length){
    for (auto i=0; i<length; i++){
        delete [] arr[i];
    }
    delete [] arr;
}
void setBlockArray(block *arr, std::vector<u64> &high, std::vector<u64> &low, int length){
    for (auto i=0; i<length; i++){
        arr[i] = toBlock(high[i], low[i]);
    }
}

void setBlockArray(block **arr, int row, std::vector<u64> &high, std::vector<u64> &low, int length){
    for (auto i=0; i<length; i++){
        arr[row][i] = toBlock(high[i], low[i]);
    }
}

void getBlockArray(std::vector<u64> &high, std::vector<u64> &low, block *arr, int length){
    for (auto i=0; i<length; i++){
        low[i] = *((u64 *)(arr+i));
        high[i] = *((u64 *)(arr+i) + 1);
    }
}

void getNumpyArray(unsigned long long *numpy_arr, int size, std::vector<u64> &vec){
    memcpy(numpy_arr, &vec[0], 8*size);
}