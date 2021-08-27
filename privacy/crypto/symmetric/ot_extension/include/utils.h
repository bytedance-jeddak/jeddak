#pragma once
#ifndef __UTILS_H
#define __UTILS_H
#include "Defines.h"
#include <cstring>
#include "RandomOracle.h"

void
sse_trans(uint8_t const *inp, uint8_t *out, int nrows, int ncols);

block toBlock(u64 low_u64);

block toBlock(u64 high_u64, u64 low_u64);

block toBlock(u8*data);

block *initBlockArray(int length);

block **initBlock2DArray(int length1, int length2);

void get2DArrayRow(std::vector<u64> &high, std::vector<u64> &low, block **arr, int row, int length);

void deleteBlockArray(block *arr);

void deleteBlock2DArray(block **arr, int length);

void setBlockArray(block *arr, std::vector<u64> &high, std::vector<u64> &low, int length);

void setBlockArray(block **arr, int row, std::vector<u64> &high, std::vector<u64> &low, int length);

void getBlockArray(std::vector<u64> &high, std::vector<u64> &low, block *arr, int length);

void getNumpyArray(unsigned long long *numpy_arr, int size, std::vector<u64> &vec);

std::vector<u8> fromBlock(const block &block);

#endif

