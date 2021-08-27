#pragma once

#include "utils.h"


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

std::vector<u8> fromU64(const u64 &u) {
    u8* start = (u8*) &u;
    return std::vector<u8>(start, start + sizeof(u64));
}

u64 toU64(const u8 *data) {
    return *((u64*)data);
}

u64 toU64(int data){
    u64 res = data;
    return res;
}

unsigned long int fromVectorU8(std::vector<u8> data){
    long long res = 0;
    for (int i=data.size()-1; i>=0; i--){
        res = res<<8;
        res += data[i];
    }
    return res;
}

void paddingToBlock(std::vector<u8> &data) {
    u64 more = (sizeof(block) - data.size() % sizeof(block)) % sizeof(block);
    for (auto i = 0; i < more; ++i) {
        data.push_back(0);
    }
}

DATA block_to_data(block blk){
    DATA res;
    std::vector<u8> vec = fromBlock(blk);
    memcpy(res.d, &vec[0], sizeof(block));
    return res;
}

block data_to_block(DATA data){
    block res = toBlock(&data.d[0]);
    return res;
}

void setDataArray(DATA arr[], unsigned long long dataLow[], unsigned long long dataHigh[], int size){
    for (auto i=0; i<size; i++)
        arr[i] = block_to_data(toBlock(dataHigh[i], dataLow[i]));
}

extern "C"
void setData(DATA *data, unsigned long long value){
    *data = block_to_data(toBlock(value));
}

DATA setData(unsigned long long low){
    return block_to_data(toBlock(low));
}

DATA setData(unsigned long long high, unsigned long long low){
    return block_to_data(toBlock(high, low));
}

extern "C"
unsigned char **gen2DArray(int npyLength1D, int npyLength2D){
    unsigned char** res;
    res = new unsigned char *[npyLength1D];
    for (auto i=0; i<npyLength1D; i++)
        res[i] = new unsigned char [npyLength2D];
    return res;
}

extern "C"
void del2DArray(unsigned char **arr, int npyLength1D){
    for (auto i=0; i<npyLength1D; i++){
        delete [] arr[i];
    }
    delete [] arr;
}
extern "C"
unsigned char get2DArrayElement(unsigned char *arr, int npyLength2D, int x, int y){
    return *(arr + x*npyLength2D + y);
}

DATA get2DArrayElement(DATA **arr, int x, int y){
    return arr[x][y];
}
extern "C"
void get2DArrayRow(unsigned char *row, int size, unsigned char **arr, int x){
    memcpy(row, arr[x], size);
}

void set2DArrayElement(DATA **arr, int x, int y, unsigned long long low){
    setData(&arr[x][y], low);
}

void set2DArrayElement(DATA **arr, int x, int y, unsigned long long high, unsigned long long low){
    arr[x][y] = setData(high, low);
}
void set2DArrayElement(unsigned char **arr, int x, int y, unsigned char data){
    arr[x][y] = data;
}
extern "C"
void set2DArrayRow(unsigned char **arr, int x, unsigned char *row, int size){
    memcpy(arr[x], row, size);
}

unsigned long long getDataValueLow(DATA data){
    return toU64(&data.d[0]);
}

unsigned long long* getDataValue(DATA data){
    auto *res = new unsigned long long[2];
    res[0] = toU64(&data.d[0]);
    res[1] = toU64(&data.d[8]);
    return res;
}

DATA **combineOtMessages(DATA otMessages1[], DATA otMessages2[], int len){
    DATA** messages;
    messages = new DATA *[len];
    for (auto i=0; i<len; i++){
        messages[i] = new DATA [2];
        messages[i][0] = otMessages1[i];
        messages[i][1] = otMessages2[i];
    }
    return messages;
}

extern "C"
void mallocArray(unsigned char *arr[], int n1, int n2){
    for (int i=0; i<n1; i++)
        arr[i] = (unsigned char *) malloc(sizeof(unsigned char) * n2);
}

extern "C"
void freeArray(unsigned char *arr[], int n1){
    for (int i=0; i<n1; i++)
        free(arr[i]);
}

extern "C"
void shortHash(u64 data, unsigned char *hashOutput, int lengthInBytes){
    RandomOracle H(lengthInBytes);
    unsigned char hashInput[8];
    memcpy(hashInput, &data, 8);
    H.Update(hashInput, 8);
    H.Final(hashOutput);
}
