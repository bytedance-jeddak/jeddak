#pragma once

#include "Defines.h"
#include <cstring>
#include "RandomOracle.h"

block toBlock(u64 low_u64);
block toBlock(u64 high_u64, u64 low_u64);
block toBlock(u8*data);

std::vector<u8> fromBlock(const block &block);

std::vector<u8> fromU64(const u64 &u);

u64 toU64(const u8 *data);

u64 toU64(int data);

unsigned long int fromVectorU8(std::vector<u8> data);

void paddingToBlock(std::vector<u8> &data);

DATA block_to_data(block blk);

block data_to_block(DATA data);

void setDataArray(DATA arr[], unsigned long long dataLow[], unsigned long long dataHigh[], int size);

extern "C"
void setData(DATA *data, unsigned long long value);

DATA setData(unsigned long long low);

DATA setData(unsigned long long high, unsigned long long low);


extern "C"
unsigned char **gen2DArray(int npyLength1D, int npyLength2D);

extern "C"
void del2DArray(unsigned char **arr, int npyLength1D);

extern "C"
unsigned char get2DArrayElement(unsigned char *arr, int npyLength2D, int x, int y);

extern "C"
void get2DArrayRow(unsigned char *row, int size, unsigned char **arr, int x);

DATA get2DArrayElement(DATA **arr, int x, int y);

void set2DArrayElement(DATA **arr, int x, int y, unsigned long long low);

void set2DArrayElement(DATA **arr, int x, int y, unsigned long long high, unsigned long long low);

void set2DArrayElement(unsigned char **arr, int x, int y, unsigned char data);
extern "C"
void set2DArrayRow(unsigned char **arr, int x, unsigned char *row, int size);

//    void getDataValue(DATA data, std::vector<unsigned char> v);
unsigned long long getDataValueLow(DATA data);

unsigned long long* getDataValue(DATA data);

DATA **combineOtMessages(DATA otMessages1[], DATA otMessages2[], int len);

extern "C"
void mallocArray(unsigned char *arr[], int n1, int n2);

extern "C"
void freeArray(unsigned char *arr[], int n1);

extern "C"
void shortHash(u64 data, unsigned char *hashOutput, int lengthInBytes);

