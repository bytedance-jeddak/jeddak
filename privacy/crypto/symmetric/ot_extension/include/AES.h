//
// Created by fsecond on 2021/3/11.
//
#pragma once
#ifndef FL_PSI_AES_H
#define FL_PSI_AES_H

#include "Defines.h"
#include <wmmintrin.h>

class AES
{
public:

    // Default constructor leave the class in an invalid state
    // until setKey(...) is called.
    AES() {};
    AES(const AES&) = default;

    // Constructor to initialize the class with the given key
    AES(const block& userKey);

    // Set the key to be used for encryption.
    void setKey(const block& userKey);

    // Encrypts the plaintext block and stores the result in ciphertext
    void ecbEncBlock(const block& plaintext, block& ciphertext) const;

    // Encrypts the plaintext block and returns the result
    block ecbEncBlock(const block& plaintext) const;

    // Encrypts blockLength starting at the plaintexts pointer and writes the result
    // to the ciphertext pointer
    void ecbEncBlocks(const block* plaintexts, u64 blockLength, block* ciphertext) const;

    void ecbEncCounterMode(u64 baseIdx, u64 length, block* ciphertext) const;
    // Returns the current key.
    const block& getKey() const { return mRoundKey[0]; }

    // The expanded key.
    block mRoundKey[11];
};

#endif //FL_PSI_AES_H
