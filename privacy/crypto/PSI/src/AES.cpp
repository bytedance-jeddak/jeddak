#include "AES.h"

#include <array>

const AES mAesFixedKey(_mm_set_epi8(36, -100, 50, -22, 92, -26, 49, 9, -82, -86, -51, -96, 98, -20, 29, -13));

AES::AES(const block & userKey)
{
    setKey(userKey);
}

block keyGenHelper(block key, block keyRcon)
{
    keyRcon = _mm_shuffle_epi32(keyRcon, _MM_SHUFFLE(3, 3, 3, 3));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, keyRcon);
}

void AES::setKey(const block & userKey)
{
    mRoundKey[0] = userKey;
    mRoundKey[1] = keyGenHelper(mRoundKey[0], _mm_aeskeygenassist_si128(mRoundKey[0], 0x01));
    mRoundKey[2] = keyGenHelper(mRoundKey[1], _mm_aeskeygenassist_si128(mRoundKey[1], 0x02));
    mRoundKey[3] = keyGenHelper(mRoundKey[2], _mm_aeskeygenassist_si128(mRoundKey[2], 0x04));
    mRoundKey[4] = keyGenHelper(mRoundKey[3], _mm_aeskeygenassist_si128(mRoundKey[3], 0x08));
    mRoundKey[5] = keyGenHelper(mRoundKey[4], _mm_aeskeygenassist_si128(mRoundKey[4], 0x10));
    mRoundKey[6] = keyGenHelper(mRoundKey[5], _mm_aeskeygenassist_si128(mRoundKey[5], 0x20));
    mRoundKey[7] = keyGenHelper(mRoundKey[6], _mm_aeskeygenassist_si128(mRoundKey[6], 0x40));
    mRoundKey[8] = keyGenHelper(mRoundKey[7], _mm_aeskeygenassist_si128(mRoundKey[7], 0x80));
    mRoundKey[9] = keyGenHelper(mRoundKey[8], _mm_aeskeygenassist_si128(mRoundKey[8], 0x1B));
    mRoundKey[10] = keyGenHelper(mRoundKey[9], _mm_aeskeygenassist_si128(mRoundKey[9], 0x36));
}

void  AES::ecbEncBlock(const block & plaintext, block & cyphertext) const
{
    cyphertext = _mm_xor_si128(plaintext, mRoundKey[0]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[1]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[2]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[3]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[4]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[5]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[6]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[7]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[8]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[9]);
    cyphertext = _mm_aesenclast_si128(cyphertext, mRoundKey[10]);
}

block AES::ecbEncBlock(const block & plaintext) const
{
    block ret;
    ecbEncBlock(plaintext, ret);
    return ret;
}

void AES::ecbEncBlocks(const block * plaintexts, u64 blockLength, block * cyphertext) const
{
    const u64 step = 8;
    u64 idx = 0;
    u64 length = blockLength - blockLength % step;

    //std::array<block, step> temp;
    block temp[step];

    for (; idx < length; idx += step)
    {
        temp[0] = _mm_xor_si128(plaintexts[idx + 0], mRoundKey[0]);
        temp[1] = _mm_xor_si128(plaintexts[idx + 1], mRoundKey[0]);
        temp[2] = _mm_xor_si128(plaintexts[idx + 2], mRoundKey[0]);
        temp[3] = _mm_xor_si128(plaintexts[idx + 3], mRoundKey[0]);
        temp[4] = _mm_xor_si128(plaintexts[idx + 4], mRoundKey[0]);
        temp[5] = _mm_xor_si128(plaintexts[idx + 5], mRoundKey[0]);
        temp[6] = _mm_xor_si128(plaintexts[idx + 6], mRoundKey[0]);
        temp[7] = _mm_xor_si128(plaintexts[idx + 7], mRoundKey[0]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[1]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[1]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[1]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[1]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[1]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[1]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[1]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[1]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[2]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[2]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[2]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[2]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[2]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[2]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[2]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[2]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[3]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[3]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[3]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[3]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[3]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[3]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[3]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[3]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[4]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[4]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[4]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[4]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[4]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[4]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[4]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[4]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[5]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[5]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[5]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[5]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[5]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[5]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[5]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[5]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[6]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[6]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[6]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[6]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[6]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[6]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[6]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[6]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[7]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[7]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[7]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[7]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[7]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[7]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[7]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[7]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[8]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[8]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[8]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[8]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[8]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[8]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[8]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[8]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[9]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[9]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[9]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[9]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[9]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[9]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[9]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[9]);

        cyphertext[idx + 0] = _mm_aesenclast_si128(temp[0], mRoundKey[10]);
        cyphertext[idx + 1] = _mm_aesenclast_si128(temp[1], mRoundKey[10]);
        cyphertext[idx + 2] = _mm_aesenclast_si128(temp[2], mRoundKey[10]);
        cyphertext[idx + 3] = _mm_aesenclast_si128(temp[3], mRoundKey[10]);
        cyphertext[idx + 4] = _mm_aesenclast_si128(temp[4], mRoundKey[10]);
        cyphertext[idx + 5] = _mm_aesenclast_si128(temp[5], mRoundKey[10]);
        cyphertext[idx + 6] = _mm_aesenclast_si128(temp[6], mRoundKey[10]);
        cyphertext[idx + 7] = _mm_aesenclast_si128(temp[7], mRoundKey[10]);
    }

    for (; idx < blockLength; ++idx)
    {
        cyphertext[idx] = _mm_xor_si128(plaintexts[idx], mRoundKey[0]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[1]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[2]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[3]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[4]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[5]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[6]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[7]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[8]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[9]);
        cyphertext[idx] = _mm_aesenclast_si128(cyphertext[idx], mRoundKey[10]);
    }

}

void AES::ecbEncCounterMode(u64 baseIdx, u64 blockLength, block * cyphertext) const
{
    const i32 step = 8;
    i32 idx = 0;
    i32 length = blockLength - blockLength % step;

    //std::array<block, step> temp;
    block temp[step];

    for (; idx < length; idx += step, baseIdx += step)
    {
        temp[0] = _mm_xor_si128(_mm_set1_epi64x(baseIdx + 0), mRoundKey[0]);
        temp[1] = _mm_xor_si128(_mm_set1_epi64x(baseIdx + 1), mRoundKey[0]);
        temp[2] = _mm_xor_si128(_mm_set1_epi64x(baseIdx + 2), mRoundKey[0]);
        temp[3] = _mm_xor_si128(_mm_set1_epi64x(baseIdx + 3), mRoundKey[0]);
        temp[4] = _mm_xor_si128(_mm_set1_epi64x(baseIdx + 4), mRoundKey[0]);
        temp[5] = _mm_xor_si128(_mm_set1_epi64x(baseIdx + 5), mRoundKey[0]);
        temp[6] = _mm_xor_si128(_mm_set1_epi64x(baseIdx + 6), mRoundKey[0]);
        temp[7] = _mm_xor_si128(_mm_set1_epi64x(baseIdx + 7), mRoundKey[0]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[1]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[1]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[1]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[1]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[1]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[1]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[1]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[1]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[2]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[2]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[2]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[2]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[2]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[2]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[2]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[2]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[3]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[3]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[3]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[3]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[3]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[3]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[3]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[3]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[4]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[4]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[4]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[4]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[4]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[4]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[4]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[4]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[5]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[5]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[5]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[5]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[5]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[5]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[5]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[5]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[6]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[6]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[6]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[6]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[6]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[6]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[6]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[6]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[7]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[7]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[7]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[7]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[7]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[7]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[7]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[7]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[8]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[8]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[8]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[8]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[8]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[8]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[8]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[8]);

        temp[0] = _mm_aesenc_si128(temp[0], mRoundKey[9]);
        temp[1] = _mm_aesenc_si128(temp[1], mRoundKey[9]);
        temp[2] = _mm_aesenc_si128(temp[2], mRoundKey[9]);
        temp[3] = _mm_aesenc_si128(temp[3], mRoundKey[9]);
        temp[4] = _mm_aesenc_si128(temp[4], mRoundKey[9]);
        temp[5] = _mm_aesenc_si128(temp[5], mRoundKey[9]);
        temp[6] = _mm_aesenc_si128(temp[6], mRoundKey[9]);
        temp[7] = _mm_aesenc_si128(temp[7], mRoundKey[9]);

        cyphertext[idx + 0] = _mm_aesenclast_si128(temp[0], mRoundKey[10]);
        cyphertext[idx + 1] = _mm_aesenclast_si128(temp[1], mRoundKey[10]);
        cyphertext[idx + 2] = _mm_aesenclast_si128(temp[2], mRoundKey[10]);
        cyphertext[idx + 3] = _mm_aesenclast_si128(temp[3], mRoundKey[10]);
        cyphertext[idx + 4] = _mm_aesenclast_si128(temp[4], mRoundKey[10]);
        cyphertext[idx + 5] = _mm_aesenclast_si128(temp[5], mRoundKey[10]);
        cyphertext[idx + 6] = _mm_aesenclast_si128(temp[6], mRoundKey[10]);
        cyphertext[idx + 7] = _mm_aesenclast_si128(temp[7], mRoundKey[10]);
    }

    for (; idx < static_cast<i32>(blockLength); ++idx, ++baseIdx)
    {
        cyphertext[idx] = _mm_xor_si128(_mm_set1_epi64x(baseIdx), mRoundKey[0]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[1]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[2]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[3]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[4]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[5]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[6]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[7]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[8]);
        cyphertext[idx] = _mm_aesenc_si128(cyphertext[idx], mRoundKey[9]);
        cyphertext[idx] = _mm_aesenclast_si128(cyphertext[idx], mRoundKey[10]);
    }

}


