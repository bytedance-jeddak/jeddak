#include "PsiReceiver.h"

#include <array>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <bitset>
#include <thread>

/** 计算输入数据集对应的PRF结果，即所有的v （其中PRF用AES加密实现） */
void PsiReceiver::evaluatePRF(block key, std::vector<block>& receiverSet, const u64& h1LengthInBytes, const u64& receiverSize, block* recvSet){
    AES commonAes;
    commonAes.setKey(key);

    block* aesInput = new block[receiverSize];
    block* aesOutput = new block[receiverSize];

    RandomOracle H1(h1LengthInBytes);
    u8 h1Output[h1LengthInBytes];

    for (auto i = 0; i < receiverSize; ++i) {
        H1.Reset();
        H1.Update((u8*)(receiverSet.data() + i), sizeof(block));
        H1.Final(h1Output);

        aesInput[i] = *(block*)h1Output;
        recvSet[i] = *(block*)(h1Output + sizeof(block));
    }

    commonAes.ecbEncBlocks(aesInput, receiverSize, aesOutput);


    for (auto i = 0; i < receiverSize; ++i) {
        recvSet[i] ^= aesOutput[i];
    }
    delete []aesInput;
    delete []aesOutput;
}
/** 计算矩阵transposed locations，即v[1], v[2],..., v[w]组成的矩阵
* 输入:
*     commonPrng: 用于生成一系列key，构造PRF
*     width:
*     widthBucket1:
*     receiverSize:
*     bucket1:
*     locationInBytes:
*     recvSet:
* 输出:
*     transLocations:
*     */
void PsiReceiver::evaluateTransLocations(block keys[], const u64& width, const u64& widthBucket1, const u64& receiverSize, const u64& bucket1, const u64& locationInBytes, block* recvSet, u8* transLocations[]){
    AES commonAes;
    block commonKey;
    block randomLocations[bucket1];

    for (auto wLeft = 0; wLeft < width; wLeft += widthBucket1) {
        auto wRight = wLeft + widthBucket1 < width ? wLeft + widthBucket1 : width;
        auto w = wRight - wLeft;
        commonAes.setKey(keys[wLeft/widthBucket1]);
        for (auto low = 0; low < receiverSize; low += bucket1) {

            auto up = low + bucket1 < receiverSize ? low + bucket1 : receiverSize;

            commonAes.ecbEncBlocks(recvSet + low, up - low, randomLocations);

            for (auto i = 0; i < w; ++i) {
                for (auto j = low; j < up; ++j) {
                    memcpy(transLocations[wLeft + i] + j * locationInBytes,
                           (u8 *) (randomLocations + (j - low)) + i * locationInBytes, locationInBytes);
                }
            }
        }
    }
}
/** 计算矩阵D
* 输入:
* 输出:
*     matrixD:
*     */
void PsiReceiver::evaluateMatrixD(const u64& width, const u64& receiverSize, const u64& logHeight, const u64& widthBucket1, const u64& locationInBytes, const u64& heightInBytes, u8 *transLocations[], u8 *matrixD[]) {
    auto shift = (1 << logHeight) - 1;
    u8 **matrixD_2d = (u8 **)matrixD;
    for (auto wLeft = 0; wLeft < width; wLeft += widthBucket1) {
        auto wRight = wLeft + widthBucket1 < width ? wLeft + widthBucket1 : width;
        auto w = wRight - wLeft;

        //////////// Compute matrix D /////////////////////////////////

        for (auto i = 0; i < w; ++i) {
            memset((u8 *)matrixD_2d[i+wLeft], 255, heightInBytes);
        }

        for (auto i = 0; i < w; ++i) {
            for (auto j = 0; j < receiverSize; ++j) {
                auto location = (*(u32 *) (transLocations[wLeft + i] + j * locationInBytes)) & shift;

                matrixD_2d[i + wLeft][location >> 3] &= ~(1 << (location & 7));
            }
        } // 得到的是矩阵D
    }
}
/** 计算矩阵Delta
 * 输入:
 *     width: 用于生成一系列key，构造PRF
 *     widthBucket1:
 *     heightInBytes:
 *     locationInBytes:
 *     logHeight:
 *     receiverSize:
 *     transLocations:
 *     otMessages:
 * 输出:
 *     matrixA:
 *     matrixDelta:
 *     */
void PsiReceiver::evaluateMatrixDelta(const u64& width, const u64& widthBucket1, const u64& heightInBytes, const u64& locationInBytes, const u64& logHeight, const u64& receiverSize, u8 *matrixD[], std::vector<std::array<block, 2>> otMessages, u8* matrixA[], u8 *matrixDelta[]){
    for (auto wLeft = 0; wLeft < width; wLeft += widthBucket1) {
        auto wRight = wLeft + widthBucket1 < width ? wLeft + widthBucket1 : width;
        auto w = wRight - wLeft;

        //////////////// Compute matrix A & Delta ///////////////////////

        for (auto i = 0; i < w; ++i) {
            PRNG prng(otMessages[i + wLeft][0]);
            prng.get(matrixA[i+wLeft], heightInBytes);
            prng.SetSeed(otMessages[i + wLeft][1]);
            prng.get(matrixDelta[i + wLeft], heightInBytes);

            for (auto j = 0; j < heightInBytes; ++j) {
                matrixDelta[i + wLeft][j] ^= matrixA[i+wLeft][j] ^ matrixD[i+wLeft][j];
            }
        }
    }
}
/** 计算H2的输入矩阵 (transposed)
 * 输入:
 *     width: 用于生成一系列key，构造PRF
 *     widthBucket1:
 *     locationInBytes:
 *     logHeight:
 *     receiverSize:
 *     transLocations:
 *     matrixA:
 * 输出:
 *     transHashInputs:
 *     */
void PsiReceiver::evaluateTransHashInput(const u64& width, const u64& widthBucket1, const u64& receiverSize, const u64& locationInBytes, const u64& logHeight, u8 *transLocations[], u8 *matrixA[], u8* transHashInputs[]){
    auto shift = (1 << logHeight) - 1;
    for (auto wLeft = 0; wLeft < width; wLeft += widthBucket1) {

        auto wRight = wLeft + widthBucket1 < width ? wLeft + widthBucket1 : width;
        auto w = wRight - wLeft;

        for (auto i = 0; i < w; ++i) {
            for (auto j = 0; j < receiverSize; ++j) {
                auto location = (*(u32*)(transLocations[wLeft + i] + j * locationInBytes)) & shift;

                transHashInputs[i + wLeft][j >> 3] |= (u8)((bool)(matrixA[i+wLeft][location >> 3] & (1 << (location & 7)))) << (j & 7);
            }
        }
    }
}

/** 计算H2的输出
 * 输入:
 *     width: 用于生成一系列key，构造PRF
 *     receiverSize:
 *     hashLengthInBytes:
 *     bucket2:
 *     widthInBytes:
 *     transHashInputs:
 * 输出:
 *     hashOutput:
 *     */
void PsiReceiver::evaluateHashOutput(const u64& width, const u64& receiverSize, const u64& hashLengthInBytes, const u64& bucket2, const u64& widthInBytes, u8 *transHashInputs[], std::unordered_map<u64, std::vector<std::pair<block, u32>>> &allHashes){
    u8 hashOutput[sizeof(block)];
    RandomOracle H(hashLengthInBytes);
    u8* hashInputs[bucket2];

    for (auto i = 0; i < bucket2; ++i) {
        hashInputs[i] = new u8[widthInBytes];
    }

    for (auto low = 0; low < receiverSize; low += bucket2) {
        auto up = low + bucket2 < receiverSize ? low + bucket2 : receiverSize;

        for (auto j = low; j < up; ++j) {
            memset(hashInputs[j - low], 0, widthInBytes);
        }

        for (auto i = 0; i < width; ++i) {
            for (auto j = low; j < up; ++j) {
                hashInputs[j - low][i >> 3] |= (u8)((bool)(transHashInputs[i][j >> 3] & (1 << (j & 7)))) << (i & 7);
            }
        }

        for (auto j = low; j < up; ++j) {
            H.Reset();
            H.Update(hashInputs[j - low], widthInBytes);
            H.Final(hashOutput);

            allHashes[*(u64*)hashOutput].push_back(std::make_pair(*(block*)hashOutput, j));
        }
    }
    for (auto i = 0; i < bucket2; ++i) {
        delete []hashInputs[i];
    }

}
/** 计算H2的输出
 * 输入:
 *     width: 用于生成一系列key，构造PRF
 *     receiverSize:
 *     hashLengthInBytes:
 *     bucket2:
 *     widthInBytes:
 *     transHashInputs:
 * 输出:
 *     hashOutput:
 *     */
//    void PsiReceiver::computePSI(){
//
//    }

void PsiReceiver::receiverInit(unsigned long int seed, DATA *receiverSet, PARAMETERS pp, unsigned char *transLocations[], unsigned char *matrixD[]){
    block commonSeed = toBlock(seed);
    PRNG commonPrng(commonSeed);
    block commonKey;
    commonPrng.get((u8*)&commonKey, sizeof(block));

    block *tmp = new block[pp.receiverSize];
    std::vector<block> receiverSetVector;
    for (auto i=0; i<pp.receiverSize; i++)
        receiverSetVector.push_back(toBlock(&receiverSet[i].d[0]));

    evaluatePRF(commonKey, receiverSetVector, pp.h1LengthInBytes, pp.receiverSize, tmp);

    //////////// Compute matrix D /////////////////////////////////
    auto height = 1 << pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    auto locationInBytes = (pp.logHeight + 7) / 8;
    auto widthBucket1 = sizeof(block) / locationInBytes;
    auto shift = (1 << pp.logHeight) - 1;
    unsigned long int bucket1 = 1<<8;

    PRNG commonPrng2(commonSeed);
    int keyLen = pp.width%widthBucket1==0 ? pp.width/widthBucket1:pp.width/widthBucket1+1;
    block keys[keyLen];
    for (auto i=0; i*widthBucket1<pp.width; i++)
        commonPrng2.get((u8 *) &keys[i], sizeof(block));

    evaluateTransLocations(keys, pp.width, widthBucket1, pp.receiverSize, bucket1, locationInBytes, tmp, transLocations);


    evaluateMatrixD(pp.width, pp.receiverSize, pp.logHeight, widthBucket1, locationInBytes, heightInBytes, transLocations, matrixD);
    delete [] tmp;

}
void PsiReceiver::receiverEvalDelta(DATA* otMessages[], PARAMETERS pp, unsigned char *matrixD[], unsigned char* matrixA[], unsigned char *matrixDelta[]){
    auto locationInBytes = (pp.logHeight + 7) / 8;
    auto widthBucket1 = sizeof(block) / locationInBytes;
    auto height = 1 << pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    std::vector<std::array<block , 2>> otMessagesVector;
    for (auto i=0; i<pp.width; i++){
        std::array<block, 2> tmp = {toBlock(&otMessages[i][0].d[0]), toBlock(&otMessages[i][1].d[0])};
        otMessagesVector.push_back(tmp);
    }

    evaluateMatrixDelta(pp.width, widthBucket1, heightInBytes, locationInBytes, pp.logHeight, pp.receiverSize, matrixD, otMessagesVector, matrixA, matrixDelta);
}

void PsiReceiver::receiverOutput(PARAMETERS pp, unsigned char* transLocations[], unsigned char* matrixA[], unsigned char* senderOutputs[], unsigned long int senderSize,
                                 std::vector<unsigned char> &psiResIdx){

    auto locationInBytes = (pp.logHeight + 7) / 8;
    auto widthBucket1 = sizeof(block) / locationInBytes;
    auto height = 1 << pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    auto widthInBytes = (pp.width + 7) / 8;
    auto receiverSizeInBytes = (pp.receiverSize + 7) / 8;
    unsigned long int bucket2 = 1<<8;
    ///////////////// Compute hash inputs (transposed) /////////////////////

    u8* transHashInputs[pp.width];
    for (auto i = 0; i < pp.width; ++i) {
        transHashInputs[i] = new u8[receiverSizeInBytes];
        memset(transHashInputs[i], 0, receiverSizeInBytes);
    }

    evaluateTransHashInput(pp.width, widthBucket1, pp.receiverSize, locationInBytes, pp.logHeight, transLocations, matrixA, transHashInputs);

    /////////////////// Compute hash outputs ///////////////////////////
    std::unordered_map<u64, std::vector<std::pair<block, u32>>> allHashes;
    evaluateHashOutput(pp.width, pp.receiverSize, pp.hashLengthInBytes, bucket2, widthInBytes, transHashInputs, allHashes);

    ///////////////// Receive hash outputs from sender and compute PSI ///////////////////
    auto psi = 0;
    if (senderSize > ((u64)1<<30))
    {
        return;
    }
    psiResIdx.assign(pp.receiverSize/8+1, 0);
    for (auto low = 0; low < senderSize; low += bucket2) {
        auto up = low + bucket2 < senderSize ? low + bucket2 : senderSize;


        for (auto idx = 0; idx < up - low; ++idx) {
            u64 mapIdx = *(u64*)(senderOutputs[low/bucket2] + idx * pp.hashLengthInBytes);

            auto found = allHashes.find(mapIdx);
            if (found == allHashes.end()) continue;

            for (auto i = 0; i < found->second.size(); ++i) {
                if (memcmp(&(found->second[i].first), senderOutputs[low/bucket2] + idx * pp.hashLengthInBytes, pp.hashLengthInBytes) == 0) {
                    ++psi;
//                        printf("%d \n", found->second[i].second);
                    *(&psiResIdx[0] + found->second[i].second/8) |= 1<< (found->second[i].second%8);
                    break;
                }
            }
        }
    }
}



void evaluateMatrixD(const u64& width, const u64& receiverSize, const u64& logHeight, const u64& widthBucket1,
                     const u64& locationInBytes, const u64& heightInBytes, u8 *transLocations[], u8 *matrixD[]) {
    auto shift = (1 << logHeight) - 1;
    for (auto wLeft = 0; wLeft < width; wLeft += widthBucket1) {
        auto wRight = wLeft + widthBucket1 < width ? wLeft + widthBucket1 : width;
        auto w = wRight - wLeft;

        //////////// Compute matrix D /////////////////////////////////

        for (auto i = 0; i < w; ++i) {
            memset((u8 *)(matrixD[(i + wLeft)]), 255, heightInBytes);
        }
        for (auto i = 0; i < w; ++i) {
            for (auto j = 0; j < receiverSize; ++j) {
                auto location = (*(u32 *) (transLocations[(wLeft + i)] + j * locationInBytes)) & shift;   // v[i]在小于m的范围
                matrixD[(i + wLeft)][location >> 3] &= ~(1 << (location & 7));     // 按字节存储，每次对一个字节处理
            }
        } // 得到的是矩阵D
    }


}

extern "C"
void receiverEvalDeltaElement(DATA otMessage1, DATA otMessage2, PARAMETERS pp, unsigned char *matrixD_element, unsigned char* matrixA_element, unsigned char *matrixDelta_element){
    auto height = 1 << pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    block ot1, ot2;
    ot1 = data_to_block(otMessage1);
    ot2 = data_to_block(otMessage2);
    PRNG prng(ot1);
    prng.get(matrixA_element, heightInBytes);
    prng.SetSeed(ot2);
    prng.get(matrixDelta_element, heightInBytes);
    for (auto j = 0; j < heightInBytes; ++j) {
        matrixDelta_element[j] ^= matrixA_element[j] ^ matrixD_element[j];
    }
}

extern "C"
unsigned char receiverGetHashInputBit(PARAMETERS pp, unsigned char *locationElement_row, unsigned char *matrixA_row){
    auto shift = (1 << pp.logHeight) - 1;
    auto loc = (*(u32*) locationElement_row) & shift;
    return (bool)(matrixA_row[(loc>>3)] & (1 << (loc & 7)));
}

extern "C"
void receiverHashOutput(PARAMETERS pp, unsigned char *hashInputWidth, unsigned char *hashOutput){
    auto widthInBytes = (pp.width+7)/8;
    RandomOracle H(pp.hashLengthInBytes);
    unsigned char hashInput[widthInBytes];
    memset(hashInput, 0, widthInBytes);
    for (auto i=0; i<pp.width; i++){
        hashInput[i>>3] |= (u8) ((bool)hashInputWidth[i]) << (i & 7);
    }
    H.Reset();
    H.Update(hashInput, widthInBytes);
    H.Final(hashOutput);
}

extern "C"
void receiverOutputElement(PARAMETERS pp, unsigned char *locationElement, unsigned char* matrixA[], unsigned char* hashOutput){
    auto widthInBytes = (pp.width+7)/8;
    auto shift = (1 << pp.logHeight) - 1;
    auto height = 1<<pp.logHeight;
    auto locationInBytes = (pp.logHeight + 7) / 8;
    auto heightInBytes = ((height + 7) / 8);

    unsigned char hashInput[widthInBytes];
    RandomOracle H(pp.hashLengthInBytes);
    memset(hashInput, 0, widthInBytes);

    for (auto i=0; i<pp.width; i++){
        auto loc = (*(u32*) (locationElement + i*locationInBytes)) & shift;
        hashInput[i>>3] |= (u8) ((bool)(matrixA[i][(loc>>3)] & (1 << (loc & 7)))) << (i & 7);
    }

    H.Reset();
    H.Update(hashInput, widthInBytes);
    H.Final(hashOutput);

}

extern "C"
void receiverOutputs(PARAMETERS pp, unsigned char *locations[], unsigned char *matrixA[],
                     unsigned char *hashOutputs[]){
    auto widthInBytes = (pp.width+7)/8;
    auto shift = (1 << pp.logHeight) - 1;
    auto locationInBytes = (pp.logHeight + 7) / 8;
    auto height = 1<<pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    unsigned char hashInput[widthInBytes];
    unsigned char hashOutput[pp.hashLengthInBytes];
    RandomOracle H(pp.hashLengthInBytes);

    for (auto i=0; i<pp.receiverSize; i++){
        memset(hashInput, 0, widthInBytes);
        for (auto j=0; j<pp.width; j++){
            auto loc = (*(u32*) (locations[j] + i*locationInBytes)) & shift;
            hashInput[j>>3] |= (u8) ((bool)(*(matrixA[j] + (loc>>3)) & (1 << (loc & 7)))) << (j & 7);
        }
        H.Reset();
        H.Update(hashInput, widthInBytes);
        H.Final(hashOutput);
        memcpy(hashOutputs[i], hashOutput, pp.hashLengthInBytes);
    }
}

extern "C"
void receiverEvaluateLocationElement(unsigned long int seed, PARAMETERS pp, DATA receiverData, unsigned char *locationElement){
    auto height = 1 << pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    auto locationInBytes = (pp.logHeight + 7) / 8;
    auto widthBucket1 = sizeof(block) / locationInBytes;
    auto shift = (1 << pp.logHeight) - 1;
    auto length2d = locationInBytes;

    block commonSeed = toBlock(seed);
    PRNG commonPrng(commonSeed);
    int keyLen = pp.width%widthBucket1==0 ? pp.width/widthBucket1:pp.width/widthBucket1+1;
    block keys[keyLen];
    for (auto i=0; i*widthBucket1<pp.width; i++)
        commonPrng.get((u8 *) &keys[i], sizeof(block));
    block k0 = keys[0];    // 暂时令k0 = k1

    ///////////////// Evaluate G_k0(x) /////////////////
    block recvData;
    AES commonAes;
    commonAes.setKey(k0);
    block aesInput;
    block aesOutput;

    RandomOracle H1(pp.h1LengthInBytes);
    u8 h1Output[pp.h1LengthInBytes];

    H1.Reset();
    H1.Update((u8 *)&receiverData.d[0], sizeof(block));
    H1.Final(h1Output);

    aesInput = *(block*)h1Output;
    recvData = *(block*)(h1Output + sizeof(block));

    commonAes.ecbEncBlocks(&aesInput, 1, &aesOutput);
    recvData ^= aesOutput;

    ///////////////// Evaluate F_k(x) /////////////////
    block randomLocation;

    for (auto wLeft = 0; wLeft < pp.width; wLeft += widthBucket1) {
        auto wRight = wLeft + widthBucket1 < pp.width ? wLeft + widthBucket1 : pp.width;
        auto w = wRight - wLeft;
        commonAes.setKey(keys[wLeft / widthBucket1]);
        commonAes.ecbEncBlocks(&recvData, 1, &randomLocation);
        for (auto i = 0; i < w; ++i) {
            memcpy((locationElement + (wLeft + i)*length2d),
                   (u8 *) (&randomLocation) + i * locationInBytes, locationInBytes);
        }
    }
}
extern "C"
void receiverEvalMatrxD(PARAMETERS pp, unsigned char *transLocations[], unsigned char *matrixD[]){
    auto height = 1 << pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    auto locationInBytes = (pp.logHeight + 7) / 8;
    auto widthBucket1 = sizeof(block) / locationInBytes;
    auto shift = (1 << pp.logHeight) - 1;
    unsigned long int bucket1 = 1<<8;
    evaluateMatrixD(pp.width, pp.receiverSize, pp.logHeight, widthBucket1, locationInBytes, heightInBytes, transLocations, matrixD);

}


