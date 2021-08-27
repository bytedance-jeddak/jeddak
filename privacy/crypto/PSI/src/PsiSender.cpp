#include "PsiSender.h"
#include "utils.h"
#include <iostream>
#include <cstdlib>
#include <bitset>

/** PRF第一步计算，计算输入数据集对应的PRF结果，即所有的v （其中PRF用AES加密实现） */
void PsiSender::evaluatePRF(block key, std::vector<block>& senderSet, const u64& h1LengthInBytes, const u64& senderSize, block* sendSet){
    AES commonAes;
    commonAes.setKey(key);
    block* aesInput = new block[senderSize];
    block* aesOutput = new block[senderSize];
    RandomOracle H1(h1LengthInBytes);
    u8 h1Output[h1LengthInBytes];

//        std::cout<<"sender set is: ";
//        block send_tmp = senderSet[0];
//        for (int i=0; i< sizeof(block); i++){
//            u8 zxc = *((u8*)&send_tmp+i);
//            for (int k=7;k>=0;k--)  {
//                std::cout << ((zxc >> k) & 1);
//            }
//            std::cout<<", ";
//        }
//        std::cout<<std::endl;

    for (auto i = 0; i < senderSize; ++i) {
        H1.Reset();
        H1.Update((u8*)(senderSet.data() + i), sizeof(block));
        H1.Final(h1Output);

        aesInput[i] = *(block*)h1Output;
        sendSet[i] = *(block*)(h1Output + sizeof(block));
    }
//        std::cout<<"(sender) aes input is: ";
//        send_tmp = aesInput[0];
//        for (int i=0; i< sizeof(block); i++){
//            u8 zxc = *((u8*)&send_tmp+i);
//            for (int k=7;k>=0;k--)  {
//                std::cout << ((zxc >> k) & 1);
//            }
//            std::cout<<", ";
//        }
//        std::cout<<std::endl;

    commonAes.ecbEncBlocks(aesInput, senderSize, aesOutput);
    for (auto i = 0; i < senderSize; ++i) {
        sendSet[i] ^= aesOutput[i];
    }
    delete [] aesInput;
    delete [] aesOutput;



//        delete []aesInput;
//        delete []aesOutput;
}

/** 计算H2的输入矩阵 (transposed)
 * 输入:
 *     width: 用于生成一系列key，构造PRF
 *     widthBucket1:
 *     locationInBytes:
 *     logHeight:
 *     senderSize:
 *     transLocations:
 *     matrixC:
 * 输出:
 *     transHashInputs:
 *     */
void PsiSender::evaluateTransHashInput(const u64& width, const u64& widthBucket1, const u64& senderSize, const u64& locationInBytes, const u64& logHeight, u8 *transLocations[], u8 *matrixC[], u8* transHashInputs[]){
    auto shift = (1 << logHeight) - 1;
    for (auto wLeft = 0; wLeft < width; wLeft += widthBucket1) {

        auto wRight = wLeft + widthBucket1 < width ? wLeft + widthBucket1 : width;
        auto w = wRight - wLeft;

        for (auto i = 0; i < w; ++i) {
            for (auto j = 0; j < senderSize; ++j) {
                auto location = (*(u32*)(transLocations[wLeft + i] + j * locationInBytes)) & shift;

                transHashInputs[i + wLeft][j >> 3] |= (u8)((bool)(matrixC[i+wLeft][location >> 3] & (1 << (location & 7)))) << (j & 7);
            }
        }
    }
}

/** 计算矩阵C[w][heightInBytes]，输入按 widthBucket1 分片，减小缓存
 *  输入：
 *      w: 矩阵行数
 *      矩阵Delta: recvMatrix[widthBucket1][heightInBytes]
 *      otMessage: OT值，用于生成ri
 *      */
void PsiSender::evaluateMatrixC(u8* matrixC[], int w, const u64& heightInBytes, u8 **recvMatrix, std::vector<block> otMessages, std::vector<u8> choices){
    for (int i=0; i<w; ++i){
        // 初始 C[i] 为OT得到的ri
        PRNG prng(otMessages[i]);
        prng.get(matrixC[i], heightInBytes);
        if (choices[i]) {
            for (auto j = 0; j < heightInBytes; ++j) {
                matrixC[i][j] ^= recvMatrix[i][j];
            }
        }
    }
}
/** 计算H2的值
 *  输入：
 *  transHashInput: u8 hashInputs[bucket2][]
 * */
void PsiSender::evaluateHashOutputs(u8** transHashInputs, const u64& hashLengthInBytes, const u64& bucket2, const u64& senderSize, const u64& widthInBytes, const u64& width, u8** hashOutputs){
    /////////////////// Compute hash outputs ///////////////////////////

    RandomOracle H(hashLengthInBytes);
    u8 hashRes[sizeof(block)];

    u8* hashInputs[bucket2];

    for (auto i = 0; i < bucket2; ++i) {
        hashInputs[i] = new u8[widthInBytes];
    }

    for (auto low = 0, idx =0; low < senderSize; low += bucket2, idx += 1) {
        auto up = low + bucket2 < senderSize ? low + bucket2 : senderSize;

        for (auto j = low; j < up; ++j) {
            memset(hashInputs[j - low], 0, widthInBytes);
        }

        for (auto i = 0; i < width; ++i) {
            for (auto j = low; j < up; ++j) {
                hashInputs[j - low][i >> 3] |= (u8)((bool)(transHashInputs[i][j >> 3] & (1 << (j & 7)))) << (i & 7);
            }
        }
//            hashOutputs[idx] = new u8[bucket2 * hashLengthInBytes];
        for (auto j = low; j < up; ++j) {
            H.Reset();
            H.Update(hashInputs[j - low], widthInBytes);
            H.Final(hashRes);

            memcpy(hashOutputs[idx] + (j - low) * hashLengthInBytes, hashRes, hashLengthInBytes);
        }

    }//计算最终hash输出

    for (auto i = 0; i < bucket2; ++i) {
        delete []hashInputs[i];
    }

}

void PsiSender::evaluateTransLocations(block keys[], const u64& width, const u64& widthBucket1, const u64& senderSize, const u64& bucket1, const u64& locationInBytes, block* recvSet, u8* transLocations[]){
    AES commonAes;
    block commonKey;
    block randomLocations[bucket1];

    for (auto wLeft = 0; wLeft < width; wLeft += widthBucket1) {
        auto wRight = wLeft + widthBucket1 < width ? wLeft + widthBucket1 : width;
        auto w = wRight - wLeft;
        commonAes.setKey(keys[wLeft/widthBucket1]);
        for (auto low = 0; low < senderSize; low += bucket1) {

            auto up = low + bucket1 < senderSize ? low + bucket1 : senderSize;

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

void PsiSender::senderInit(unsigned long int seed, DATA *senderSet, PARAMETERS pp, unsigned char *transLocations[]){
    block commonKey;
    block commonSeed = toBlock(seed);
    PRNG commonPrng(commonSeed);
    commonPrng.get((u8*)&commonKey, sizeof(block));
    std::vector<block> senderSetVector;
    for (auto i=0; i<pp.senderSize; i++)
        senderSetVector.push_back(toBlock(&senderSet[i].d[0]));
    block *tmp = new block[pp.senderSize];
    evaluatePRF(commonKey, senderSetVector, pp.h1LengthInBytes, pp.senderSize, tmp);


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
    evaluateTransLocations(keys, pp.width, widthBucket1, pp.senderSize, bucket1, locationInBytes, tmp, transLocations);
    delete [] tmp;
}

void PsiSender::senderOutput(unsigned long int seed, unsigned char *transLocations[], DATA *otMessages, unsigned char *choices, unsigned char *matrixDelta[], PARAMETERS pp,
                             unsigned char *hashOutputs[]){
    auto height = 1<<pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    auto widthInBytes = (pp.width + 7) / 8;
    auto locationInBytes = (pp.logHeight + 7) / 8;//索引位置需要的字节数
    auto senderSizeInBytes = (pp.senderSize + 7) / 8;
    auto widthBucket1 = sizeof(block) / locationInBytes;//按照128比特分割索引，1个block可以计算几个索引位置
    unsigned int bucket1, bucket2;
    bucket1 = bucket2 = 1 << 8;

    //////////////// Extend OTs and compute matrix C ///////////////////
    u8* matrixC[pp.width];
    for (auto i = 0; i < pp.width; ++i) {
        matrixC[i] = new u8[heightInBytes];//矩阵C   heightInBytes 高度的字节长度表示
    }
    std::vector<block> otMessagesVector;
    std::vector<u8> choicesVector;
    for (int i=0; i<pp.width; i++){
        otMessagesVector.push_back(toBlock(&otMessages[i].d[0]));
        choicesVector.push_back(choices[i]);
    }
    evaluateMatrixC(matrixC, pp.width, heightInBytes, matrixDelta, otMessagesVector, choicesVector);

    ///////////////// Compute hash inputs (transposed) /////////////////////
    u8* transHashInputs[pp.width];
    for (auto i = 0; i < pp.width; ++i) {
        transHashInputs[i] = new u8[senderSizeInBytes];//senderSizeInBytes 发送方的字节长度表示
        memset(transHashInputs[i], 0, senderSizeInBytes);
    }

    evaluateTransHashInput(pp.width, widthBucket1, pp.senderSize, locationInBytes, pp.logHeight, transLocations, matrixC, transHashInputs);

    evaluateHashOutputs(transHashInputs,pp.hashLengthInBytes, bucket2, pp.senderSize, widthInBytes, pp.width, hashOutputs);

    for (auto i = 0; i < pp.width; ++i) {
        delete [] matrixC[i];
        delete [] transHashInputs[i];
    }
}



extern "C"
void senderEvaluateLocationElement(unsigned long int seed, PARAMETERS pp, DATA receiverData, unsigned char *locationElement){
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
void senderEvalMatrxCElement(PARAMETERS pp, DATA otMessage, unsigned char choice, unsigned char *matrixDelta_element, unsigned char *matrixC_element){
    auto height = 1<<pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    PRNG prng(data_to_block(otMessage));
    prng.get(matrixC_element, heightInBytes);
    if (choice) {
        for (auto j = 0; j < heightInBytes; ++j) {
            matrixC_element[j] ^= matrixDelta_element[j];
        }
    }
}

extern "C"
void senderOutputElement(PARAMETERS pp, unsigned char *locationElement, unsigned char *matrixC[],
                         unsigned char *hashOutput){
    auto widthInBytes = (pp.width+7)/8;
    auto shift = (1 << pp.logHeight) - 1;
    auto locationInBytes = (pp.logHeight + 7) / 8;
    auto height = 1<<pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    unsigned char hashInput[widthInBytes];
    RandomOracle H(pp.hashLengthInBytes);
    memset(hashInput, 0, widthInBytes);

    for (auto i=0; i<pp.width; i++){
        auto loc = (*(u32*) (locationElement + i*locationInBytes)) & shift;
        hashInput[i>>3] |= (u8) ((bool)(*(matrixC[i] + (loc>>3)) & (1 << (loc & 7)))) << (i & 7);
    }

    H.Reset();
    H.Update(hashInput, widthInBytes);
    H.Final(hashOutput);
}

extern "C"
void senderOutputs(PARAMETERS pp, unsigned char *locations[], unsigned char *matrixC[],
                   unsigned char *hashOutputs[]){
    auto widthInBytes = (pp.width+7)/8;
    auto shift = (1 << pp.logHeight) - 1;
    auto locationInBytes = (pp.logHeight + 7) / 8;
    auto height = 1<<pp.logHeight;
    auto heightInBytes = (height + 7) / 8;
    unsigned char hashInput[widthInBytes];
    unsigned char hashOutput[pp.hashLengthInBytes];
    RandomOracle H(pp.hashLengthInBytes);

    for (auto i=0; i<pp.senderSize; i++){
        memset(hashInput, 0, widthInBytes);
        for (auto j=0; j<pp.width; j++){
            auto loc = (*(u32*) (locations[j] + i*locationInBytes)) & shift;
            hashInput[j>>3] |= (u8) ((bool)(*(matrixC[j] + (loc>>3)) & (1 << (loc & 7)))) << (j & 7);
        }
        H.Reset();
        H.Update(hashInput, widthInBytes);
        H.Final(hashOutput);
        memcpy(hashOutputs[i], hashOutput, pp.hashLengthInBytes);
    }
}

