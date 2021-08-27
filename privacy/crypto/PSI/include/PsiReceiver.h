#pragma once

#include "Defines.h"
#include "utils.h"
#include "AES.h"
#include "PRNG.h"
#include "Blake.h"
#include <vector>
#include <unordered_map>
#include "array"
#include "RandomOracle.h"


class PsiReceiver {
private:
    void evaluatePRF(block key, std::vector<block>& receiverSet, const u64& h1LengthInBytes, const u64& receiverSize, block* recvSet);
    void evaluateTransLocations(block keys[], const u64& width, const u64& widthBucket1, const u64& receiverSize, const u64& bucket1, const u64& locationInBytes, block* recvSet, u8* transLocations[]);
    void evaluateMatrixD(const u64& width, const u64& receiverSize, const u64& logHeight, const u64& widthBucket1, const u64& locationInBytes, const u64& heightInBytes, u8 *transLocations[], u8 *matrixD[]);
    void evaluateMatrixDelta(const u64& width, const u64& widthBucket1, const u64& heightInBytes, const u64& locationInBytes, const u64& logHeight, const u64& receiverSize, u8 *matrixD[], std::vector<std::array<block, 2>> otMessages, u8* matrixA[], u8 *matrixDelta[]);
    void evaluateTransHashInput(const u64& width, const u64& widthBucket1, const u64& receiverSize, const u64& locationInBytes, const u64& logHeight, u8 *transLocations[], u8 *matrixA[], u8* transHashInputs[]);
    void evaluateHashOutput(const u64& width, const u64& receiverSize, const u64& hashLengthInBytes, const u64& bucket2, const u64& widthInBytes, u8 *transHashInputs[], std::unordered_map<u64, std::vector<std::pair<block, u32>>> &allHashes);

public:
    PsiReceiver() {}
    void receiverInit(unsigned long int seed, DATA *receiverSet, PARAMETERS pp, unsigned char *transLocations[], unsigned char *matrixD[]);
    void receiverEvalDelta(DATA* otMessages[], PARAMETERS pp, unsigned char *matrixD[], unsigned char* matrixA[], unsigned char *matrixDelta[]);
    void receiverOutput(PARAMETERS pp, unsigned char* transLocations[], unsigned char* matrixA[], unsigned char* senderOutputs[], unsigned long int senderSize,
                        std::vector<unsigned char> &psiResIdx);
};


void evaluateMatrixD(const u64& width, const u64& receiverSize, const u64& logHeight, const u64& widthBucket1,
                     const u64& locationInBytes, const u64& heightInBytes, u8 *transLocations[], u8 *matrixD[]);

/** Parallelized Interface */
// locationElement: width * locationInBytes
extern "C"
void receiverEvaluateLocationElement(unsigned long int seed, PARAMETERS pp, DATA receiverData,
                                     unsigned char *locationElement);
// transLocations: width * (pp.senderSize * locationInBytes)
// matrixD: width * heightInBytes
extern "C"
void receiverEvalMatrxD(PARAMETERS pp, unsigned char *transLocations[], unsigned char *matrixD[]);

extern "C"
void receiverEvalDeltaElement(DATA otMessage1, DATA otMessage2, PARAMETERS pp, unsigned char *matrixD_element,
                              unsigned char *matrixA_element, unsigned char *matrixDelta_element);

extern "C"
unsigned char receiverGetHashInputBit(PARAMETERS pp, unsigned char *locationElement_row, unsigned char *matrixA_row);

extern "C"
void receiverHashOutput(PARAMETERS pp, unsigned char *hashInputWidth, unsigned char *hashOutput);

// locationElement: width * locationInBytes
// matrixA: width * heightInBytes
extern "C"
void receiverOutputElement(PARAMETERS pp, unsigned char *locationElement, unsigned char *matrixA[],
                           unsigned char *hashOutput);

extern "C"
void receiverOutputs(PARAMETERS pp, unsigned char *locations[], unsigned char *matrixA[],
                   unsigned char *hashOutputs[]);
