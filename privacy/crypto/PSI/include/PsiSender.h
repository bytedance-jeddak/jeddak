#pragma once

#include "Defines.h"
#include "AES.h"
#include "PRNG.h"
#include "utils.h"
#include "Blake.h"
#include <vector>
#include "RandomOracle.h"


class PsiSender {
private:
    /** Basic Operation */
    void evaluatePRF(block key, std::vector<block>& senderSet, const u64& h1LengthInBytes, const u64& senderSize, block* sendSet);
    void evaluateTransLocations(block keys[], const u64& width, const u64& widthBucket1, const u64& senderSize, const u64& bucket1, const u64& locationInBytes, block* sendSet, u8* transLocations[]);
    void evaluateTransHashInput(const u64& width, const u64& widthBucket1, const u64& senderSize, const u64& locationInBytes, const u64& logHeight, u8 *transLocations[], u8 *matrixA[], u8* transHashInputs[]);
    void evaluateMatrixC(u8* matrixC[], int w, const u64& heightInBytes, u8 **recvMatrix, std::vector<block> otMessages, std::vector<u8> choices);
    void evaluateHashOutputs(u8** transHashInputs, const u64& hashLengthInBytes, const u64& bucket2, const u64& senderSize, const u64& widthInBytes, const u64& width, u8** hashOutputs);
public:
    PsiSender() {}
    /** Interface */
    void senderInit(unsigned long int seed, DATA *senderSet, PARAMETERS pp, unsigned char *transLocations[]);
    void senderOutput(unsigned long int seed, unsigned char *transLocations[], DATA *otMessages, unsigned char *choices, unsigned char *matrixDelta[], PARAMETERS pp, unsigned char *hashOutputs[]);
};


/** Parallelized Interface */
extern "C"
void senderEvaluateLocationElement(unsigned long int seed, PARAMETERS pp, DATA receiverData,
                             unsigned char *locationElement);

extern "C"
void senderEvalMatrxCElement(PARAMETERS pp, DATA otMessage, unsigned char choice,
                             unsigned char *matrixDelta_element, unsigned char *matrixC_element);

extern "C"
void senderOutputElement(PARAMETERS pp, unsigned char *locationElement, unsigned char *matrixC[],
                         unsigned char *hashOutput);

extern "C"
void senderOutputs(PARAMETERS pp, unsigned char *locations[], unsigned char *matrixC[],
                         unsigned char *hashOutputs[]);
