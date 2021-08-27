//
// Created by fsecond on 2021/5/25.
//

#ifndef OBLIVIOUS_TRANSFER_IKNPOTESENDER_H
#define OBLIVIOUS_TRANSFER_IKNPOTESENDER_H
#include "Defines.h"
#include "PRNG.h"
#include "RandomOracle.h"
#include "utils.h"
#include "AES.h"
class OTeSender{
public:
    std::vector<block> x0;
    std::vector<block> x1;
    int ot_num;
    block aes_key;
    OTeSender(block *m0, block *m1, int length, u64 key = 0){
        x0 = std::vector<block>(m0, m0+length);
        x1 = std::vector<block>(m1, m1+length);
        ot_num = length;
        aes_key = toBlock(key);
    }

    void comp_y(block *y0, block *y1, block *s, block *k_s, block *u_cols[]);

};

#endif //OBLIVIOUS_TRANSFER_IKNPOTESENDER_H
