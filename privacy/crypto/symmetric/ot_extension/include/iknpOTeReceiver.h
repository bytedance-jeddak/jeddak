//
// Created by fsecond on 2021/5/25.
//

#ifndef OBLIVIOUS_TRANSFER_IKNPOTE_H
#define OBLIVIOUS_TRANSFER_IKNPOTE_H
#include "Defines.h"
#include "PRNG.h"
#include "RandomOracle.h"
#include "utils.h"
class OTeReceiver{
public:
    std::vector<block> compress_choices;
    int ot_num;
    block aes_key;
    OTeReceiver(std::vector<u8> &choices, int ot_length, u64 key = 0){
        ot_num = ot_length;
        int compress_length = (ot_length+127)/128;
        u8 block_tmp[16];
        for (auto i=0; i<compress_length; i++){
            memset(block_tmp, 0, 16);
            for (auto j=0; j<128 && j+i*128 < ot_num; j++){
                block_tmp[j/8] |= (choices[j+i*128] << (j%8));
            }
            compress_choices.push_back(toBlock(block_tmp));
        }
        aes_key = toBlock(key);
    }

    void comp_trans_matrix_u(block *u_cols[], block *t_rows, block *k0, block *k1);

    void output(block *output, block *t_rows, block *y0, block *y1);

};

void compress_choices(block *compress_choices, std::vector<u8> choices, int ot_num);

#endif //OBLIVIOUS_TRANSFER_IKNPOTE_H
