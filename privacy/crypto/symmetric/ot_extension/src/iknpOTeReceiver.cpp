//
// Created by fsecond on 2021/5/25.
//

#include "iknpOTeReceiver.h"

void OTeReceiver::comp_trans_matrix_u(block *u_cols[], block *t_rows, block *k0, block *k1){
    PRNG prng;
    auto block_num = (ot_num+127)/128;
    block *t0[128];
    block tmp[128];
    for (auto & t0_col : t0){
        t0_col = new block [block_num];
        memset(t0_col, 0, 16 * block_num);
    }

    block *t1[128];
    for (auto & t1_col : t1){
        t1_col = new block [block_num];
        memset(t1_col, 0, 16 * block_num);
    }

    for (auto i=0; i<128; i++){
        prng.SetSeed(k0[i]);
        prng.get(t0[i], block_num);
        prng.SetSeed(k1[i]);
        prng.get(t1[i], block_num);
        for (auto j=0; j<block_num; j++) {
            *(u_cols[i] + j) = t0[i][j] ^ t1[i][j] ^ compress_choices[j];
        }
    }
    for (auto i=0; i<block_num; i++){
        for (auto j=0; j<128; j++){
            tmp[j] = t0[j][i];
        }
        sse_trans((u8 *)tmp, (u8 *)(t_rows + i*128), 128, 128);
    }
    for (auto & t0_col : t0){
        delete [] t0_col;
    }
    for (auto & t1_col : t1){
        delete [] t1_col;
    }

}

void OTeReceiver::output(block *output, block *t_rows,  block *y0, block *y1){
    if (IKNP_SHA){
        auto hashLengthInBytes = 128/8;
        RandomOracle H(hashLengthInBytes);
        for (auto i=0; i<ot_num; i++){
            bool c = *((u8 *)(&compress_choices[i/128]) + (i%128)/8) & (1 << (i&7));
            H.Reset();
            H.Update(i);
            H.Update(t_rows[i]);
            H.Final(output[i]);
            if (!c){
                output[i]= output[i]^y0[i];
            } else{
                output[i]= output[i]^y1[i];
            }
        }
    }else if (IKNP_AES){
        AES fixedKeyAES;
        fixedKeyAES.setKey(aes_key);
        for (auto i=0; i<ot_num; i++){
            bool c = *((u8 *)(&compress_choices[i/128]) + (i%128)/8) & (1 << (i&7));
            block input = t_rows[i] ^ toBlock(i);
            fixedKeyAES.ecbEncBlock(input, output[i]);
            if (!c){
                output[i]= output[i]^y0[i];
            } else{
                output[i]= output[i]^y1[i];
            }
        }
    }
}

void compress_choices(block *compress_choices, std::vector<u8> choices, int ot_num){
    int compress_length = (ot_num+127)/128;
    u8 block_tmp[16];
    for (auto i=0; i<compress_length; i++){
        memset(block_tmp, 0, 16);
        for (auto j=0; j<128 && j+i*128 < ot_num; j++){
            block_tmp[j/8] |= (choices[j+i*128] << (j%8));
        }
        *((block *)compress_choices + i) = toBlock(block_tmp);
    }
}