
#include "iknpOTeSender.h"

void OTeSender::comp_y(block *y0, block *y1, block *s, block *k_s, block *u_cols[]){
    auto row_num = (ot_num+127) - (ot_num+127) % 128;
    auto block_num = (ot_num+127)/128;
    block *q_rows = new block[row_num];
    memset(q_rows, 0, 16 * row_num);

    block *q_cols[128];
    for (auto & q_col : q_cols){
        q_col = new block[block_num];
        memset(q_col, 0, 16 * block_num);
    }

    PRNG prng;
    for (auto i=0; i<128; i++){
        bool s_i = *((u8 *)(s) + i/8) & (1<<(i%8));
        prng.SetSeed(k_s[i]);
        prng.get(q_cols[i], block_num);

        if(s_i) {
            for (auto j = 0; j < block_num; j++) {
                q_cols[i][j] = q_cols[i][j] ^ u_cols[i][j];
            }
        }
    }
    block tmp[128];
    for (auto i=0; i<block_num; i++){
        for (auto j=0; j<128; j++){
            tmp[j] = q_cols[j][i];
        }
        sse_trans((u8 *)tmp, (u8 *)(q_rows + i*128), 128, 128);
    }
    for (auto & q_col : q_cols){
        delete [] q_col;
    }
    if (IKNP_SHA){
        auto hashLengthInBytes = 16;
        RandomOracle H(hashLengthInBytes);
        for (auto i=0; i<row_num; i++){
            H.Reset();
            H.Update(i);
            H.Update(q_rows[i]);
            H.Final(y0[i]);
            y0[i] = y0[i] ^ x0[i];

            H.Reset();
            H.Update(i);
            H.Update(q_rows[i]^(*s));
            H.Final(y1[i]);
            y1[i] = y1[i] ^ x1[i];

        }
    }else if (IKNP_AES){
        AES fixedKeyAES;
        fixedKeyAES.setKey(aes_key);
        for (auto i=0; i<ot_num; i++){
            block input = q_rows[i] ^ toBlock(i);
            fixedKeyAES.ecbEncBlock(input, y0[i]);
            y0[i] = y0[i] ^ x0[i];

            input = input^(*s);
            fixedKeyAES.ecbEncBlock(input, y1[i]);
            y1[i] = y1[i] ^ x1[i];
        }
    }


    delete [] q_rows;

}