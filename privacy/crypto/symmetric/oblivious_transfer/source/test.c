#include "stdlib.h"
#include "string.h"
#include "ecp_ED25519.h"
#include "amcl.h"
#include "randapi.h"

int main(){
    csprng rng;
    BIG_256_56 q,r;
    char raw[100];
    octet RAW={0,100,raw};
    RAW.len=100;
    RAW.val[0]=1;
    RAW.val[1]=2;
    for (int i = 4; i < 100; i++) RAW.val[i] = i;
    BIG_256_56_rcopy(q,CURVE_Order_ED25519);
    BIG_256_56_output(q);
    printf("\n");
    CREATE_CSPRNG(&rng,&RAW);
    BIG_256_56_randomnum(r,q,&rng);
    BIG_256_56_output(r);
}