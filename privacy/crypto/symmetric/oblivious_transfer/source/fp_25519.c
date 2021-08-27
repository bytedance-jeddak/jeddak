/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/

/* AMCL mod p functions */
/* Small Finite Field arithmetic */
/* SU=m, SU is Stack Usage (NOT_SPECIAL Modulus) */

#include "fp_25519.h"

/* Fast Modular Reduction Methods */

/* r=d mod m */
/* d MUST be normalised */
/* Products must be less than pR in all cases !!! */
/* So when multiplying two numbers, their product *must* be less than MODBITS+BASEBITS*NLEN */
/* Results *may* be one bit bigger than MODBITS */

#if MODTYPE_25519 == PSEUDO_MERSENNE
/* r=d mod m */

/* Converts from BIG integer to residue form mod Modulus */
void FP_25519_nres(FP_25519 *y,BIG_256_56 x)
{
    BIG_256_56_copy(y->g,x);
    y->XES=1;
}

/* Converts from residue form back to BIG integer form */
void FP_25519_redc(BIG_256_56 x,FP_25519 *y)
{
    BIG_256_56_copy(x,y->g);
}

/* reduce a DBIG to a BIG exploiting the special form of the modulus */
void FP_25519_mod(BIG_256_56 r,DBIG_256_56 d)
{
    BIG_256_56 t,b;
    chunk v,tw;
    BIG_256_56_split(t,b,d,MODBITS_25519);

    /* Note that all of the excess gets pushed into t. So if squaring a value with a 4-bit excess, this results in
       t getting all 8 bits of the excess product! So products must be less than pR which is Montgomery compatible */

    if (MConst_25519 < NEXCESS_256_56)
    {
        BIG_256_56_imul(t,t,MConst_25519);
        BIG_256_56_norm(t);
        BIG_256_56_add(r,t,b);
        BIG_256_56_norm(r);
        tw=r[NLEN_256_56-1];
        r[NLEN_256_56-1]&=TMASK_25519;
        r[0]+=MConst_25519*((tw>>TBITS_25519));
    }
    else
    {
        v=BIG_256_56_pmul(t,t,MConst_25519);
        BIG_256_56_add(r,t,b);
        BIG_256_56_norm(r);
        tw=r[NLEN_256_56-1];
        r[NLEN_256_56-1]&=TMASK_25519;
#if CHUNK == 16
        r[1]+=muladd_256_56(MConst_25519,((tw>>TBITS_25519)+(v<<(BASEBITS_256_56-TBITS_25519))),0,&r[0]);
#else
        r[0]+=MConst_25519*((tw>>TBITS_25519)+(v<<(BASEBITS_256_56-TBITS_25519)));
#endif
    }
    BIG_256_56_norm(r);
}
#endif

/* This only applies to Curve C448, so specialised (for now) */
#if MODTYPE_25519 == GENERALISED_MERSENNE

void FP_25519_nres(FP_25519 *y,BIG_256_56 x)
{
    BIG_256_56_copy(y->g,x);
    y->XES=1;
}

/* Converts from residue form back to BIG integer form */
void FP_25519_redc(BIG_256_56 x,FP_25519 *y)
{
    BIG_256_56_copy(x,y->g);
}

/* reduce a DBIG to a BIG exploiting the special form of the modulus */
void FP_25519_mod(BIG_256_56 r,DBIG_256_56 d)
{
    BIG_256_56 t,b;
    chunk carry;
    BIG_256_56_split(t,b,d,MBITS_25519);

    BIG_256_56_add(r,t,b);

    BIG_256_56_dscopy(d,t);
    BIG_256_56_dshl(d,MBITS_25519/2);

    BIG_256_56_split(t,b,d,MBITS_25519);

    BIG_256_56_add(r,r,t);
    BIG_256_56_add(r,r,b);
    BIG_256_56_norm(r);
    BIG_256_56_shl(t,MBITS_25519/2);

    BIG_256_56_add(r,r,t);

    carry=r[NLEN_256_56-1]>>TBITS_25519;

    r[NLEN_256_56-1]&=TMASK_25519;
    r[0]+=carry;

    r[224/BASEBITS_256_56]+=carry<<(224%BASEBITS_256_56); /* need to check that this falls mid-word */
    BIG_256_56_norm(r);
}

#endif

#if MODTYPE_25519 == MONTGOMERY_FRIENDLY

/* convert to Montgomery n-residue form */
void FP_25519_nres(FP_25519 *y,BIG_256_56 x)
{
    DBIG_256_56 d;
    BIG_256_56 r;
    BIG_256_56_rcopy(r,R2modp_25519);
    BIG_256_56_mul(d,x,r);
    FP_25519_mod(y->g,d);
    y->XES=2;
}

/* convert back to regular form */
void FP_25519_redc(BIG_256_56 x,FP_25519 *y)
{
    DBIG_256_56 d;
    BIG_256_56_dzero(d);
    BIG_256_56_dscopy(d,y->g);
    FP_25519_mod(x,d);
}

/* fast modular reduction from DBIG to BIG exploiting special form of the modulus */
void FP_25519_mod(BIG_256_56 a,DBIG_256_56 d)
{
    int i;

    for (i=0; i<NLEN_256_56; i++)
        d[NLEN_256_56+i]+=muladd_256_56(d[i],MConst_25519-1,d[i],&d[NLEN_256_56+i-1]);

    BIG_256_56_sducopy(a,d);
    BIG_256_56_norm(a);
}

#endif

#if MODTYPE_25519 == NOT_SPECIAL

/* convert to Montgomery n-residue form */
void FP_25519_nres(FP_25519 *y,BIG_256_56 x)
{
    DBIG_256_56 d;
    BIG_256_56 r;
    BIG_256_56_rcopy(r,R2modp_25519);
    BIG_256_56_mul(d,x,r);
    FP_25519_mod(y->g,d);
    y->XES=2;
}

/* convert back to regular form */
void FP_25519_redc(BIG_256_56 x,FP_25519 *y)
{
    DBIG_256_56 d;
    BIG_256_56_dzero(d);
    BIG_256_56_dscopy(d,y->g);
    FP_25519_mod(x,d);
}


/* reduce a DBIG to a BIG using Montgomery's no trial division method */
/* d is expected to be dnormed before entry */
/* SU= 112 */
void FP_25519_mod(BIG_256_56 a,DBIG_256_56 d)
{
    BIG_256_56 mdls;
    BIG_256_56_rcopy(mdls,Modulus_25519);
    BIG_256_56_monty(a,mdls,MConst_25519,d);
}

#endif

/* test x==0 ? */
/* SU= 48 */
int FP_25519_iszilch(FP_25519 *x)
{
    BIG_256_56 m,t;
    BIG_256_56_rcopy(m,Modulus_25519);
    BIG_256_56_copy(t,x->g);
    BIG_256_56_mod(t,m);
    return BIG_256_56_iszilch(t);
}

void FP_25519_copy(FP_25519 *y,FP_25519 *x)
{
    BIG_256_56_copy(y->g,x->g);
    y->XES=x->XES;
}

void FP_25519_rcopy(FP_25519 *y, const BIG_256_56 c)
{
    BIG_256_56 b;
    BIG_256_56_rcopy(b,c);
    FP_25519_nres(y,b);
}

/* Swap a and b if d=1 */
void FP_25519_cswap(FP_25519 *a,FP_25519 *b,int d)
{
    sign32 t,c=d;
    BIG_256_56_cswap(a->g,b->g,d);

    c=~(c-1);
    t=c&((a->XES)^(b->XES));
    a->XES^=t;
    b->XES^=t;

}

/* Move b to a if d=1 */
void FP_25519_cmove(FP_25519 *a,FP_25519 *b,int d)
{
    sign32 c=-d;

    BIG_256_56_cmove(a->g,b->g,d);
    a->XES^=(a->XES^b->XES)&c;
}

void FP_25519_zero(FP_25519 *x)
{
    BIG_256_56_zero(x->g);
    x->XES=1;
}

int FP_25519_equals(FP_25519 *x,FP_25519 *y)
{
    FP_25519 xg,yg;
    FP_25519_copy(&xg,x);
    FP_25519_copy(&yg,y);
    FP_25519_reduce(&xg);
    FP_25519_reduce(&yg);
    if (BIG_256_56_comp(xg.g,yg.g)==0) return 1;
    return 0;
}

/* output FP */
/* SU= 48 */
void FP_25519_output(FP_25519 *r)
{
    BIG_256_56 c;
    FP_25519_redc(c,r);
    BIG_256_56_output(c);
}

void FP_25519_rawoutput(FP_25519 *r)
{
    BIG_256_56_rawoutput(r->g);
}

#ifdef GET_STATS
int tsqr=0,rsqr=0,tmul=0,rmul=0;
int tadd=0,radd=0,tneg=0,rneg=0;
int tdadd=0,rdadd=0,tdneg=0,rdneg=0;
#endif

#ifdef FUSED_MODMUL

/* Insert fastest code here */

#endif

/* r=a*b mod Modulus */
/* product must be less that p.R - and we need to know this in advance! */
/* SU= 88 */
void FP_25519_mul(FP_25519 *r,FP_25519 *a,FP_25519 *b)
{
    DBIG_256_56 d;

    if ((sign64)a->XES*b->XES>(sign64)FEXCESS_25519)
    {
#ifdef DEBUG_REDUCE
        printf("Product too large - reducing it\n");
#endif
        FP_25519_reduce(a);  /* it is sufficient to fully reduce just one of them < p */
    }

#ifdef FUSED_MODMUL
    FP_25519_modmul(r->g,a->g,b->g);
#else
    BIG_256_56_mul(d,a->g,b->g);
    FP_25519_mod(r->g,d);
#endif
    r->XES=2;
}


/* multiplication by an integer, r=a*c */
/* SU= 136 */
void FP_25519_imul(FP_25519 *r,FP_25519 *a,int c)
{
    int s=0;

    if (c<0)
    {
        c=-c;
        s=1;
    }

#if MODTYPE_25519==PSEUDO_MERSENNE || MODTYPE_25519==GENERALISED_MERSENNE
    DBIG_256_56 d;
    BIG_256_56_pxmul(d,a->g,c);
    FP_25519_mod(r->g,d);
    r->XES=2;

#else
    //Montgomery
    BIG_256_56 k;
    FP_25519 f;
    if (a->XES*c<=FEXCESS_25519)
    {
        BIG_256_56_pmul(r->g,a->g,c);
        r->XES=a->XES*c;    // careful here - XES jumps!
    }
    else
    {
        // don't want to do this - only a problem for Montgomery modulus and larger constants
        BIG_256_56_zero(k);
        BIG_256_56_inc(k,c);
        BIG_256_56_norm(k);
        FP_25519_nres(&f,k);
        FP_25519_mul(r,a,&f);
    }
#endif

    if (s)
    {
        FP_25519_neg(r,r);
        FP_25519_norm(r);
    }
}

/* Set r=a^2 mod m */
/* SU= 88 */
void FP_25519_sqr(FP_25519 *r,FP_25519 *a)
{
    DBIG_256_56 d;

    if ((sign64)a->XES*a->XES>(sign64)FEXCESS_25519)
    {
#ifdef DEBUG_REDUCE
        printf("Product too large - reducing it\n");
#endif
        FP_25519_reduce(a);
    }

    BIG_256_56_sqr(d,a->g);
    FP_25519_mod(r->g,d);
    r->XES=2;
}

/* SU= 16 */
/* Set r=a+b */
void FP_25519_add(FP_25519 *r,FP_25519 *a,FP_25519 *b)
{
    BIG_256_56_add(r->g,a->g,b->g);
    r->XES=a->XES+b->XES;
    if (r->XES>FEXCESS_25519)
    {
#ifdef DEBUG_REDUCE
        printf("Sum too large - reducing it \n");
#endif
        FP_25519_reduce(r);
    }
}

/* Set r=a-b mod m */
/* SU= 56 */
void FP_25519_sub(FP_25519 *r,FP_25519 *a,FP_25519 *b)
{
    FP_25519 n;
    FP_25519_neg(&n,b);
    FP_25519_add(r,a,&n);
}

// https://graphics.stanford.edu/~seander/bithacks.html
// constant time log to base 2 (or number of bits in)

static int logb2(unsign32 v)
{
    int r;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;

    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    r = (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    return r;
}

// find appoximation to quotient of a/m
// Out by at most 2.
// Note that MAXXES is bounded to be 2-bits less than half a word
static int quo(BIG_256_56 n,BIG_256_56 m)
{
    int sh;
    chunk num,den;
    int hb=CHUNK/2;
    if (TBITS_25519<hb)
    {
        sh=hb-TBITS_25519;
        num=(n[NLEN_256_56-1]<<sh)|(n[NLEN_256_56-2]>>(BASEBITS_256_56-sh));
        den=(m[NLEN_256_56-1]<<sh)|(m[NLEN_256_56-2]>>(BASEBITS_256_56-sh));
    }
    else
    {
        num=n[NLEN_256_56-1];
        den=m[NLEN_256_56-1];
    }
    return (int)(num/(den+1));
}

/* SU= 48 */
/* Fully reduce a mod Modulus */
void FP_25519_reduce(FP_25519 *a)
{
    BIG_256_56 m,r;
    int sr,sb,q;
    chunk carry;

    BIG_256_56_rcopy(m,Modulus_25519);

    BIG_256_56_norm(a->g);

    if (a->XES>16)
    {
        q=quo(a->g,m);
        carry=BIG_256_56_pmul(r,m,q);
        r[NLEN_256_56-1]+=(carry<<BASEBITS_256_56); // correction - put any carry out back in again
        BIG_256_56_sub(a->g,a->g,r);
        BIG_256_56_norm(a->g);
        sb=2;
    }
    else sb=logb2(a->XES-1);  // sb does not depend on the actual data

    BIG_256_56_fshl(m,sb);

    while (sb>0)
    {
// constant time...
        sr=BIG_256_56_ssn(r,a->g,m);  // optimized combined shift, subtract and norm
        BIG_256_56_cmove(a->g,r,1-sr);
        sb--;
    }

    //BIG_256_56_mod(a->g,m);
    a->XES=1;
}

void FP_25519_norm(FP_25519 *x)
{
    BIG_256_56_norm(x->g);
}

/* Set r=-a mod Modulus */
/* SU= 64 */
void FP_25519_neg(FP_25519 *r,FP_25519 *a)
{
    int sb;
    BIG_256_56 m;

    BIG_256_56_rcopy(m,Modulus_25519);

    sb=logb2(a->XES-1);
    BIG_256_56_fshl(m,sb);
    BIG_256_56_sub(r->g,m,a->g);
    r->XES=((sign32)1<<sb)+1;

    if (r->XES>FEXCESS_25519)
    {
#ifdef DEBUG_REDUCE
        printf("Negation too large -  reducing it \n");
#endif
        FP_25519_reduce(r);
    }

}

/* Set r=a/2. */
/* SU= 56 */
void FP_25519_div2(FP_25519 *r,FP_25519 *a)
{
    BIG_256_56 m;
    BIG_256_56_rcopy(m,Modulus_25519);
    FP_25519_copy(r,a);

    if (BIG_256_56_parity(a->g)==0)
    {

        BIG_256_56_fshr(r->g,1);
    }
    else
    {
        BIG_256_56_add(r->g,r->g,m);
        BIG_256_56_norm(r->g);
        BIG_256_56_fshr(r->g,1);
    }
}

#if MODTYPE_25519 == PSEUDO_MERSENNE || MODTYPE_25519==GENERALISED_MERSENNE

// See eprint paper https://eprint.iacr.org/2018/1038
// If p=3 mod 4 r= x^{(p-3)/4}, if p=5 mod 8 r=x^{(p-5)/8}

static void FP_25519_fpow(FP_25519 *r,FP_25519 *x)
{
    int i,j,k,bw,w,c,nw,lo,m,n;
    FP_25519 xp[11],t,key;
    const int ac[]= {1,2,3,6,12,15,30,60,120,240,255};
// phase 1
    FP_25519_copy(&xp[0],x);	// 1
    FP_25519_sqr(&xp[1],x); // 2
    FP_25519_mul(&xp[2],&xp[1],x);  //3
    FP_25519_sqr(&xp[3],&xp[2]);  // 6
    FP_25519_sqr(&xp[4],&xp[3]); // 12
    FP_25519_mul(&xp[5],&xp[4],&xp[2]); // 15
    FP_25519_sqr(&xp[6],&xp[5]); // 30
    FP_25519_sqr(&xp[7],&xp[6]); // 60
    FP_25519_sqr(&xp[8],&xp[7]); // 120
    FP_25519_sqr(&xp[9],&xp[8]); // 240
    FP_25519_mul(&xp[10],&xp[9],&xp[5]); // 255

#if MODTYPE_25519==PSEUDO_MERSENNE
    n=MODBITS_25519;
#endif
#if MODTYPE_25519==GENERALISED_MERSENNE  // Goldilocks ONLY
    n=MODBITS_25519/2;
#endif

    if (MOD8_25519==5)
    {
        n-=3;
        c=(MConst_25519+5)/8;
    }
    else
    {
        n-=2;
        c=(MConst_25519+3)/4;
    }

    bw=0;
    w=1;
    while (w<c)
    {
        w*=2;
        bw+=1;
    }
    k=w-c;

    if (k!=0)
    {
        i=10;
        while (ac[i]>k) i--;
        FP_25519_copy(&key,&xp[i]);
        k-=ac[i];
    }
    while (k!=0)
    {
        i--;
        if (ac[i]>k) continue;
        FP_25519_mul(&key,&key,&xp[i]);
        k-=ac[i];
    }

// phase 2
    FP_25519_copy(&xp[1],&xp[2]);
    FP_25519_copy(&xp[2],&xp[5]);
    FP_25519_copy(&xp[3],&xp[10]);

    j=3;
    m=8;
    nw=n-bw;
    while (2*m<nw)
    {
        FP_25519_copy(&t,&xp[j++]);
        for (i=0; i<m; i++)
            FP_25519_sqr(&t,&t);
        FP_25519_mul(&xp[j],&xp[j-1],&t);
        m*=2;
    }

    lo=nw-m;
    FP_25519_copy(r,&xp[j]);

    while (lo!=0)
    {
        m/=2;
        j--;
        if (lo<m) continue;
        lo-=m;
        FP_25519_copy(&t,r);
        for (i=0; i<m; i++)
            FP_25519_sqr(&t,&t);
        FP_25519_mul(r,&t,&xp[j]);
    }
// phase 3

    if (bw!=0)
    {
        for (i=0; i<bw; i++ )
            FP_25519_sqr(r,r);
        FP_25519_mul(r,r,&key);
    }
#if MODTYPE_25519==GENERALISED_MERSENNE  // Goldilocks ONLY
    FP_25519_copy(&key,r);
    FP_25519_sqr(&t,&key);
    FP_25519_mul(r,&t,x);
    for (i=0; i<n+1; i++)
        FP_25519_sqr(r,r);
    FP_25519_mul(r,r,&key);
#endif
}

void FP_25519_inv(FP_25519 *r,FP_25519 *x)
{
    FP_25519 y,t;
    FP_25519_fpow(&y,x);
    if (MOD8_25519==5)
    {
        // r=x^3.y^8
        FP_25519_sqr(&t,x);
        FP_25519_mul(&t,&t,x);
        FP_25519_sqr(&y,&y);
        FP_25519_sqr(&y,&y);
        FP_25519_sqr(&y,&y);
        FP_25519_mul(r,&t,&y);
    }
    else
    {
        FP_25519_sqr(&y,&y);
        FP_25519_sqr(&y,&y);
        FP_25519_mul(r,&y,x);
    }
}

#else

void FP_25519_pow(FP_25519 *r,FP_25519 *a,BIG_256_56 b)
{
    sign8 w[1+(NLEN_256_56*BASEBITS_256_56+3)/4];
    FP_25519 tb[16];
    BIG_256_56 t;
    int i,nb;

    FP_25519_norm(a);
    BIG_256_56_norm(b);
    BIG_256_56_copy(t,b);
    nb=1+(BIG_256_56_nbits(t)+3)/4;
    /* convert exponent to 4-bit window */
    for (i=0; i<nb; i++)
    {
        w[i]=BIG_256_56_lastbits(t,4);
        BIG_256_56_dec(t,w[i]);
        BIG_256_56_norm(t);
        BIG_256_56_fshr(t,4);
    }

    FP_25519_one(&tb[0]);
    FP_25519_copy(&tb[1],a);
    for (i=2; i<16; i++)
        FP_25519_mul(&tb[i],&tb[i-1],a);

    FP_25519_copy(r,&tb[w[nb-1]]);
    for (i=nb-2; i>=0; i--)
    {
        FP_25519_sqr(r,r);
        FP_25519_sqr(r,r);
        FP_25519_sqr(r,r);
        FP_25519_sqr(r,r);
        FP_25519_mul(r,r,&tb[w[i]]);
    }
    FP_25519_reduce(r);
}

/* set w=1/x */
void FP_25519_inv(FP_25519 *w,FP_25519 *x)
{

    BIG_256_56 m2;
    BIG_256_56_rcopy(m2,Modulus_25519);
    BIG_256_56_dec(m2,2);
    BIG_256_56_norm(m2);
    FP_25519_pow(w,x,m2);
}
#endif

/* SU=8 */
/* set n=1 */
void FP_25519_one(FP_25519 *n)
{
    BIG_256_56 b;
    BIG_256_56_one(b);
    FP_25519_nres(n,b);
}

/* is r a QR? */
int FP_25519_qr(FP_25519 *r)
{
    int j;
    BIG_256_56 m;
    BIG_256_56 b;
    BIG_256_56_rcopy(m,Modulus_25519);
    FP_25519_redc(b,r);
    j=BIG_256_56_jacobi(b,m);
    FP_25519_nres(r,b);
    if (j==1) return 1;
    return 0;

}

/* Set a=sqrt(b) mod Modulus */
/* SU= 160 */
void FP_25519_sqrt(FP_25519 *r,FP_25519 *a)
{
    FP_25519 v,i;
    BIG_256_56 b;
    BIG_256_56 m;
    BIG_256_56_rcopy(m,Modulus_25519);
    BIG_256_56_mod(a->g,m);
    BIG_256_56_copy(b,m);
    if (MOD8_25519==5)
    {
        FP_25519_copy(&i,a); // i=x
        BIG_256_56_fshl(i.g,1); // i=2x
#if MODTYPE_25519 == PSEUDO_MERSENNE   || MODTYPE_25519==GENERALISED_MERSENNE
        FP_25519_fpow(&v,&i);
#else
        BIG_256_56_dec(b,5);
        BIG_256_56_norm(b);
        BIG_256_56_fshr(b,3); // (p-5)/8
        FP_25519_pow(&v,&i,b); // v=(2x)^(p-5)/8
#endif
        FP_25519_mul(&i,&i,&v); // i=(2x)^(p+3)/8
        FP_25519_mul(&i,&i,&v); // i=(2x)^(p-1)/4
        BIG_256_56_dec(i.g,1);  // i=(2x)^(p-1)/4 - 1
        FP_25519_mul(r,a,&v);
        FP_25519_mul(r,r,&i);
        FP_25519_reduce(r);
    }
    if (MOD8_25519==3 || MOD8_25519==7)
    {
#if MODTYPE_25519 == PSEUDO_MERSENNE   || MODTYPE_25519==GENERALISED_MERSENNE
        FP_25519_fpow(r,a);
        FP_25519_mul(r,r,a);
#else
        BIG_256_56_inc(b,1);
        BIG_256_56_norm(b);
        BIG_256_56_fshr(b,2); /* (p+1)/4 */
        FP_25519_pow(r,a,b);
#endif
    }
}
