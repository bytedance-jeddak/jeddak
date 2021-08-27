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

/* ECDH/ECIES/ECDSA Functions - see main program below */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "ecdh_ED25519.h"

/* Calculate a public/private EC GF(p) key pair. W=S.G mod EC(p),
 * where S is the secret key and W is the public key
 * and G is fixed generator.
 * If RNG is NULL then the private key is provided externally in S
 * otherwise it is generated randomly internally */
int ECP_ED25519_KEY_PAIR_GENERATE(csprng *RNG, octet* S, octet *W)
{
    BIG_256_56 r, s;
    ECP_ED25519 G;
    int res = 0;

    ECP_ED25519_generator(&G);

    BIG_256_56_rcopy(r, CURVE_Order_ED25519);
    if (RNG != NULL)
    {
        BIG_256_56_randomnum(s, r, RNG);
    }
    else
    {
        BIG_256_56_fromBytes(s, S->val);
        BIG_256_56_mod(s, r);
    }

#ifdef AES_S
    BIG_256_56_mod2m(s, 2 * AES_S);
#endif

    S->len = EGS_ED25519;
    BIG_256_56_toBytes(S->val, s);

    ECP_ED25519_mul(&G, s);

    ECP_ED25519_toOctet(W, &G, false); /* To use point compression on public keys, change to true */

    return res;
}

/* Validate public key */
int ECP_ED25519_PUBLIC_KEY_VALIDATE(octet *W)
{
    BIG_256_56 q, r, k;
    ECP_ED25519 WP;
    int valid, nb;
    int res = 0;

    BIG_256_56_rcopy(q, Modulus_25519);
    BIG_256_56_rcopy(r, CURVE_Order_ED25519);

    valid = ECP_ED25519_fromOctet(&WP, W);
    if (!valid) res = ECDH_INVALID_PUBLIC_KEY;

    if (res == 0)
    {
        /* Check point is not in wrong group */
        nb = BIG_256_56_nbits(q);
        BIG_256_56_one(k);
        BIG_256_56_shl(k, (nb + 4) / 2);
        BIG_256_56_add(k, q, k);
        BIG_256_56_sdiv(k, r); /* get co-factor */

        while (BIG_256_56_parity(k) == 0)
        {
            ECP_ED25519_dbl(&WP);
            BIG_256_56_fshr(k, 1);
        }

        if (!BIG_256_56_isunity(k)) ECP_ED25519_mul(&WP, k);
        if (ECP_ED25519_isinf(&WP)) res = ECDH_INVALID_PUBLIC_KEY;
    }

    return res;
}

/* IEEE-1363 Diffie-Hellman online calculation Z=S.WD */
int ECP_ED25519_SVDP_DH(octet *S, octet *WD, octet *Z)
{
    BIG_256_56 r, s, wx;
    int valid;
    ECP_ED25519 W;
    int res = 0;

    BIG_256_56_fromBytes(s, S->val);

    valid = ECP_ED25519_fromOctet(&W, WD);

    if (!valid) res = ECDH_ERROR;
    if (res == 0)
    {
        BIG_256_56_rcopy(r, CURVE_Order_ED25519);
        BIG_256_56_mod(s, r);

        ECP_ED25519_mul(&W, s);
        if (ECP_ED25519_isinf(&W)) res = ECDH_ERROR;
        else
        {
#if CURVETYPE_ED25519!=MONTGOMERY
            ECP_ED25519_get(wx, wx, &W);
#else
            ECP_ED25519_get(wx, &W);
#endif
            Z->len = MODBYTES_256_56;
            BIG_256_56_toBytes(Z->val, wx);
        }
    }
    return res;
}

#if CURVETYPE_ED25519!=MONTGOMERY

/* IEEE ECDSA Signature, C and D are signature on F using private key S */
int ECP_ED25519_SP_DSA(int sha, csprng *RNG, octet *K, octet *S, octet *F, octet *C, octet *D)
{
    char h[128];
    octet H = {0, sizeof(h), h};

    BIG_256_56 r, s, f, c, d, u, vx, w;
    ECP_ED25519 G, V;

    ehashit(sha, F, -1, NULL, &H, sha);

    ECP_ED25519_generator(&G);

    BIG_256_56_rcopy(r, CURVE_Order_ED25519);

    BIG_256_56_fromBytes(s, S->val);

    int hlen = H.len;
    if (H.len > MODBYTES_256_56) hlen = MODBYTES_256_56;
    BIG_256_56_fromBytesLen(f, H.val, hlen);

    if (RNG != NULL)
    {
        do
        {
            BIG_256_56_randomnum(u, r, RNG);
            BIG_256_56_randomnum(w, r, RNG); /* side channel masking */

#ifdef AES_S
            BIG_256_56_mod2m(u, 2 * AES_S);
#endif
            ECP_ED25519_copy(&V, &G);
            ECP_ED25519_mul(&V, u);

            ECP_ED25519_get(vx, vx, &V);

            BIG_256_56_copy(c, vx);
            BIG_256_56_mod(c, r);
            if (BIG_256_56_iszilch(c)) continue;

            BIG_256_56_modmul(u, u, w, r);

            BIG_256_56_invmodp(u, u, r);
            BIG_256_56_modmul(d, s, c, r);

            BIG_256_56_add(d, f, d);

            BIG_256_56_modmul(d, d, w, r);

            BIG_256_56_modmul(d, u, d, r);
        }
        while (BIG_256_56_iszilch(d));
    }
    else
    {
        BIG_256_56_fromBytes(u, K->val);
        BIG_256_56_mod(u, r);

#ifdef AES_S
        BIG_256_56_mod2m(u, 2 * AES_S);
#endif
        ECP_ED25519_copy(&V, &G);
        ECP_ED25519_mul(&V, u);

        ECP_ED25519_get(vx, vx, &V);

        BIG_256_56_copy(c, vx);
        BIG_256_56_mod(c, r);
        if (BIG_256_56_iszilch(c)) return ECDH_ERROR;


        BIG_256_56_invmodp(u, u, r);
        BIG_256_56_modmul(d, s, c, r);

        BIG_256_56_add(d, f, d);

        BIG_256_56_modmul(d, u, d, r);
        if (BIG_256_56_iszilch(d)) return ECDH_ERROR;
    }

    C->len = D->len = EGS_ED25519;

    BIG_256_56_toBytes(C->val, c);
    BIG_256_56_toBytes(D->val, d);

    return 0;
}

/* IEEE1363 ECDSA Signature Verification. Signature C and D on F is verified using public key W */
int ECP_ED25519_VP_DSA(int sha, octet *W, octet *F, octet *C, octet *D)
{
    char h[128];
    octet H = {0, sizeof(h), h};

    BIG_256_56 r, f, c, d, h2;
    int res = 0;
    ECP_ED25519 G, WP;
    int valid;

    ehashit(sha, F, -1, NULL, &H, sha);

    ECP_ED25519_generator(&G);

    BIG_256_56_rcopy(r, CURVE_Order_ED25519);

    OCT_shl(C, C->len - MODBYTES_256_56);
    OCT_shl(D, D->len - MODBYTES_256_56);

    BIG_256_56_fromBytes(c, C->val);
    BIG_256_56_fromBytes(d, D->val);

    int hlen = H.len;
    if (hlen > MODBYTES_256_56) hlen = MODBYTES_256_56;

    BIG_256_56_fromBytesLen(f, H.val, hlen);

    //BIG_fromBytes(f,H.val);

    if (BIG_256_56_iszilch(c) || BIG_256_56_comp(c, r) >= 0 || BIG_256_56_iszilch(d) || BIG_256_56_comp(d, r) >= 0)
        res = ECDH_INVALID;

    if (res == 0)
    {
        BIG_256_56_invmodp(d, d, r);
        BIG_256_56_modmul(f, f, d, r);
        BIG_256_56_modmul(h2, c, d, r);

        valid = ECP_ED25519_fromOctet(&WP, W);

        if (!valid) res = ECDH_ERROR;
        else
        {
            ECP_ED25519_mul2(&WP, &G, h2, f);

            if (ECP_ED25519_isinf(&WP)) res = ECDH_INVALID;
            else
            {
                ECP_ED25519_get(d, d, &WP);
                BIG_256_56_mod(d, r);
                if (BIG_256_56_comp(d, c) != 0) res = ECDH_INVALID;
            }
        }
    }

    return res;
}

/* IEEE1363 ECIES encryption. Encryption of plaintext M uses public key W and produces ciphertext V,C,T */
void ECP_ED25519_ECIES_ENCRYPT(int sha, octet *P1, octet *P2, csprng *RNG, octet *W, octet *M, int tlen, octet *V, octet *C, octet *T)
{

    int i, len;
    char z[EFS_ED25519], vz[3 * EFS_ED25519 + 1], k[2 * AESKEY_ED25519], k1[AESKEY_ED25519], k2[AESKEY_ED25519], l2[8], u[EFS_ED25519];
    octet Z = {0, sizeof(z), z};
    octet VZ = {0, sizeof(vz), vz};
    octet K = {0, sizeof(k), k};
    octet K1 = {0, sizeof(k1), k1};
    octet K2 = {0, sizeof(k2), k2};
    octet L2 = {0, sizeof(l2), l2};
    octet U = {0, sizeof(u), u};

    if (ECP_ED25519_KEY_PAIR_GENERATE(RNG, &U, V) != 0) return;
    if (ECP_ED25519_SVDP_DH(&U, W, &Z) != 0) return;

    OCT_copy(&VZ, V);
    OCT_joctet(&VZ, &Z);

    KDF2(sha, &VZ, P1, 2 * AESKEY_ED25519, &K);

    K1.len = K2.len = AESKEY_ED25519;
    for (i = 0; i < AESKEY_ED25519; i++)
    {
        K1.val[i] = K.val[i];
        K2.val[i] = K.val[AESKEY_ED25519 + i];
    }

    AES_CBC_IV0_ENCRYPT(&K1, M, C);

    OCT_jint(&L2, P2->len, 8);

    len = C->len;
    OCT_joctet(C, P2);
    OCT_joctet(C, &L2);
    HMAC(sha, C, &K2, tlen, T);
    C->len = len;
}

/* IEEE1363 ECIES decryption. Decryption of ciphertext V,C,T using private key U outputs plaintext M */
int ECP_ED25519_ECIES_DECRYPT(int sha, octet *P1, octet *P2, octet *V, octet *C, octet *T, octet *U, octet *M)
{

    int i, len;
    char z[EFS_ED25519], vz[3 * EFS_ED25519 + 1], k[2 * AESKEY_ED25519], k1[AESKEY_ED25519], k2[AESKEY_ED25519], l2[8], tag[32];
    octet Z = {0, sizeof(z), z};
    octet VZ = {0, sizeof(vz), vz};
    octet K = {0, sizeof(k), k};
    octet K1 = {0, sizeof(k1), k1};
    octet K2 = {0, sizeof(k2), k2};
    octet L2 = {0, sizeof(l2), l2};
    octet TAG = {0, sizeof(tag), tag};

    if (ECP_ED25519_SVDP_DH(U, V, &Z) != 0) return 0;

    OCT_copy(&VZ, V);
    OCT_joctet(&VZ, &Z);

    KDF2(sha, &VZ, P1, 2 * AESKEY_ED25519, &K);

    K1.len = K2.len = AESKEY_ED25519;
    for (i = 0; i < AESKEY_ED25519; i++)
    {
        K1.val[i] = K.val[i];
        K2.val[i] = K.val[AESKEY_ED25519 + i];
    }

    if (!AES_CBC_IV0_DECRYPT(&K1, C, M)) return 0;

    OCT_jint(&L2, P2->len, 8);

    len = C->len;
    OCT_joctet(C, P2);
    OCT_joctet(C, &L2);
    HMAC(sha, C, &K2, T->len, &TAG);
    C->len = len;

    if (!OCT_ncomp(T, &TAG, T->len)) return 0;

    return 1;

}

#endif
