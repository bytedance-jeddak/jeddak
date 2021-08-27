#!/bin/env python
# coding=UTF-8

from ctypes import *
import random
import string
import binascii
import os
import time

_file = 'libecc.so'
_path = os.path.join(*(os.path.split(__file__)[:-1] + (_file,)))
ecc = cdll.LoadLibrary(_path)

BIG_256_56 = c_longlong * 5


class FP_25519(Structure):
    _fields_ = [("g", BIG_256_56),
                ("XES", c_int)]


class ECP_ED25519(Structure):
    _fields_ = [
        ("x", FP_25519),
        ("y", FP_25519),
        ("z", FP_25519)
    ]


class csprng(Structure):
    _fields_ = [
        ("ira", c_uint * 21),
        ("rndptr", c_int),
        ("borrow", c_uint),
        ("pool_ptr", c_int),
        ("pool", c_char * 32)
    ]


class octet(Structure):
    _fields_ = [
        ("len", c_int),
        ("max", c_int),
        ("val", c_char_p)
    ]


class hash256(Structure):
    _fields_ = [
        ("length", c_uint * 2),
        ("h", c_uint * 8),
        ("w", c_uint * 80),
        ("hlen", c_int)
    ]


class NP01OTSender:
    def __init__(self):
        self.Cr = ECP_ED25519()
        self.C = ECP_ED25519()
        self.r = BIG_256_56()
        self.gr = ECP_ED25519()
        self.g = ECP_ED25519()
        rng = csprng()
        q = BIG_256_56()
        raw = create_string_buffer(100)
        RAW = octet(100, 100, cast(raw, c_char_p))
        rd = BIG_256_56()
        for i in range(0, 100):
            raw[i] = random.randint(0, 255)
        ecc.BIG_256_56_rcopy(q, ecc.CURVE_Order_ED25519)
        ecc.CREATE_CSPRNG(pointer(rng), pointer(RAW))
        ecc.BIG_256_56_randomnum(self.r, q, pointer(rng))
        ecc.BIG_256_56_randomnum(rd, q, pointer(rng))

        self.g = ECP_ED25519()
        ecc.ECP_ED25519_generator(pointer(self.g))
        ecc.ECP_ED25519_copy(pointer(self.C), pointer(self.g))
        ecc.ECP_ED25519_mul(pointer(self.C), rd)

        ecc.ECP_ED25519_copy(pointer(self.Cr), pointer(self.C))
        ecc.ECP_ED25519_mul(pointer(self.Cr), self.r)

        ecc.ECP_ED25519_copy(pointer(self.gr), pointer(self.g))
        ecc.ECP_ED25519_mul(pointer(self.gr), self.r)

    def calEnc(self, PK0, M0, M1):
        PK0r = ECP_ED25519()
        PK1r = ECP_ED25519()
        hashPK0r = create_string_buffer(32)
        hashPK1r = create_string_buffer(32)
        pk0rstring = create_string_buffer(33)
        PK0rS = octet(0, 33, cast(pk0rstring, c_char_p))
        pk1rstring = create_string_buffer(33)
        PK1rS = octet(0, 33, cast(pk1rstring, c_char_p))

        ecc.ECP_ED25519_copy(pointer(PK0r), pointer(PK0))
        ecc.ECP_ED25519_mul(pointer(PK0r), self.r)

        ecc.ECP_ED25519_copy(pointer(PK1r), pointer(self.Cr))
        ecc.ECP_ED25519_sub(pointer(PK1r), pointer(PK0r))

        ecc.ECP_ED25519_toOctet(pointer(PK0rS), pointer(PK0r), c_bool(True))
        ecc.ECP_ED25519_toOctet(pointer(PK1rS), pointer(PK1r), c_bool(True))

        sh256 = hash256()
        ecc.HASH256_init(pointer(sh256))
        for i in range(0, 33):
            ecc.HASH256_process(pointer(sh256), int.from_bytes(pk0rstring[i], byteorder='big', signed=False))
        ecc.HASH256_process(pointer(sh256), c_int(0))
        ecc.HASH256_hash(pointer(sh256), hashPK0r)

        ecc.HASH256_init(pointer(sh256))
        for i in range(0, 33):
            ecc.HASH256_process(pointer(sh256), int.from_bytes(pk1rstring[i], byteorder='big', signed=False))
        ecc.HASH256_process(pointer(sh256), c_int(1))
        ecc.HASH256_hash(pointer(sh256), hashPK1r)

        # M0长度=M1长度
        C0 = list()
        C1 = list()
        M0 = str(M0)
        M1 = str(M1)
        str_len = max(len(M0), len(M1))
        M0 = "0" * (str_len - len(M0)) + M0
        M1 = "0" * (str_len - len(M1)) + M1
        for i in range(0, len(M0)):
            C0.append(int.from_bytes(hashPK0r[i % 32], byteorder='big', signed=False) ^ ord(str(M0)[i]))
            C1.append(int.from_bytes(hashPK1r[i % 32], byteorder='big', signed=False) ^ ord(str(M1)[i]))

        return C0, C1


class NP01OTReceiver:
    def __init__(self, C, g, num):
        self.C = C
        self.g = g
        self.k = list()
        self.PK0 = list()
        self.PK1 = list()
        rng = csprng()
        q = BIG_256_56()
        raw = create_string_buffer(100)
        RAW = octet(100, 100, cast(raw, c_char_p))
        for i in range(0, 100):
            raw[i] = random.randint(0, 255)
        ecc.BIG_256_56_rcopy(q, ecc.CURVE_Order_ED25519)
        ecc.CREATE_CSPRNG(pointer(rng), pointer(RAW))

        k = BIG_256_56()
        PK0 = ECP_ED25519()
        PK1 = ECP_ED25519()
        for i in range(0, num):
            ecc.BIG_256_56_randomnum(k, q, pointer(rng))

            ecc.ECP_ED25519_copy(pointer(PK0), pointer(g))
            ecc.ECP_ED25519_mul(pointer(PK0), k)

            ecc.ECP_ED25519_copy(pointer(PK1), pointer(C))
            ecc.ECP_ED25519_sub(pointer(PK1), pointer(PK0))
            self.k.append(k)
            self.PK0.append(PK0)
            self.PK1.append(PK1)

    def dec(self, gr, choice, C0, C1, index):
        grk = ECP_ED25519()
        ecc.ECP_ED25519_copy(pointer(grk), pointer(gr))
        ecc.ECP_ED25519_mul(pointer(grk), self.k[index])
        M = ""

        grkstring = create_string_buffer(33)
        grkS = octet(0, 33, cast(grkstring, c_char_p))
        ecc.ECP_ED25519_toOctet(pointer(grkS), pointer(grk), c_bool(True))

        sh256 = hash256()
        hashgrk = create_string_buffer(32)
        ecc.HASH256_init(pointer(sh256))
        for i in range(0, 33):
            ecc.HASH256_process(pointer(sh256), int.from_bytes(grkstring[i], byteorder='big', signed=False))
        if choice == 0:
            ecc.HASH256_process(pointer(sh256), c_int(0))
            ecc.HASH256_hash(pointer(sh256), hashgrk)
            for i in range(0, len(C0)):
                M = M + (chr(C0[i] ^ int.from_bytes(hashgrk[i % 32], byteorder='big', signed=False)))
        elif choice == 1:
            ecc.HASH256_process(pointer(sh256), c_int(1))
            ecc.HASH256_hash(pointer(sh256), hashgrk)
            for i in range(0, len(C0)):
                M = M + (chr(C1[i] ^ int.from_bytes(hashgrk[i % 32], byteorder='big', signed=False)))
        return M

