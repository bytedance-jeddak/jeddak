import os
import numpy as np
from ctypes import *
from privacy.crypto.symmetric.ot_extension import iknpOTe
from privacy.crypto.symmetric.oblivious_transfer.NP01 import NP01OTReceiver, NP01OTSender

_file = '_iknpOTe.so'
_path = os.path.join(*(os.path.split(__file__)[:-1] + (_file,)))
_IKNP = cdll.LoadLibrary(_path)

u64_Max = 0xFFFFFFFFFFFFFFFF


def _to_block_array(numpy_arr, length):
    c_arr = iknpOTe.initBlockArray(length)
    c_arr_high = iknpOTe.u64Vector(length)
    c_arr_high[:] = (numpy_arr[:] >> 64).tolist()
    c_arr_low = iknpOTe.u64Vector(length)
    c_arr_low[:] = (numpy_arr[:] & u64_Max).tolist()
    iknpOTe.setBlockArray(c_arr, c_arr_high, c_arr_low, length)
    return c_arr


def _to_2d_block_array(numpy_arr, length1, length2):
    c_arr = iknpOTe.initBlock2DArray(length1, length2)
    c_high = iknpOTe.u64Vector(length2)
    c_low = iknpOTe.u64Vector(length2)
    for i in range(length1):
        c_high[:] = (numpy_arr[i][:] >> 64).tolist()
        c_low[:] = (numpy_arr[i][:] & u64_Max).tolist()
        iknpOTe.setBlockArray(c_arr, i, c_high, c_low, length2)
    return c_arr


def _get_block_array(block_arr, length):
    c_row_high = iknpOTe.u64Vector(length)
    c_row_low = iknpOTe.u64Vector(length)
    iknpOTe.getBlockArray(c_row_high, c_row_low, block_arr, length)
    high = np.zeros(length, dtype=np.uint64)
    low = np.zeros(length, dtype=np.uint64)
    iknpOTe.getNumpyArray(high, c_row_high)
    iknpOTe.getNumpyArray(low, c_row_low)
    numpy_arr = (high.astype(object) << 64) + low.astype(object)
    return numpy_arr


def _get_2d_block_array(block_arr, length1, length2):
    numpy_arr = np.zeros([length1, length2], dtype=object)
    c_row_high = iknpOTe.u64Vector(length2)
    c_row_low = iknpOTe.u64Vector(length2)
    high = np.zeros(length2, dtype=np.uint64)
    low = np.zeros(length2, dtype=np.uint64)
    for i in range(128):
        iknpOTe.get2DArrayRow(c_row_high, c_row_low, block_arr, i, length2)
        iknpOTe.getNumpyArray(high, c_row_high)
        iknpOTe.getNumpyArray(low, c_row_low)
        numpy_arr[i][:] = (high.astype(object) << 64) + low.astype(object)
    return numpy_arr


def _free_block_array(block_arr):
    iknpOTe.deleteBlockArray(block_arr)


def _free_2d_block_array(block_arr, length):
    iknpOTe.deleteBlock2DArray(block_arr, length)


class IKNP_Receiver:
    def __init__(self, length, choices, key=0):
        """
        :param length: ot num, at least 128 and power of 2
        :param choices: numpy array
        :param key: AES key for hashing
        """
        if length < 128 or (length - 1)&length !=0:
            raise ValueError("OT length should be at least 128 and power of 2!")
        if len(choices) != length:
            raise ValueError("choices length is not equal with OT length")
        self.length = length
        c_choices = iknpOTe.uncharVector(self.length)
        c_choices[:] = choices.tolist()
        self.recv = iknpOTe.OTeReceiver(c_choices, self.length, key)
        self._base_messages = IKNP_Receiver.gen_random_messages(128)
        self._baseOtSender = NP01OTSender()

    def run(self, messenger, party):
        """
        :param messenger: Messenger
        :param party: str
        """
        pub_info = self.get_base_pub_info()
        messenger.send(pub_info, parties=[party], tag="base ot knowledge")
        pk_list = messenger.receive(parties=[party], tag="base ot pk")[0]
        base_cipher = self._encrypt_base_ot(pk_list)
        messenger.send(base_cipher, parties=[party], tag="base ot ciphertext")
        u_cols, t_rows = self._pre_comp()
        messenger.send(u_cols, parties=[party], tag="ot extension matrix u")
        y0, y1 = messenger.receive(parties=[party], tag="ot extension y0, y1")[0]
        recvOtMessages = self._output(t_rows, y0, y1)
        return recvOtMessages

    @property
    def base_messages(self):
        return self._base_messages
    @staticmethod
    def gen_random_choices(num):
        choices = np.random.randint(0, 2, num, dtype=np.uint8)
        return choices

    @staticmethod
    def gen_random_messages(num):
        messages0 = (np.random.randint(0, pow(2, 64), num, dtype=np.uint64).astype(dtype=object) << 64) + \
                    np.random.randint(0, pow(2, 64), num, dtype=np.uint64).astype(dtype=object)
        messages1 = (np.random.randint(0, pow(2, 64), num, dtype=np.uint64).astype(dtype=object) << 64) + \
                    np.random.randint(0, pow(2, 64), num, dtype=np.uint64).astype(dtype=object)
        return messages0, messages1

    def get_base_pub_info(self):
        return self._baseOtSender.C, self._baseOtSender.g, self._baseOtSender.gr

    def _encrypt_base_ot(self, pk_list):
        """
        :param pkList: public key received from sender
        :return : base OT public knowledge and cipher list
        """
        base_cipher = []
        for i in range(0, 128):
            PK = pk_list[i]
            C0base, C1base = self._baseOtSender.calEnc(
                PK, self._base_messages[0][i], self._base_messages[1][i]
            )
            base_cipher.append((C0base, C1base))
        return base_cipher

    def _pre_comp(self, k0=None, k1=None):
        """
        :param k0: numpy array[int], 128 blocks, base ot messages
        :param k1: numpy array[int], 128 blocks, base ot messages
        :return u_cols, t_rows: numpy array, numpy array
        """
        if not (k0 and k1):
            k0, k1 = self._base_messages
        block_num = int((self.length+127)/128)

        c_u_cols = iknpOTe.initBlock2DArray(128, block_num)
        c_t_rows = iknpOTe.initBlockArray(self.length)
        c_k0 = _to_block_array(k0, 128)
        c_k1 = _to_block_array(k1, 128)

        self.recv.comp_trans_matrix_u(c_u_cols, c_t_rows, c_k0, c_k1)

        u_cols = _get_2d_block_array(c_u_cols, 128, block_num)
        t_rows = _get_block_array(c_t_rows, self.length)

        _free_2d_block_array(c_u_cols, 128)
        _free_block_array(c_t_rows)
        _free_block_array(c_k0)
        _free_block_array(c_k1)
        return u_cols, t_rows

    def _output(self, t_rows, y0, y1):
        """
        :param t_rows: numpy array[int], self.length blocks
        :param y0: numpy array[int], self.length blocks
        :param y1: numpy array[int], self.length blocks
        :return output: numpy array[int], self.length blocks
        """
        c_output = iknpOTe.initBlockArray(self.length)
        c_t_rows = _to_block_array(t_rows, self.length)
        c_y0 = _to_block_array(y0, self.length)
        c_y1 = _to_block_array(y1, self.length)
        self.recv.output(c_output, c_t_rows, c_y0, c_y1)
        _free_block_array(c_t_rows)
        _free_block_array(c_y0)
        _free_block_array(c_y1)
        output = _get_block_array(c_output, self.length)
        return output


class IKNP_Sender:
    def __init__(self, length, m0, m1, key=0):
        """
        :param length: ot num, at least 128 and power of 2
        :param m0: numpy array
        :param m1: numpy array
        :param key: AES key
        """
        if length < 128 or (length - 1)&length !=0:
            raise ValueError("OT length should be at least 128 and power of 2!")
        self.length = length
        c_m0 = _to_block_array(m0, self.length)
        c_m1 = _to_block_array(m1, self.length)
        self.sender = iknpOTe.OTeSender(c_m0, c_m1, self.length, key)
        _free_block_array(c_m0)
        _free_block_array(c_m1)
        self._public_info = None
        self._baseOtReceiver = None
        self._base_choices = IKNP_Sender.gen_random_choices(128)
        self._base_message = np.zeros(128, dtype=object)

    def run(self, messenger, party):
        """
        :param messenger: Messenger
        :param party: str
        """
        pub_info = messenger.receive(parties=[party], tag="base ot knowledge")[0]
        pk_list = self._gen_base_ot_key(pub_info)
        messenger.send(pk_list, parties=[party], tag="base ot pk")
        base_cipher = messenger.receive(parties=[party], tag="base ot ciphertext")[0]
        self._decrypt_base_ot(base_cipher)
        u_cols = messenger.receive(parties=[party], tag="ot extension matrix u")[0]
        y0, y1 = self._pre_comp(u_cols)
        messenger.send((y0, y1), parties=[party], tag="ot extension y0, y1")

    @property
    def base_choices(self):
        return self._base_choices

    @staticmethod
    def gen_random_choices(num):
        choices = np.random.randint(0, 2, num, dtype=np.uint8)
        return choices

    @staticmethod
    def gen_random_messages(num):
        messages0 = (np.random.randint(0, pow(2, 64), num, dtype=np.uint64).astype(dtype=object) << 64) + \
                    np.random.randint(0, pow(2, 64), num, dtype=np.uint64).astype(dtype=object)
        messages1 = (np.random.randint(0, pow(2, 64), num, dtype=np.uint64).astype(dtype=object) << 64) + \
                    np.random.randint(0, pow(2, 64), num, dtype=np.uint64).astype(dtype=object)
        return messages0, messages1

    def _gen_base_ot_key(self, public_info):
        self._public_info = public_info
        self._baseOtReceiver = NP01OTReceiver(public_info[0], public_info[1], 128)
        pkList = []
        for i in range(128):
            pkList.append(self._baseOtReceiver.PK1[i] if self._base_choices[i] else self._baseOtReceiver.PK0[i])
        return pkList

    def _decrypt_base_ot(self, base_cipher):
        for i in range(0, 128):
            result = self._baseOtReceiver.dec(self._public_info[2], self._base_choices[i],
                                              base_cipher[i][0], base_cipher[i][1], i)
            self._base_message[i] = int(result)

    def _pre_comp(self, u_cols, base_choices=None, base_message=None):
        """
        :param u_cols: numpy array[int], 128 * block_num
        :param base_choices: numpy array[int], 128 uint8
        :param base_message: numpy array[int], 128 blocks
        :return y0, y1: numpy array[int], numpy array[int]
        """
        if not (base_choices and base_message):
            base_choices = self._base_choices
            base_message = self._base_message
        block_num = int((self.length+127)/128)
        c_base_choices = iknpOTe.initBlockArray(1)
        c_tmp = iknpOTe.uncharVector(128)
        c_tmp[:] = base_choices.tolist()
        iknpOTe.compress_choices(c_base_choices, c_tmp, 128)
        c_base_message = _to_block_array(base_message, 128)
        c_y0 = iknpOTe.initBlockArray(self.length)
        c_y1 = iknpOTe.initBlockArray(self.length)
        c_u_cols = _to_2d_block_array(u_cols, 128, block_num)

        self.sender.comp_y(c_y0, c_y1, c_base_choices, c_base_message, c_u_cols)
        _free_block_array(c_base_choices)
        _free_block_array(c_base_message)
        _free_2d_block_array(c_u_cols, 128)

        y0 = _get_block_array(c_y0, self.length)
        y1 = _get_block_array(c_y1, self.length)

        _free_block_array(c_y0)
        _free_block_array(c_y1)
        return y0, y1

