import numpy as np

from privacy.crypto.cryptosystem import Ciphertext


class LinearAlgebra:
    @staticmethod
    def pdot(v, u):
        """
        Plaintext inner product
        :param v: numpy.ndarray
        :param u: numpy.ndarray
        :return:
        """
        assert len(v) == len(u)

        if len(v) == 0 or len(u) == 0:
            return np.array([])
        else:
            return v.dot(u)

    @staticmethod
    def cmult(a, v):
        """
        Ciphertext scalar product a * v
        :param a: Ciphertext
        :param v: numpy.ndarray or list
        :return: list[Ciphertext]
        """
        if isinstance(a, Ciphertext):
            product = [0 for _ in range(len(v))]
            for i in range(len(product)):
                product[i] = a * v[i]
            return product
        else:
            raise TypeError("invalid scalar type: {}".format(type(a)))

    @staticmethod
    def cadd(v, u):
        """
        Ciphertext vector addition v + u
        :param v: list[Ciphertext]
        :param u: list[Ciphertext]
        :return: list[Ciphertext]
        """
        summation = [0 for _ in range(len(v))]

        for i in range(len(summation)):
            summation[i] = v[i] + u[i]

        return summation

    @staticmethod
    def cneg(v):
        """
        Ciphertext negative -v
        :param v:
        :return:
        """
        for i in range(len(v)):
            v[i] = -v[i]
        return v
