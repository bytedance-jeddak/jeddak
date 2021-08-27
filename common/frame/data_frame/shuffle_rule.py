import copy
import numpy as np


class ShuffleRule(object):
    """
    Bijective mapping
    """
    def __init__(self, permutation):
        """

        :param permutation: array of mapped indices
        """
        self._injection, self._surjection = {}, {}

        for index in range(len(permutation)):
            self._injection[index] = permutation[index]
            self._surjection[permutation[index]] = index

    @staticmethod
    def generate(perm_length):
        """

        :param perm_length: int
        :return:
        """
        perm = np.random.permutation(perm_length)
        return ShuffleRule(perm)

    def map(self, index):
        return self._injection[index]

    def invert_index(self, mapped_index):
        return self._surjection[mapped_index]

    def map_array(self, arr):
        m_arr = copy.deepcopy(arr)

        for i in range(len(arr)):
            m_arr[i] = arr[self._injection[i]]

        return m_arr

    def invert_array(self, arr):
        inv_arr = copy.deepcopy(arr)

        for i in range(len(arr)):
            inv_arr[i] = arr[self._surjection[i]]

        return inv_arr
