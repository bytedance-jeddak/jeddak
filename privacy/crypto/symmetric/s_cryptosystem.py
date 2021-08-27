from privacy.crypto.cryptosystem import Cryptosystem


class SymmetricCryptosystem(Cryptosystem):
    def __init__(self, key):
        """

        :param key: dict
        """
        self._key = key
