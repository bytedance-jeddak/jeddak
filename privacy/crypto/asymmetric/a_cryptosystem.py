from privacy.crypto.cryptosystem import Cryptosystem


class AsymmetricCryptosystem(Cryptosystem):
    def __init__(self, public_key, private_key=None):
        """

        :param public_key: dict
        :param private_key: dict
        """
        self._public_key = public_key
        self._private_key = private_key
