from privacy.crypto.cryptosystem import Cryptosystem


class Plain(Cryptosystem):
    def encrypt(self, plaintext):
        return plaintext

    def decrypt(self, ciphertext):
        return ciphertext

    @staticmethod
    def generate(*args):
        """
        Generate an instance

        :param args:
        :return:
        """
        return Plain()
