class Cryptosystem(object):
    def encrypt(self, plaintext):
        pass

    def decrypt(self, ciphertext):
        pass

    @staticmethod
    def generate(*args):
        """
        Generate an instance

        :param args:
        :return:
        """
        pass


class Ciphertext(object):
    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message

    def __eq__(self, other):
        if self._message == other.message:
            return True
        else:
            return False

    def __hash__(self):
        """
        Must be explicitly inherited
        :return:
        """
        return hash(str(self._message))

