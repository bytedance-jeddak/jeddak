import random

from privacy.crypto.cryptosystem import Ciphertext
from privacy.crypto.symmetric.s_cryptosystem import SymmetricCryptosystem
from privacy.math.integer_arithmetic import IntegerArithmetic


class DiffieHellman(SymmetricCryptosystem):
    """
    A commutative encryption scheme inspired by Pohlig, Stephen, and Martin Hellman. "An improved algorithm
        for computing logarithms over GF(p) and its cryptographic significance." 1978
    Enc(x) = x^a mod p, with public knowledge p being a prime and satisfying that (p - 1) / 2 is also a prime
    Dec(y) = y^(a^(-1) mod phi(p)) mod p
    """
    def __init__(self, key={}, public_knowledge=None):
        """

        :param key: {'exponent': int, 'exponent_inverse': int}
        :param public_knowledge: {'modulus': int}
        """
        super(DiffieHellman, self).__init__(key)

        self._public_knowledge = public_knowledge

    @property
    def exponent(self):
        return self._key['exponent']

    @property
    def exponent_inverse(self):
        return self._key['exponent_inverse']

    @property
    def modulus(self):
        return self._public_knowledge['modulus']

    def encrypt(self, plaintext):
        """

        :param plaintext: int or DiffieHellmanCiphertext
        :return: DiffieHellmanCiphertext
        """
        if type(plaintext) == int:
            pass
        elif type(plaintext) == DiffieHellmanCiphertext:
            plaintext = plaintext.message
        else:
            raise TypeError("invalid plaintext type: {}".format(type(plaintext)))

        ciphertext = IntegerArithmetic.powmod(plaintext, self.exponent, self.modulus)

        return DiffieHellmanCiphertext(ciphertext)

    def decrypt(self, ciphertext):
        """

        :param ciphertext: int or DiffieHellmanCiphertext
        :return:
        """
        if type(ciphertext) == int:
            pass
        elif type(ciphertext) == DiffieHellmanCiphertext:
            ciphertext = ciphertext.message
        else:
            raise TypeError("invalid plaintext type: {}".format(type(ciphertext)))

        plaintext = IntegerArithmetic.powmod(ciphertext, self.exponent_inverse, self.modulus)

        return plaintext

    @staticmethod
    def generate_with_public_knowledge(key_size=1024, public_knowledge=None):
        """
        Generate an instance with a modulus and absent exponent
        If public_knowledge does not present, a random one will be picked
        :param key_size: int
        :param public_knowledge: dict
        :return: DiffieHellman
        """
        if public_knowledge is not None:
            instance = DiffieHellman(public_knowledge=public_knowledge)

        else:
            key_size_half = key_size // 2
            while True:
                modulus_half = IntegerArithmetic.generate_prime(2 ** (key_size_half - 1), 2 ** key_size_half - 1)
                modulus = modulus_half * 2 + 1
                if IntegerArithmetic.is_prime(modulus):
                    instance = DiffieHellman(public_knowledge={'modulus': modulus})
                    break

        return instance

    def fill_exponent_at_random(self):
        """
        Fill an exponent at random
        :return:
        """
        while True:
            exponent = random.randint(2, self.modulus)
            if IntegerArithmetic.gcd(exponent, self.modulus - 1) == 1:
                exponent_inverse = IntegerArithmetic.invert(exponent, self.modulus - 1)
                break

        self._key['exponent'] = exponent
        self._key['exponent_inverse'] = exponent_inverse

    @staticmethod
    def generate(key_size=1024):
        instance = DiffieHellman.generate_with_public_knowledge(key_size)
        instance.fill_exponent_at_random()
        return instance


class DiffieHellmanCiphertext(Ciphertext):
    def __init__(self, message):
        """

        :param message: int
        """
        super(DiffieHellmanCiphertext, self).__init__(message)

    def __eq__(self, other):
        if type(other) == DiffieHellmanCiphertext and self._message == other.message:
            return True
        elif type(other) == int and self._message == other:
            return True
        else:
            raise TypeError("invalid diffie-hellman comparison type: {}".format(type(other)))

    def __hash__(self):
        return super(DiffieHellmanCiphertext, self).__hash__()
