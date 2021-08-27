import random

import gmpy2


class IntegerArithmetic:
    POWMOD_GMP_SIZE = pow(2, 64)

    @staticmethod
    def powmod(x, e, n):
        """
        x ** e % n
        :param x:
        :param e:
        :param n:
        :return:
        """
        if x == 1:
            return 1

        if max(x, e, n) < IntegerArithmetic.POWMOD_GMP_SIZE:
            return pow(x, e, n)

        else:
            return int(gmpy2.powmod(x, e, n))

    @staticmethod
    def invert(x, n):
        """
        x ** (-1) % n
        :param x:
        :param n:
        :return:
        """
        x = int(gmpy2.invert(gmpy2.mpz(x), gmpy2.mpz(n)))

        if x == 0:
            raise ZeroDivisionError('invert(a, b) no inverse exists')

        return x

    @staticmethod
    def getprimeover(n):
        """
        Get a n-bit prime at random
        :param n:
        :return:
        """
        r = gmpy2.mpz(random.SystemRandom().getrandbits(n))
        r = gmpy2.bit_set(r, n - 1)

        return int(gmpy2.next_prime(r))

    @staticmethod
    def isqrt(n):
        """
        Get the integer square root of n
        :param n:
        :return:
        """
        return int(gmpy2.isqrt(n))

    @staticmethod
    def is_prime(n):
        """
        true if n is probably a prime, false otherwise
        :param n:
        :return:
        """
        return gmpy2.is_prime(int(n))

    @staticmethod
    def legendre(a, p):
        return pow(a, (p - 1) // 2, p)

    @staticmethod
    def tonelli(n, p):
        q = p - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1
        if s == 1:
            return pow(n, (p + 1) // 4, p)
        for z in range(2, p):
            if p - 1 == IntegerArithmetic.legendre(z, p):
                break
        c = pow(z, q, p)
        r = pow(n, (q + 1) // 2, p)
        t = pow(n, q, p)
        m = s
        while (t - 1) % p != 0:
            t2 = (t * t) % p
            for i in range(1, m):
                if (t2 - 1) % p == 0:
                    break
                t2 = (t2 * t2) % p
            b = pow(c, 1 << (m - i - 1), p)
            r = (r * b) % p
            c = (b * b) % p
            t = (t * c) % p
            m = i
        return r

    @staticmethod
    def gcd(a, b):
        return int(gmpy2.gcd(a, b))

    @staticmethod
    def next_prime(n):
        return int(gmpy2.next_prime(n))

    @staticmethod
    def generate_prime(left, right):
        """
        Generate a prime over (left, right]
        :param left:
        :param right:
        :return:
        """
        while True:
            random_integer = random.randint(left, right)
            random_prime = IntegerArithmetic.next_prime(random_integer)
            if random_prime <= right:
                return random_prime
