from Crypto.PublicKey import RSA
import random
import gmpy2


class ObliviousTransferSender:
    def __init__(self, rsa_bit=1024):
        key = RSA.generate(rsa_bit)
        self.e = key.e
        self.d = key.d
        self.n = key.n

    def genRandVal(self):
        x0 = random.randint(0, self.n-1)
        x1 = random.randint(0, self.n-1)
        return x0, x1

    def getPublicInfo(self):
        return self.n, self.e

    def blindMessage(self, m0, m1, x0, x1, v):

        k0 = gmpy2.powmod(v - x0, self.d, self.n)
        k1 = gmpy2.powmod(v - x1, self.d, self.n)
        res0 = (m0 + k0) % self.n
        res1 = (m1 + k1) % self.n
        return res0, res1


class ObliviousTransferReceiver:
    def __init__(self, n, e):
        self.n = n
        self.e = e

    def genChoices(self, seed, size):
        random.seed(seed)
        return [random.randint(0, 1) for i in range(size)]

    def evaluate(self, x0, x1, b):
        k = random.randint(0, self.n)
        v = x0 if b == 0 else x1
        v = (v + gmpy2.powmod(k, self.e, self.n)) % self.n
        return v, k

    def reveal(self, m0_prime, m1_prime, b, k):
        m_prime = m0_prime if b ==0 else m1_prime
        m = (m_prime - k) % self.n
        return m


if __name__ == "__main__":
    # bit = 1
    # sender = ObliviousTransferSender()
    # n, e = sender.getPublicInfo()
    # receiver = ObliviousTransferReceiver(n, e)
    # m0 = 123123123123123123123123123123123123123123
    # m1 = 456456456456456456456456456456456456456456456456
    # x0, x1 = sender.genRandVal()
    # v, k = receiver.evaluate(x0, x1, bit)
    # m0_prime, m1_prime = sender.blindMessage(m0, m1, x0, x1, v)
    # res = receiver.reveal(m0_prime, m1_prime, bit, k)
    # print(sender.getPublicInfo())
    # print(res)
    x = pow(2, 10)
    y = pow(2, x)
    print(y)



