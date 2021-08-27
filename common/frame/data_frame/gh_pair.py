from privacy.crypto.cryptosystem import Ciphertext


class GHPair(object):
    """
    The pair of first- and second-order derivative
    """
    def __init__(self, grad, hess=None):
        self._grad = grad
        self._hess = hess

    def __str__(self):
        return "grad: {}, hess: {}".format(self._grad, self._hess)

    def __add__(self, other):
        assert type(other) == GHPair or other == 0

        if other == 0:
            return self

        if self._hess is None or other.hess is None:
            return GHPair(self._grad + other.grad)

        return GHPair(self._grad + other.grad, self._hess + other.hess)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        assert type(other) == GHPair or other == 0

        if other == 0:
            return self

        if self._hess is None or other.hess is None:
            return GHPair(self._grad - other.grad)

        return GHPair(self._grad - other.grad, self._hess - other.hess)

    def __rsub__(self, other):
        return -1 * (self - other)

    def __mul__(self, other):
        assert type(other) != GHPair

        if self._hess is None or other.hess is None:
            return GHPair(self._grad * other)

        return GHPair(self._grad * other, self._hess * other)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1 * self

    @staticmethod
    def get_identity():
        return GHPair(0.0, 0.0)

    @property
    def grad(self):
        return self._grad

    @property
    def hess(self):
        return self._hess

    def apply(self, f, g=None):
        """
        Apply function to grad and hess
        :param f:
        :param g:
        :return:
        """
        if self._hess is None:
            return GHPair(f(self._grad))

        if g is None:
            return GHPair(f(self._grad), f(self._hess))

        else:
            return GHPair(f(self._grad), g(self._hess))

    def is_ciphertext(self):
        if isinstance(self._grad, Ciphertext) and isinstance(self._hess, Ciphertext):
            return True
        elif not isinstance(self._grad, Ciphertext) and not isinstance(self._grad, Ciphertext):
            return False
        else:
            raise TypeError("inconsistent grad and hess type: {} {}".format(
                type(self._grad), type(self._hess)))

    def is_first_order(self):
        return self._hess is None

    def get_first_order(self):
        return GHPair(self._grad)
