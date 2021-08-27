from common.util import constant
import numpy as np

from fl.operator.operator import Operator


class LinkFunction(Operator):
    LARGE_EXPONENT = 50

    def __init__(self, link_type=constant.LinkType.LOGISTIC, need_log=False):
        """

        :param link_type: constant.LINK_TYPE.LINEAR, constant.LINK_TYPE.LOGISTIC or constant.LINK_TYPE.POISSON
        :param need_log: bool
        """
        super(LinkFunction, self).__init__(need_log)

        self._link_type = link_type
        assert self._link_type in (constant.LinkType.LINEAR, constant.LinkType.LOGISTIC, constant.LinkType.POISSON)

    def eval(self, z):
        """
        Evaluate the function of z
        :param z:
        :return:
        """
        if self._link_type == constant.LinkType.LINEAR:
            return z

        elif self._link_type == constant.LinkType.LOGISTIC:
            return np.log(z / (1 - z))

        elif self._link_type == constant.LinkType.POISSON:
            return np.log(z)

        else:
            raise ValueError("invalid link type: {}".format(self._link_type))

    def invert(self, eta):
        """
        Evaluate the inverse function of eta
        :param eta:
        :return:
        """
        if self._link_type == constant.LinkType.LINEAR:
            return eta

        elif self._link_type == constant.LinkType.LOGISTIC:
            if eta > 0:
                return 1 / (1 + np.exp(-eta))
            else:
                exp_eta = np.exp(eta)
                return exp_eta / (1 + exp_eta)

        elif self._link_type == constant.LinkType.POISSON:
            return np.exp(eta)

        else:
            raise ValueError("invalid link type: {}".format(self._link_type))

    def int_invert(self, eta):
        """
        Evaluate the inverse anti-derivative function of eta
        :return:
        """
        if self._link_type == constant.LinkType.LINEAR:
            return np.square(eta) / 2

        elif self._link_type == constant.LinkType.LOGISTIC:
            if eta > LinkFunction.LARGE_EXPONENT:
                return eta
            else:
                return np.log(1 + np.exp(eta))

        elif self._link_type == constant.LinkType.POISSON:
            return np.exp(eta)

        else:
            raise ValueError("invalid link type: {}".format(self._link_type))
