from common.util import constant
import numpy as np

from fl.operator.operator import Operator


class Regularizer(Operator):
    def __init__(self, penalty=constant.Regularizer.L2, strength=1.0, need_log=False):
        """
        reg_l2 = 1 / 2 * strength * np.linalg.norm(w) ** 2
        reg_l1 = strength * np.linalg.norm(w, 1)
        :param penalty: constant.REGULARIZER.L2, constant.REGULARIZER.L1
        :param strength: float, 1 / C
        :param need_log: bool
        """
        super(Regularizer, self).__init__(need_log)

        self._penalty = penalty
        assert self._penalty in (constant.Regularizer.L2, constant.Regularizer.L1)

        self._strength = strength

    def get_loss(self, coef, intercept=None):
        if intercept is None:
            intercept = 0.0

        if self._penalty == constant.Regularizer.L2:
            reg_val = (np.square(np.linalg.norm(coef)) + np.square(intercept)) / 2

        elif self._penalty == constant.Regularizer.L1:
            reg_val = np.linalg.norm(coef, 1) + np.abs(intercept)

        else:
            raise ValueError("invalid penalty: {}".format(self._penalty))

        return self._strength * reg_val

    def get_likelihood(self, coef, intercept=None):
        return -self.get_loss(coef, intercept)

    def get_loss_gradient(self, coef, intercept=None):
        if self._penalty == constant.Regularizer.L2:
            reg_grad = coef

        elif self._penalty == constant.Regularizer.L1:
            reg_grad = np.sign(coef)

        else:
            raise ValueError("invalid penalty: {}".format(self._penalty))

        if intercept is not None:
            if self._penalty == constant.Regularizer.L2:
                intercept_grad = intercept

            elif self._penalty == constant.Regularizer.L1:
                intercept_grad = np.sign(intercept)

            else:
                raise ValueError("invalid penalty: {}".format(self._penalty))

            return self._strength * np.append(reg_grad, intercept_grad)

        else:
            return self._strength * reg_grad

    def get_likelihood_gradient(self, coef, intercept=None):
        return -self.get_loss_gradient(coef, intercept)
