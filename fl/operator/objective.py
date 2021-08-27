from common.util import constant
from scipy.special import loggamma

import numpy as np

from fl.operator.operator import Operator
from fl.operator.link_function import LinkFunction


class Objective(Operator):
    EPS = 1e-16

    def __init__(self, objective_type=constant.Objective.BINARY_LOGISTIC, max_delta_step=0.7, need_log=False):
        """

        :param objective_type:
        :param max_delta_step: in effect with default 0.7 for Poisson
        :param need_log: bool
        """
        super(Objective, self).__init__(need_log)

        self._objective_type = objective_type
        self._max_delta_step = max_delta_step

    def eval(self, y, y_hat):
        """
        Evaluate loss under the prescribed metric
        According to the source code in xgboost/src/metric/elementwise_metric
        :param y:
        :param y_hat:
        :return:
        """
        y_hat = self.pred(y_hat)

        if self._objective_type == constant.Objective.REG_SQUAREDERROR:
            return np.square(y - y_hat)

        elif self._objective_type == constant.Objective.BINARY_LOGISTIC:
            if y_hat < Objective.EPS:
                y_hat = Objective.EPS
            elif 1 - y_hat < Objective.EPS:
                y_hat = 1 - Objective.EPS
            return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        elif self._objective_type == constant.Objective.COUNT_POISSON:
            if y_hat < Objective.EPS:
                y_hat = Objective.EPS
            return loggamma(y + 1.0) + y_hat - np.log(y_hat) * y

        else:
            raise ValueError("invalid objective type: {}".format(self._objective_type))

    def grad(self, y, y_hat):
        """
        Get gradient w.r.t. y_hat
        :param y:
        :param y_hat:
        :return:
        """
        y_hat = self.pred(y_hat)

        if self._objective_type == constant.Objective.REG_SQUAREDERROR:
            return 2.0 * (y_hat - y)

        elif self._objective_type == constant.Objective.BINARY_LOGISTIC:
            return y_hat - y

        elif self._objective_type == constant.Objective.COUNT_POISSON:
            return y_hat - y

        else:
            raise ValueError("invalid objective type: {}".format(self._objective_type))

    def hess(self, y, y_hat):
        """
        Get hessian w.r.t. y_hat
        :param y:
        :param y_hat:
        :return:
        """
        y_hat = self.pred(y_hat)

        if self._objective_type == constant.Objective.REG_SQUAREDERROR:
            return 2.0

        elif self._objective_type == constant.Objective.BINARY_LOGISTIC:
            return y_hat * (1.0 - y_hat)

        elif self._objective_type == constant.Objective.COUNT_POISSON:
            return y_hat * np.exp(self._max_delta_step)

        else:
            raise ValueError("invalid objective type: {}".format(self._objective_type))

    def pred(self, y_hat):
        if self._objective_type == constant.Objective.REG_SQUAREDERROR:
            link_function = LinkFunction(link_type=constant.LinkType.LINEAR)
            return link_function.invert(y_hat)

        elif self._objective_type == constant.Objective.BINARY_LOGISTIC:
            link_function = LinkFunction(link_type=constant.LinkType.LOGISTIC)
            return link_function.invert(y_hat)

        elif self._objective_type == constant.Objective.COUNT_POISSON:
            link_function = LinkFunction(link_type=constant.LinkType.POISSON)
            return link_function.invert(y_hat)

        else:
            raise ValueError("invalid objective type: {}".format(self._objective_type))
