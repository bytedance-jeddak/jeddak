from common.util import constant

import numpy as np

from fl.operator.operator import Operator


class Loss(Operator):
    def __init__(self, loss_type=constant.LossType.GBDT, need_log=False):
        super(Loss, self).__init__(need_log)

        self._loss_type = loss_type

    def eval(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        if self._loss_type == constant.LossType.GBDT:
            # retrieve arguments
            gh_pair = kwargs['gh_pair']
            lam = kwargs['lam']

            # compute loss
            if gh_pair.is_first_order():
                loss = -0.5 * np.square(gh_pair.grad) / lam

            else:
                loss = -0.5 * np.square(gh_pair.grad) / (gh_pair.hess + lam)

            return loss

        else:
            raise ValueError("invalid loss type: {}".format(self._loss_type))

    def get_weight(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        if self._loss_type == constant.LossType.GBDT:
            # retrieve arguments
            gh_pair = kwargs['gh_pair']
            lam = kwargs['lam']

            # compute weight
            if gh_pair == 0:
                return 0

            else:
                weight = - gh_pair.grad / (gh_pair.hess + lam)

            return weight

        else:
            raise ValueError("invalid loss type: {}".format(self._loss_type))
