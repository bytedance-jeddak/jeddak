import numpy as np

from common.factory.logger_factory import LoggerFactory
from common.frame.data_frame.sample import Sample


class NNHost(object):
    """
    base of host model
    """

    def __init__(self):
        self.btm_model = None
        self.interactor = None
        self.dataset_loader = None
        self._privacy_budget = 2.0

        self._byte_logger = LoggerFactory.get_global_instance()
        self.backend = None

    def build_model(self, **kwargs):
        # delegated to subclass
        pass

    def forward(self, x_train):
        """
        param x_train:
        return: None
        """
        y_btm = self.btm_model.forward(x_train)

        # send y_btm output of bottom model
        self.interactor.interact_y_host(y_btm)
        # decrypt p_mid output of guest mid
        self.interactor.interact_p_host(y_btm)

        self._cache_x_btm = x_train

    def backward(self, y_train):
        """
        param y_train: 
        return: None
        """
        # decrypt dL/dw for guest mid model
        self.interactor.interact_w_host()
        # noise_acc_dist = self.interactor.interact_w_host()
        # noise_acc = self.interactor.interact_w_host()

        # self.interactor.temp_send_acc(noise_acc_dist)
        # self.interactor.temp_send_acc(noise_acc)

        # decrypt dL/dy of bottom model
        g_y = self.interactor.interact_g_host()
        self.btm_model.backward(self._cache_x_btm, g_y)

    def train(self, x_train, y_train=None):
        self.forward(x_train)
        self.backward(y_train)

    def predict(self, input_data):
        """
        param input_data: input data to be predicted
        return: None
        """
        print('enter predict')
        self.forward(input_data)

    def save_model(self, btm_model_path):
        # override by sub class
        pass

    def load_model(self, btm_model_path):
        # override by sub class
        pass

    def forward_1(self, x_train):
        """
        param x_train:
        return:
        """
        y_btm = self.btm_model.forward(x_train)
        return y_btm

    def interact_f(self, y_btm, epoch=None, batch=None):
        # send y_btm output of bottom model
        self.interactor.interact_async_send_host(y_btm, "interact_f", epoch=epoch, batch=batch)

    def interact_b(self):
        # decrypt dL/dy of bottom model
        g_y = self.interactor.interact_async_receive_host()
        return g_y

    def backward_1(self, x_btm, g_y):
        """
        param y_train:
        return:
        """
        self.btm_model.backward(x_btm, g_y)
