import logging

import numpy as np

from common.factory.logger_factory import LoggerFactory
from common.frame.data_frame.sample import Sample


class NNGuest(object):
    """
    base of guest model
    """

    def __init__(self):
        self.btm_model = None
        self.mid_model = None
        self.top_model = None
        self.interactor = None
        self.use_mid = True
        self.mid_shape_out = None
        self._byte_logger = LoggerFactory.get_global_instance()
        self._current_loss = None
        self.use_ldp = False
        self._privacy_budget = 2.0

    @property
    def current_loss(self):
        return self._current_loss

    def build_model(self, **kwargs):
        # delegated to subclass
        pass

    def forward(self, x_train):
        """
        param x_train:
        return: None
        """

        if not self.btm_model:
            p_guest = np.zeros(x_train.shape)
        else:
            p_guest = self.btm_model.forward(x_train)

        # encrypted y output of host bottom model [[y]]
        y_btm_host = self.interactor.interact_y_guest()

        # encrypted p output of mid model w*[[y]]
        if self.use_mid:
            p_mid = self.mid_model.forward_multiply(y_btm_host)

            # decrypted p output of mid model p=w*y
            dec_p_mid = self.interactor.interact_p_guest(p_mid)
            p_host = self.mid_model.forward_activate(dec_p_mid)
        else:
            dec_p_mid = y_btm_host
            p_host = dec_p_mid

        # merge two vector
        p_merge = self.merge_p(p_host, p_guest)

        # cache vars
        self._cache_x_btm = x_train
        self._cache_y_btm_host = y_btm_host
        self._cache_p_host = dec_p_mid
        self._cache_p_merge = p_merge

    def backward(self, y_train):
        """
        param y_train:
        return: None
        """

        self._current_loss = self.top_model.evaluate(self._cache_p_merge, y_train)
        g_p, _ = self.top_model.get_input_data_gradient(self._cache_p_merge, y_train)

        logging.info("loss:{}".format(self._current_loss))

        # split host and guest gradient
        g_p_host, g_p_guest = self.split_p(g_p, self.mid_shape_out)
        self.top_model.backward(self._cache_p_merge, y_train)

        if self.use_mid:
            logging.info("use mid model")
            # encrypted gradient dL/dw=dL/dp*[[y]]
            g_w, g_y = self.mid_model.backward(self._cache_y_btm_host,
                                               self._cache_p_host, g_p_host)

            # decrypted gradient dL/dw=dL/dp*y
            g_w = self.interactor.interact_w_guest(g_w)
            self.mid_model.update_w(g_w, bias=False)

        else:
            logging.info("skip mid model")
            g_y = g_p_host

        # send dL/dy to host
        self.interactor.interact_g_guest(g_y, g_p_host)

        if self.btm_model is not None:
            self.btm_model.backward(self._cache_x_btm, g_p_guest)

    def train(self, x_train, y_train):
        self.forward(x_train)
        self.backward(y_train)

    def predict(self, input_data, p_merge=None):
        """
        param input_data: input data to be predicted
        return: output prediction
        """
        print('enter predict')
        self.forward(input_data)
        if not p_merge:
            p_merge = self._cache_p_merge

        output = self.top_model.forward(p_merge)

        return output

    def eval(self):
        pass

    def save_model(self, btm_model_path, mid_model_path, top_model_path):
        # override by sub class
        pass

    def load_model(self, btm_model_path, mid_model_path, top_model_path):
        # override by sub class
        pass

    @staticmethod
    def merge_p(a, b):
        return np.concatenate((a, b), axis=1)

    @staticmethod
    def split_p(p_merge, size):
        return p_merge[:, 0:size], p_merge[:, size:]

    def forward_1(self, x_train):
        p_guest = self.btm_model.forward(x_train)
        return p_guest

    def interact_f(self):
        # encrypted y output of host bottom model [[y]]
        y_btm_host = self.interactor.interact_async_receive_guest()
        return y_btm_host

    def forward_2(self, p_host, p_guest):
        # merge two vector
        p_merge = self.merge_p(p_host, p_guest)
        return p_merge

    def backward_1(self, y_train, p_merge):
        """
        param y_train:
        return: None
        """
        print("evaluate top model loss:")
        self._current_loss = self.top_model.evaluate(p_merge, y_train)
        print(self._current_loss)
        g_p = self.top_model.get_input_data_gradient(p_merge,
                                                     y_train)
        self.top_model.backward(p_merge, y_train)

        # split host and guest gradient
        g_p_host, g_p_guest = self.split_p(g_p, self.mid_shape_out)
        return g_p_host, g_p_guest

    def interact_b(self, g_y, epoch=None, batch=None):
        # send dL/dy (g_p_host) to host
        self.interactor.interact_async_send_guest(g_y, "interact_b", epoch=epoch, batch=batch)

    def backward_2(self, x_btm, g_p_guest):
        self.btm_model.backward(x_btm, g_p_guest)
