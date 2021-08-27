import multiprocessing
from collections import Iterable

import numpy as np
from joblib import Parallel, delayed

from common.factory.encryptor_factory import EncryptorFactory
from common.util import constant
from privacy.crypto.plain.plain import Plain


class NNInteractive(object):
    """
    base of interaction functions, and 
    a simple implementation using shared memory
    """

    def __init__(self, messenger, message, guest_party=None, host_party=None, role=None,
                 privacy_mode=constant.Encryptor.CPAILLIER) -> None:
        self._messenger = messenger
        self._message = message
        self.guest_party = guest_party
        self.host_party = host_party
        self.role = role

        self.encrypt_method = privacy_mode
        self._encryptor = self.get_encryptor(self.encrypt_method, 1024)

        self.num_jobs = multiprocessing.cpu_count()
        if not self.num_jobs:
            self.num_jobs = 4

    def interact_y_host(self, y_btm):
        """
        param y_btm: y of host bottom model
        return: None
        """
        pass

    def interact_y_guest(self):
        """
        param:
        return: y output of host bottom model
        """
        pass

    def interact_p_host(self, y_btm):
        """
        param:
        return: None
        """
        pass

    def interact_p_guest(self, p_mid):
        """
        param p_mid:  encrypted p output of guest mid model
        return: real p output of guest mid model
        """
        pass

    def interact_w_host(self):
        """
        param: 
        return: None
        """
        pass

    def interact_w_guest(self, g_mid_w):
        """
        param g_mid_w: encrypted gradient of guest mid model dL/dw
        return: real gradient of guest mid model dL/dw
        """
        pass

    def interact_g_host(self):
        """
        param:
        return: real gradient of host bottom model dL/dy
        """
        pass

    def interact_g_guest(self, g_btm_y, g_p_host):
        """
        param g_btm_y: encrypted gradient of host bottom model dL/dy
        return: None
        """
        pass

    def get_noise_acc(self):
        return self.noise_acc

    def set_noise_acc(self, noise_acc):
        self.noise_acc = noise_acc

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, b):
        self._batch = b

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, e):
        self._epoch = e

    def encrypt(self, x):
        return self.__operator(self._encryptor.encrypt, x)

    def decrypt(self, x):
        return self.__operator(self._encryptor.decrypt, x)

    @staticmethod
    def get_encryptor(method, key_size):
        if method in constant.NeuralNetwork.SUPPORT_PRIVACY_MODE:
            encryptor = EncryptorFactory.get_instance(
                task_type=constant.TaskType.LINEAR_REGRESSION,
                encrypter=method,
                key_size=key_size)
        elif method == constant.Encryptor.PLAIN:
            encryptor = Plain.generate()
        else:
            raise ValueError(
                f"Unknown support encrypted method of {method}"
            )
        return encryptor

    @staticmethod
    def __operator(func, x):
        if not isinstance(x, Iterable):
            return func(x)
        else:
            if isinstance(x, np.ndarray):
                shape = x.shape
                enc_x = np.array([func(_x) for _x in x.flatten()])
                return enc_x.reshape(shape)
            else:
                raise ValueError("Not support type for encrypt type")

    def encrypt_parallel(self, x):
        fn = self._encryptor.encrypt
        return self.__operator_parallel(fn, x)

    def decrypt_parallel(self, x):
        fn = self._encryptor.decrypt
        return self.__operator_parallel(fn, x)

    def __operator_parallel(self, func, x):
        if not isinstance(x, Iterable):
            return func(x)
        else:
            if isinstance(x, np.ndarray):
                shape = x.shape
                res = Parallel(n_jobs=self.num_jobs)(
                    delayed(lambda fn, x: np.array([fn(_x) for _x in x.flatten()]))(func, x1) for x1 in
                    np.array_split(x.flatten(), self.num_jobs))
                enc_x = np.hstack(res)
                return enc_x.reshape(shape)
            else:
                raise ValueError("Not support type for encrypt type")
