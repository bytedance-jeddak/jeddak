import numpy as np
from common.util import constant
from fl.nn.backend.keras.util import generate_random_nums
from fl.nn.interactive.interactive import NNInteractive


class NNInteractivePHE(NNInteractive):
    """
    interaction functions based on PHE algorithm
    """

    def __init__(self,
                 messenger,
                 message,
                 learning_rate,
                 guest_party=None,
                 host_party=None,
                 role=None,
                 privacy_mode=constant.Encryptor.PLAIN) -> None:
        super(NNInteractivePHE, self).__init__(messenger,
                                               message,
                                               guest_party=guest_party,
                                               host_party=host_party,
                                               role=role,
                                               privacy_mode=privacy_mode)
        self.learning_rate = learning_rate
        self.noise_acc = None

    def interact_y_host(self, y_btm):
        """
        param y_btm: y of host bottom model
        return: None
        """
        encrypted_input_data = self.encrypt_parallel(y_btm)
        self._messenger.send(encrypted_input_data,
                             tag=self._message.CIPHER_HOST_BTM_OUT,
                             suffix=[self.epoch, self.batch],
                             parties=self.guest_party)

    def interact_y_guest(self):
        """
        param:
        return: encrypted y output of host bottom model
        """
        host_input_data = self._messenger.receive(
            tag=self._message.CIPHER_HOST_BTM_OUT,
            suffix=[self.epoch, self.batch],
            parties=self.host_party)[0]
        return host_input_data

    def interact_p_host(self, y_btm):
        """
        param:
        return: None
        """
        activation_input_data = self._messenger.receive(
            tag=self._message.CIPHER_GUEST_MID_OUT,
            suffix=[self.epoch, self.batch],
            parties=self.guest_party)[0]
        dec_activation_input_data = self.decrypt(activation_input_data)

        if self.noise_acc is None:
            self.noise_acc = generate_random_nums(np.zeros(
                [y_btm.shape[1], activation_input_data.shape[1]]),
                lower=-1,
                upper=1)

        dec_activation_input_data -= np.matmul(y_btm, self.noise_acc)

        self._messenger.send(dec_activation_input_data,
                             tag=self._message.PLAIN_GUEST_MID_OUT,
                             suffix=[self.epoch, self.batch],
                             parties=self.guest_party)

    def interact_p_guest(self, p_mid):
        """
        param p_mid:  encrypted p output of guest mid model
        return: real p output of guest mid model
        """
        activation_input_data_random = generate_random_nums(p_mid)
        activation_input_data_with_random = p_mid + activation_input_data_random
        self._messenger.send(activation_input_data_with_random,
                             tag=self._message.CIPHER_GUEST_MID_OUT,
                             suffix=[self.epoch, self.batch],
                             parties=self.host_party)

        dec_activation_input_data = self._messenger.receive(
            tag=self._message.PLAIN_GUEST_MID_OUT,
            suffix=[self.epoch, self.batch],
            parties=self.host_party)[0]

        dec_activation_input_data -= activation_input_data_random
        return dec_activation_input_data

    def interact_w_host(self):
        """
        param: 
        return: None
        """
        delta_w_host = self._messenger.receive(
            tag=self._message.CIPHER_GUEST_MID_GRADIENT_WEIGHTS,
            suffix=[self.epoch, self.batch],
            parties=self.guest_party)[0]

        dec_delta_w_host = self.decrypt(delta_w_host)

        noise_i = generate_random_nums(delta_w_host)
        dec_delta_w_host -= noise_i / self.learning_rate

        self._messenger.send(
            dec_delta_w_host,
            tag=self._message.PLAIN_GUEST_MID_GRADIENT_WEIGHTS,
            suffix=[self.epoch, self.batch],
            parties=self.guest_party)

        self._messenger.send(self.encrypt(self.noise_acc),
                             tag=self._message.CIPHER_HOST_EACC,
                             suffix=[self.epoch, self.batch],
                             parties=self.guest_party)

        self.noise_acc += noise_i

    def interact_w_guest(self, g_mid_w):
        """
        param g_mid_w: encrypted gradient of guest mid model dL/dw
        return: real gradient of guest mid model dL/dw
        """
        delta_w_host_rd = generate_random_nums(g_mid_w)
        delta_w_host_with_random = g_mid_w + delta_w_host_rd

        self._messenger.send(
            delta_w_host_with_random,
            tag=self._message.CIPHER_GUEST_MID_GRADIENT_WEIGHTS,
            suffix=[self.epoch, self.batch],
            parties=self.host_party)

        dec_delta_w_host = self._messenger.receive(
            tag=self._message.PLAIN_GUEST_MID_GRADIENT_WEIGHTS,
            suffix=[self.epoch, self.batch],
            parties=self.host_party)[0]
        dec_delta_w_host -= delta_w_host_rd
        return dec_delta_w_host

    def interact_g_host(self):
        """
        param:
        return: real gradient of host bottom model dL/dy
        """

        host_grad = self._messenger.receive(
            tag=self._message.HOST_BTM_GRADIENT_OUT,
            suffix=[self.epoch, self.batch],
            parties=self.guest_party)[0]
        host_grad = self.decrypt(host_grad)
        return host_grad

    def interact_g_guest(self, g_btm_y, g_p_host):
        """
        param g_btm_y: encrypted gradient of host bottom model dL/dy
        return: None
        """

        noise_acc = self._messenger.receive(tag=self._message.CIPHER_HOST_EACC,
                                            suffix=[self.epoch, self.batch],
                                            parties=self.host_party)[0]

        host_input_data_grads = g_btm_y - np.matmul(g_p_host, noise_acc.T)

        self._messenger.send(host_input_data_grads,
                             tag=self._message.HOST_BTM_GRADIENT_OUT,
                             suffix=[self.epoch, self.batch],
                             parties=self.host_party)
