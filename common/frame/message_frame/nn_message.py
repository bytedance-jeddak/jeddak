from common.frame.message_frame.message import Message


class NNMessage(Message):
    CIPHER_HOST_BTM_OUT = 'encrypted_host_input_data'
    CIPHER_GUEST_MID_OUT = 'activation_input_data'
    PLAIN_GUEST_MID_OUT = 'dec_activation_input_data'
    CIPHER_GUEST_MID_GRADIENT_WEIGHTS = 'delta_w_host_rd'
    PLAIN_GUEST_MID_GRADIENT_WEIGHTS = 'dec_delta_w_host_rd'
    HOST_BTM_GRADIENT_OUT = 'host_grad'
    CIPHER_HOST_EACC = 'noise_acc'

