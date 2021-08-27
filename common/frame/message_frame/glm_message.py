from common.frame.message_frame.message import Message
from common.frame.message_frame.smm_message import SMMMessage
from common.frame.message_frame.fed_ave_message import FedAveMessage


class GLMMessage(Message, FedAveMessage, SMMMessage):
    LINEAR_PREDICTOR = 'linear_predictor'
    REG_LIKELIHOOD = 'reg_likelihood'
    ENC_RESIDUE = 'enc_residue'
    MASKED_ENC_GRADIENT_HOST = 'masked_enc_gradient_host'
    MASKED_GRADIENT_HOST = 'masked_gradient_host'
    EARLY_STOP = 'early_stop'
    PRED_HOST = 'pred_host'
    BATCH_ID = 'batch_id'
