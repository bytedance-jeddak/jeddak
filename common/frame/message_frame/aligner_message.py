from common.frame.message_frame.message import Message


class AlignerMessage(Message):
    PLAIN_IDS = 'plain_ids'
    PLAIN_INTERSECTION = 'plain_intersection'
    DIFFIE_HELLMAN_PUBLIC_KNOWLEDGE = 'diffie_hellman_public_knowledge'
    DIFFIE_HELLMAN_PUBLIC_KNOWLEDGE_AND_G = 'diffie_hellman_public_knowledge_and_g'
    ENC_G = 'enc_g'
    ENC_H = 'enc_h'
    EENC_G = 'eenc_g'
    EENC_H = 'eenc_h'
    ENC_I = 'enc_i'
    TEE_INTERSECTION_TABLE = 'tee_intersection_table'
    TEE_SALT = 'tee_salt'
    TEE_META = 'tee_meta'
    TEE_NONCE = 'tee_nonce'
    TEE_SIG_PUBKEY = 'tee_sig_pubkey'
    TEE_E_AES_KEY = 'tee_e_aes_key'
    TEE_DATA = 'tee_data'
    TEE_BUCKET_NUM = 'tee_bucket_num'
