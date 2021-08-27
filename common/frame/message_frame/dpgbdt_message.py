from common.frame.message_frame.message import Message


class DPGBDTMessage(Message):
    GRAD_HESS = 'grad_hess'
    ENC_HISTOGRAM = 'enc_histogram'
    HOST_LOCAL_BEST = 'host_local_best'
    CURRENT_LEVEL_NODES = 'current_level_nodes'
    NEXT_LEVEL_NODES = 'next_level_nodes'
    SAMPLE_LOC_G2H = 'sample_loc_g2h'
    SAMPLE_LOC_H2G = 'sample_loc_h2g'
    MIN_CHILD_WEIGHT_BREACH = 'min_child_weight_breach'
    ALL_REACH_LEAVES = 'all_reach_leaves'
    TRAVERSE_SYNC = 'traverse_sync'
    LABEL_DIM = 'label_dim'
