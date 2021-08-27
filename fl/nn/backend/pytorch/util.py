from torch.nn import AdaptiveLogSoftmaxWithLoss
from torch.nn import BCELoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import CELU
from torch.nn import CTCLoss
from torch.nn import CosineEmbeddingLoss
from torch.nn import CrossEntropyLoss
from torch.nn import ELU
from torch.nn import GELU
from torch.nn import Hardshrink
from torch.nn import Hardsigmoid
from torch.nn import Hardswish
from torch.nn import Hardtanh
from torch.nn import HingeEmbeddingLoss
from torch.nn import KLDivLoss
from torch.nn import L1Loss
from torch.nn import LeakyReLU
from torch.nn import LogSigmoid
from torch.nn import LogSoftmax
from torch.nn import MSELoss
from torch.nn import MarginRankingLoss
from torch.nn import MultiLabelMarginLoss
from torch.nn import MultiLabelSoftMarginLoss
from torch.nn import MultiMarginLoss
from torch.nn import MultiheadAttention
from torch.nn import NLLLoss
from torch.nn import PReLU
from torch.nn import PoissonNLLLoss
from torch.nn import RReLU
from torch.nn import ReLU
from torch.nn import ReLU6
from torch.nn import SELU
from torch.nn import SiLU
from torch.nn import Sigmoid
from torch.nn import SmoothL1Loss
from torch.nn import SoftMarginLoss
from torch.nn import Softmax
from torch.nn import Softmax2d
from torch.nn import Softmin
from torch.nn import Softplus
from torch.nn import Softshrink
from torch.nn import Softsign
from torch.nn import Tanh
from torch.nn import Tanhshrink
from torch.nn import Threshold
from torch.nn import TripletMarginLoss
from torch.optim import ASGD
from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import Adam
from torch.optim import Adamax
from torch.optim import LBFGS
from torch.optim import RMSprop
from torch.optim import Rprop
from torch.optim import SGD
from torch.optim import SparseAdam


def loss_fn(y, g_y):
    """
    param y: prediction tensor
    param g_y: gradient tensor
    """
    return y * g_y


torch_loss_fn = {
    "L1Loss": L1Loss,
    "MSELoss": MSELoss,
    "CrossEntropyLoss": CrossEntropyLoss,
    "NLLLoss": NLLLoss,
    "PoissonNLLLoss": PoissonNLLLoss,
    "KLDivLoss": KLDivLoss,
    "BCELoss": BCELoss,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "MarginRankingLoss": MarginRankingLoss,
    "HingeEmbeddingLoss": HingeEmbeddingLoss,
    "MultiLabelMarginLoss": MultiLabelMarginLoss,
    "SmoothL1Loss": SmoothL1Loss,
    "SoftMarginLoss": SoftMarginLoss,
    "MultiLabelSoftMarginLoss": MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": CosineEmbeddingLoss,
    "MultiMarginLoss": MultiMarginLoss,
    "TripletMarginLoss": TripletMarginLoss,
    "CTCLoss": CTCLoss
}

torch_optimizer = {
    "SGD": SGD,
    "ASGD": ASGD,
    "Rprop": Rprop,
    "Adagrad": Adagrad,
    "Adadelta": Adadelta,
    "RMSprop": RMSprop,
    "Adam": Adam,
    "Adamax": Adamax,
    "SparseAdam": SparseAdam,
    "LBFGS": LBFGS
}

torch_activation = {
    "ELU": ELU,
    "Hardshrink": Hardshrink,
    "Hardsigmoid": Hardsigmoid,
    "Hardtanh": Hardtanh,
    "Hardswish": Hardswish,
    "LeakyReLU": LeakyReLU,
    "LogSigmoid": LogSigmoid,
    "MultiheadAttention": MultiheadAttention,
    "PReLU": PReLU,
    "ReLU": ReLU,
    "ReLU6": ReLU6,
    "RReLU": RReLU,
    "SELU": SELU,
    "CELU": CELU,
    "GELU": GELU,
    "Sigmoid": Sigmoid,
    "SiLU": SiLU,
    "Softplus": Softplus,
    "Softshrink": Softshrink,
    "Softsign": Softsign,
    "Tanh": Tanh,
    "Tanhshrink": Tanhshrink,
    "Threshold": Threshold,
    "Softmin": Softmin,
    "Softmax": Softmax,
    "Softmax2d": Softmax2d,
    "LogSoftmax": LogSoftmax,
    "AdaptiveLogSoftmaxWithLoss": AdaptiveLogSoftmaxWithLoss
}
