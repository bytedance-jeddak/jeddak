import torch

from common.util import constant
from fl.nn.backend.nn_guest import NNGuest
from fl.nn.backend.pytorch.mid_model import MidModel
from fl.nn.backend.pytorch.nn_model import NNModel
from fl.nn.backend.pytorch.util import loss_fn
from fl.nn.backend.pytorch.util import torch_loss_fn
from fl.nn.backend.pytorch.util import torch_optimizer


class VNNModelTorchGuest(NNGuest):
    """
    keras guest model
    """

    def __init__(self,
                 optimizer,
                 loss_func,
                 learning_rate=0.001,
                 batch_size=None,
                 use_mid=True,
                 mid_shape_in=None,
                 mid_shape_out=None,
                 mid_activation=None) -> None:
        super(VNNModelTorchGuest, self).__init__()

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.loss_fn = torch_loss_fn[loss_func]()
        self.optimizer = torch_optimizer[optimizer]

        self.use_mid = use_mid
        self.mid_shape_in = mid_shape_in
        self.mid_shape_out = mid_shape_out
        self.mid_activation = mid_activation

        self.frame = constant.NeuralNetwork.PYTORCH

    def build_model_from_path(self, btm_path, mid_path, top_path):
        self.btm_model = NNModel(loss_fn=loss_fn,
                                 optimizer=self.optimizer,
                                 learning_rate=self.learning_rate,
                                 batch_size=self.batch_size)

        self.btm_model.set_model(torch.load(btm_path))

        if self.use_mid:
            self.mid_model = MidModel(loss_fn=loss_fn,
                                      optimizer=self.optimizer,
                                      in_size=self.mid_shape_in,
                                      out_size=self.mid_shape_out,
                                      learning_rate=self.learning_rate)
            if mid_path:
                self.mid_model.set_model(torch.load(mid_path))
            else:
                self.mid_model.set_model()

        self.top_model = NNModel(loss_fn=self.loss_fn,
                                 optimizer=self.optimizer,
                                 learning_rate=self.learning_rate,
                                 batch_size=self.batch_size)

        self.top_model.set_model(torch.load(top_path))

    def build_model_from_conf(self, btm_conf, mid_conf, top_conf):
        pass

    def save_model(self, btm_model_path, mid_model_path, top_model_path):
        self.btm_model.save_model(btm_model_path)
        if self.use_mid:
            self.mid_model.save_model(mid_model_path)
        self.top_model.save_model(top_model_path)

    def load_model(self, btm_model_path, mid_model_path, top_model_path):
        self.btm_model.load_model(btm_model_path)
        if self.use_mid:
            self.mid_model.load_model(mid_model_path)
        self.top_model.load_model(top_model_path)
