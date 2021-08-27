import torch

from fl.nn.backend.nn_host import NNHost
from fl.nn.backend.pytorch.nn_model import NNModel
from fl.nn.backend.pytorch.util import loss_fn
from fl.nn.backend.pytorch.util import torch_optimizer


class VNNModelTorchHost(NNHost):
    """
    keras host model
    """

    def __init__(self,
                 optimizer,
                 learning_rate=0.001,
                 metircs="acuracy",
                 batch_size=None) -> None:
        super(VNNModelTorchHost, self).__init__()
        self.optimizer = torch_optimizer[optimizer]
        self.loss_fn = loss_fn
        self.learning_rate = float(learning_rate)
        self.metircs = metircs
        self.batch_size = batch_size

    def build_model_from_path(self, btm_path):
        self.btm_model = NNModel(loss_fn=loss_fn,
                                 optimizer=self.optimizer,
                                 learning_rate=self.learning_rate,
                                 batch_size=self.batch_size)
        self.btm_model.set_model(torch.load(btm_path))

    def build_model_from_conf(self, btm_conf):
        pass

    def save_model(self, btm_model_path):
        self.btm_model.save_model(btm_model_path)

    def load_model(self, btm_model_path):
        self.btm_model.load_model(btm_model_path)
