from common.util import constant
from fl.nn.backend.keras.util import loss_fn
from fl.nn.backend.nn_host import NNHost
from fl.nn.backend.keras.nn_model import NNModel as NNModel


class NNModelKerasHost(NNHost):
    """
    keras host model
    """

    def __init__(self,
                 optimizer,
                 loss_func,
                 train_mode,
                 learning_rate=0.001,
                 metrics="accuracy",
                 batch_size=None) -> None:
        super(NNModelKerasHost, self).__init__()
        self.optimizer = optimizer
        # self.loss_fn = loss_func
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.batch_size = batch_size

        self.backend = constant.NeuralNetwork.KERAS
        self.train_mode = train_mode

    def build_model(self, **kwargs):
        pass

    def build_model_from_conf(self, btm_conf):
        self.btm_model = NNModel(loss_fn=loss_fn,
                                 optimizer=self.optimizer,
                                 batch_size=self.batch_size)
        self.btm_model.build_model_from_conf(btm_conf)
        self.btm_model.model.summary()

    def build_model_from_path(self, btm_conf):
        self.btm_model = NNModel(loss_fn=loss_fn,
                                 optimizer=self.optimizer,
                                 batch_size=self.batch_size)
        self.btm_model.build_model_from_path(btm_conf)
        self.btm_model.model.summary()

    def save_model(self, btm_model_path):
        self.btm_model.save_model(btm_model_path)

    def load_model(self, btm_model_path):
        self.btm_model.load_model(btm_model_path, {"loss_fn": loss_fn})
