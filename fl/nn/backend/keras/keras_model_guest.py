from tensorflow import keras
from common.util import constant
from fl.nn.backend.keras.mid_model import MidModel
from fl.nn.backend.keras.nn_model import NNModel
from fl.nn.backend.keras.util import loss_fn
from fl.nn.backend.nn_guest import NNGuest
from fl.nn.backend.keras.nn_model import NNModel
from fl.nn.backend.keras.mid_model import MidModel


class NNModelKerasGuest(NNGuest):
    """
    keras guest model
    """

    def __init__(self,
                 optimizer,
                 loss_func,
                 train_mode,
                 learning_rate=0.001,
                 batch_size=None,
                 use_mid=True,
                 mid_shape_in=None,
                 mid_shape_out=None,
                 mid_activation=None) -> None:
        super(NNModelKerasGuest, self).__init__()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_mid = use_mid
        self.mid_shape_in = mid_shape_in
        self.mid_shape_out = mid_shape_out
        self.mid_activation = mid_activation

        self.train_mode = train_mode
        self.backend = constant.NeuralNetwork.KERAS

        try:
            self.loss_fn = getattr(keras.losses, loss_func)
        except:
            print("can not find loss {}, set it to loss_fn".format(loss_func))
            self.loss_fn = loss_fn

    def build_model(self, **kwargs):
        pass

    def build_model_from_conf(self, btm_conf, mid_conf, top_conf):
        self.btm_model = None
        if btm_conf:
            self.btm_model = NNModel(loss_fn=loss_fn,
                                     optimizer=self.optimizer,
                                     learning_rate=self.learning_rate,
                                     batch_size=self.batch_size,
                                     metrics=None)

            self.btm_model.build_model_from_conf(btm_conf)
            self.btm_model.model.summary()
            self.btm_model.set_model_id("btm_model")

        if self.use_mid:
            self.mid_model = MidModel(learning_rate=self.learning_rate)
            if mid_conf:
                self.mid_model.build_model_from_conf(mid_conf)
                self.mid_shape_in = self.mid_model.model.input_shape[1]
                self.mid_shape_out = self.mid_model.model.output_shape[1]
                self.mid_model.model.summary()
            else:
                self.mid_model.model = keras.Sequential([
                    keras.Input(shape=self.mid_shape_in),
                    keras.layers.Dense(self.mid_shape_out,
                                       activation=self.mid_activation, use_bias=False),
                ])
                self.mid_model.model.compile(loss=loss_fn,
                                             optimizer=self.optimizer)
        else:
            if not self.mid_shape_out:
                self.mid_shape_out = self.top_model.model.input_shape[1] - self.btm_model.model.output_shape[1]

        self.top_model = NNModel(loss_fn=self.loss_fn,
                                 optimizer=self.optimizer,
                                 learning_rate=self.learning_rate,
                                 batch_size=self.batch_size)
        self.top_model.build_model_from_conf(top_conf)
        self.top_model.model.summary()
        self.top_model.set_model_id("top_model")

    def build_model_from_path(self, btm_path, mid_path, top_path):
        self.btm_model = None
        if btm_path:
            self.btm_model = NNModel(loss_fn=loss_fn,
                                     optimizer=self.optimizer,
                                     learning_rate=self.learning_rate,
                                     batch_size=self.batch_size)

            self.btm_model.build_model_from_path(btm_path)
            self.btm_model.model.summary()

        if self.use_mid:
            self.mid_model = MidModel(learning_rate=self.learning_rate)
            if mid_path:
                self.mid_model.build_model_from_path(mid_path)
                self.mid_shape_in = self.mid_model.model.input_shape[1]
                self.mid_shape_out = self.mid_model.model.output_shape[1]
                self.mid_model.model.summary()
            else:
                self.mid_model.model = keras.Sequential([
                    keras.Input(shape=self.mid_shape_in),
                    keras.layers.Dense(self.mid_shape_out,
                                       activation=self.mid_activation,
                                       use_bias=False),
                ])
                self.mid_model.model.compile(loss=loss_fn,
                                             optimizer=self.optimizer)
                self.mid_model.model_conf = self.mid_model.model.get_config()
        else:
            if not self.mid_shape_out:
                self.mid_shape_out = self.top_model.model.input_shape[1] - self.btm_model.model.output_shape[1]

        self.top_model = NNModel(loss_fn=self.loss_fn,
                                 optimizer=self.optimizer,
                                 learning_rate=self.learning_rate,
                                 batch_size=self.batch_size)
        self.top_model.build_model_from_path(top_path)
        self.top_model.model.summary()

    def save_model(self, btm_model_path, mid_model_path, top_model_path):
        self.btm_model.save_model(btm_model_path)
        if self.use_mid:
            self.mid_model.save_model(mid_model_path)
        self.top_model.save_model(top_model_path)

    def load_model(self, btm_model_path, mid_model_path, top_model_path):
        self.btm_model.load_model(btm_model_path, {"loss_fn": loss_fn})
        if self.use_mid:
            self.mid_model.load_model(mid_model_path, {"loss_fn": loss_fn})
        self.top_model.load_model(top_model_path)
