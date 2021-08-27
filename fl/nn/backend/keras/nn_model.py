from collections import Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras

from fl.nn.backend.keras.util import auc


class NNModel(object):
    """
    basic nn model, wrapper of top or bottom model
    """

    def __init__(self,
                 loss_fn,
                 optimizer="adam",
                 metrics=None,
                 batch_size=None,
                 epochs=1,
                 **kwargs):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        if "learning_rate" in kwargs:
            if optimizer == "adam":
                self.optimizer = keras.optimizers.Adam(
                    learning_rate=kwargs["learning_rate"])
            elif optimizer == "sgd":
                self.optimizer = keras.optimizers.SGD(
                    learning_rate=kwargs["learning_rate"])

        self.session = None
        self.model_conf = None
        self.model_id = "nn_model"

    def set_model_id(self, model_id):
        self.model_id = model_id

    def set_session(self, session):
        self.session = session

    def load_data(self, data):
        """
        load model from file
        """
        pass

    def build_model_from_conf(self, model_conf):
        """
        build keras model from conf
        """
        self.model = keras.Sequential.from_config(model_conf)
        self.model.compile(loss=self.loss_fn,
                           optimizer=self.optimizer,
                           metrics=self.metrics)
        self.model_conf = model_conf

    def build_model_from_path(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.model.compile(loss=self.loss_fn,
                           optimizer=self.optimizer,
                           metrics=self.metrics)
        self.model_conf = model_path

    def build_model(self, conf):
        if isinstance(conf, dict):
            return self.build_model_from_conf(conf)
        elif isinstance(conf, str) and conf.endswith("h5"):
            return self.build_model_from_path(conf)
        else:
            raise ValueError("Unknown conf type of {}, should be dict or '.h5'")


    def get_input_shape(self):
        return self.model.input_shape[1:]

    def get_output_shape(self):
        return self.model.output_shape[1:]

    def set_layer_weight(self, weight, layer_idx):
        self.model.layers[layer_idx].set_weights(weight)

    def get_layer_weight(self, layer_idx):
        return self.model.layers[layer_idx].get_weights()

    def get_all_layers_weight(self):
        return [(idx, layer.get_weights())
                for idx, layer in enumerate(self.model.layers)]

    def set_all_layers_weight(self, weights):
        if len(weights) != len(self.model.layers):
            raise ValueError(
                f"layer numbers not match, {len(weights)} != {len(self.model.layers)}"
            )

        for layer_w in weights:
            self.set_layer_weight(weight=layer_w[1], layer_idx=layer_w[0])

    def set_weights(self, weights):
        """
        set weights from existent model
        :param weights:
        :return:
        """
        self.model.set_weights(weights)

    def get_weights(self):
        """
        get model weights of each layers
        :return: np.array
        """
        return self.model.get_weights()

    def train(self, input_data, y_true):
        """
        param input_data: input tensor
        return: none
        """
        self.model.fit(input_data,
                       y_true,
                       batch_size=self.batch_size,
                       epochs=self.epochs)

    def forward(self, input_data):
        """
        Forward propergation, predict the output tensor
        param input_data: input numpy array
        return: output np array
        """
        output = self.model(input_data)
        return tf.compat.v1.keras.backend.eval(output)

    def forward_tensor(self, input_data):
        """
        Forward propergation, predict the output tensor
        param input_data: input tensor or numpy array
        return: output tensor
        """
        output = self.model(input_data)
        return output

    def backward(self, input_data, y_true):
        """
        Backward propergation, trainning and update layers
        param input_data: input tensor
        param y_true: labels 
        return: none
        """
        self.model.fit(input_data,
                       y_true,
                       batch_size=self.batch_size,
                       epochs=self.epochs)

    def backward_tensor(self, input_data, y_true):
        """
        Backward propergation, trainning and update layers
        param input_data: input tensor
        param y_true: labels
        return: none
        """
        self.model.fit(input_data,
                       y_true,
                       batch_size=self.batch_size,
                       epochs=self.epochs)

    def get_input_data_gradient(self, input_data, y_true):
        """
        param input_data: input numpy array
        return: gradient np array of dL/dx, output to input
        """
        input_data = tf.convert_to_tensor(input_data)
        grads, loss_value = self.get_input_data_gradient_tensor(input_data, y_true)

        y_pred = self.predict(input_data)

        y_true = np.array(y_true[0]).reshape(-1, 1)
        y_pred = np.array(y_pred).reshape(-1, 1)

        # print("local auc:{}".format(auc(y_true, y_pred)))

        return tf.compat.v1.keras.backend.eval(grads), loss_value

    def get_input_data_gradient_tensor(self, input_data, y_true):
        """
        param input_data: input tensor
        return: gradient tensor of dL/dx, output to input
        """
        # y, ids = y_true
        # y_true = y
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            y_new2 = self.model(input_data)
            loss_value = self.loss_fn(y_true, y_new2)
        grads = tape.gradient(loss_value, input_data)
        return grads, loss_value

    def predict(self, input_data):
        """
        param input_data: input tensor or numpy array
        return: output tensor
        """
        return tf.compat.v1.keras.backend.eval(self.model(input_data))

    def predict_tensor(self, input_data):
        """
        param input_data: input tensor or numpy array
        return: output tensor
        """
        y_new2 = self.model(input_data)
        return y_new2

    def compute_loss(self, input_data, y_true):
        input_data = tf.convert_to_tensor(input_data)
        loss_fn = getattr(tf.keras.losses, self.loss_fn)

        y_pred = self.model(input_data)
        loss = loss_fn(y_true, y_pred)

        if isinstance(loss, Iterable):
            loss = np.mean(loss_fn(y_true, y_pred).numpy())

        return loss

    def evaluate(self, input_data, y_true):
        return self.model.evaluate(input_data, y_true)

    def save_model(self, path):
        """
        save model to json_str and h5
        """
        self.model.save(path)

    def load_model(self, path, custom_objects=None):
        """
        param path:  model path
        """
        self.model = keras.models.load_model(path,
                                             custom_objects=custom_objects)
