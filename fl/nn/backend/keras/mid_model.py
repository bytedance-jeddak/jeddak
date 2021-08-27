import multiprocessing
import time

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from tensorflow import keras

from fl.nn.backend.keras.util import loss_fn


class MidModel(object):
    """
    intermediate layer of nn model, connecting top and bottom layers
    """

    def __init__(self,
                 optimizer="sgd",
                 metrics=None,
                 batch_size=None,
                 epochs=1,
                 **kwargs):

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics

        self.batch_size = batch_size
        self.epochs = epochs

        self.output_size = None

        # weight
        self.host_weight = None

        # conf
        if "learning_rate" in kwargs:
            self.learning_rate = kwargs["learning_rate"]

        self.model_conf = None

        self.use_bias = False
        self.bias = None
        self.activation = "linear"

        # data
        self.host_input_data = None
        self.activation_input_data = None

        self.model = None
        self.dense_layer = None

        self.activation_gradient = None
        self.model_id = "nn_model"

        self.num_jobs = multiprocessing.cpu_count()
        if not self.num_jobs:
            self.num_jobs = 4

    def set_model_id(self, model_id):
        self.model_id = model_id

    def build_model(self, conf):
        if isinstance(conf, dict):
            return self.build_model_from_conf(conf)
        elif isinstance(conf, str) and conf.endswith("h5"):
            return self.build_model_from_path(conf)
        else:
            raise ValueError("Unknown conf type of {}, should be dict or '.h5'")

    def build_model_from_conf(self, model_conf):
        self.model = keras.Sequential.from_config(model_conf)
        self.model.compile(loss=loss_fn, optimizer=self.optimizer)

        self.dense_layer = self.model.layers[0]
        self.model_conf = model_conf
        self.use_bias = self.model.layers[0].use_bias

    def build_model_from_path(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.model.compile(loss=self.loss_fn,
                           optimizer=self.optimizer,
                           metrics=self.metrics)
        self.model_conf = model_path
        # self.use_bias = self.model.layers[0].use_bias

    def __get_weight_from_model(self, dense_layer=None):
        if dense_layer is None:
            dense_layer = self.model.layers[0]

        weight = dense_layer.get_weights()[0]

        bias = np.zeros(self.output_size)
        if self.use_bias:
            bias = dense_layer.get_weights()[1]

        return weight, bias

    def __forward_dense(self, x, w):
        # return np.matmul(x, w)
        res = Parallel(n_jobs=self.num_jobs)(
            delayed(lambda x: np.matmul(x, w))(x1) for x1 in np.array_split(x, self.num_jobs))
        ret = np.vstack(res)
        return ret

    # def apply_weight(self, delta_w):
    #     self.host_weight -= self.learning_rate * delta_w

    def apply_bias(self, delta_w):
        self.bias -= np.mean(delta_w, axis=0) * self.learning_rate

    def set_weights(self, weights):
        if weights[0] is not None:
            self.host_weight = weights[0]
        if weights[1] is not None:
            self.bias = weights[1]

    def get_weights(self):
        if self.host_weight is None:
            self.host_weight, self.bias = self.__get_weight_from_model(
                self.dense_layer)

        return [self.host_weight, self.bias]

    def forward_multiply(self, host_input_data):
        """
        param input_data: input of model
        return: w * y
        """
        print("Start interactive layer training 1...")

        self.host_input_data = host_input_data

        self.dense_layer = self.model.layers[0]

        if self.host_weight is None:
            self.host_weight, self.bias = self.__get_weight_from_model(
                self.dense_layer)
        t1 = time.time()  # print(" time ", time.time() - t1)
        self.activation_input_data = self.__forward_dense(
            self.host_input_data, self.host_weight)
        print("self.activation_input_data = self.__forward_dense time ", time.time() - t1)
        if self.use_bias is True:
            self.activation_input_data += self.bias

        return self.activation_input_data

    def forward_activate(self, activation_input_data):
        """
        param input_data: w * y
        return: f(w * y)
        """
        print("Start interactive layer training 2...")

        output = self.dense_layer.activation(activation_input_data)
        return output

    def backward(self, input_data, activation_input_data, output_gradient):
        """
        param input_data: y
        param activation_input_data: w * y
        param output_gradient: gradient of mid output dL/dp 
        return: dL/dw, dL/dy
        """

        act_input_tensor = tf.convert_to_tensor(activation_input_data)

        with tf.GradientTape() as grand_tape:
            grand_tape.watch(act_input_tensor)
            activation_output = tf.function(
                self.dense_layer.activation)(act_input_tensor)
            activation_backward = grand_tape.gradient([activation_output],
                                                      [act_input_tensor])
            # session.run(activation_backward)

            activation_backward = tf.compat.v1.keras.backend.eval(
                activation_backward[0])

        activation_gradient = output_gradient * activation_backward
        self.activation_gradient = activation_gradient

        t1 = time.time()  # print("g_w, g_y = self.mid_model.backward time ", time.time() - t1)
        # g_w = np.matmul(input_data.T, activation_gradient)
        res = Parallel(n_jobs=self.num_jobs)(delayed(lambda x: np.matmul(x, activation_gradient))(x1) for x1 in
                                             np.array_split(input_data.T, self.num_jobs))
        g_w = np.vstack(res)
        print("g_w = np.matmul time ", time.time() - t1)
        g_y = np.matmul(activation_gradient, self.host_weight.T)
        print("g_y = np.matmul time ", time.time() - t1)

        return g_w, g_y

    def update_w(self, g_w, bias=True):
        self.host_weight -= self.learning_rate * g_w
        if self.bias is not None and bias is True:
            self.bias -= np.sum(self.activation_gradient, axis=0) * self.learning_rate

    def save_model(self, path):
        weights = [self.host_weight]
        if self.use_bias:
            weights.append(self.bias)

        self.model.layers[0].set_weights(weights)
        self.model.save(path)

    def load_model(self, path, custom_objects=None):
        if not path:
            return None
        self.model = keras.models.load_model(path,
                                             custom_objects=custom_objects)
        self.host_weight, self.bias = self.model.layers[0].get_weights()
