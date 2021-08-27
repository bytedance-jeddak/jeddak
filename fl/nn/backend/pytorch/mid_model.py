import numpy as np
import torch


class MidModel(object):
    """
    intermediate layer of nn model, connecting top and bottom layers
    """

    def __init__(self,
                 loss_fn=None,
                 optimizer=None,
                 metrics=None,
                 batch_size=None,
                 in_size=None,
                 out_size=None,
                 epochs=1,
                 learning_rate=1e-3,
                 **kwargs):
        # shape

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics

        self.batch_size = batch_size
        self.epochs = epochs

        self.input_size = in_size
        self.output_size = out_size

        # weight
        # self.weight = None
        self.host_weight = None

        self.learning_rate = float(learning_rate)

        self.use_bias = True
        self.bias = None
        self.activation = "linear"

        # data
        self.host_input_data = None
        self.activation_input_data = None

        # self.model = torch.nn.Linear(self.input_size, self.output_size)

    def build_model(self, **kwargs):
        pass

    def build_model_from_conf(self, model_conf):
        pass

    def set_model(self, model=None):
        """
        """
        if model:
            self.model = model
        else:
            self.model = torch.nn.Sequential(torch.nn.Linear(self.input_size, self.output_size))
        self.optimizer = self.optimizer(self.model.parameters(),
                                        lr=self.learning_rate)

    def get_shape_in(self):
        pass

    def get_shape_out(self):
        pass

    def __get_weight_from_model(self):
        weight = self.model[0].weight.detach().numpy()

        if self.use_bias:
            bias = self.model[0].bias.detach().numpy()
        else:
            bias = np.zeros(self.output_size)

        return weight, bias

    def __forward_dense(self, x, w):
        return np.matmul(x, w.T)

    def apply_update(self, delta_w):
        self.host_weight -= self.learning_rate * delta_w

    def apply_bias(self, delta_w):
        self.bias -= np.mean(delta_w, axis=0) * self.learning_rate

    def set_host_weight(self, weight, bias=None):
        self.host_weight = weight
        if bias is not None:
            self.bias = bias

    def forward_multiply(self, host_input_data):
        """
        param input_data: input of model
        return: w * y
        """
        # print("Start interactive layer training 1...")

        self.host_input_data = host_input_data

        # self.dense_layer = self.model.layers[0]

        if self.host_weight is None:
            self.host_weight, self.bias = self.__get_weight_from_model()
            # self.dense_layer)
        self.activation_input_data = self.__forward_dense(
            self.host_input_data, self.host_weight)

        if self.bias is not None:
            self.activation_input_data += self.bias

        return self.activation_input_data

    def forward_activate(self, activation_input_data):
        """
        param input_data: w * y
        return: f(w * y)
        """
        # print("Start interactive layer training 2...")
        return activation_input_data
        # output = self.dense_layer.activation(activation_input_data)
        # return output

    def backward(self, input_data, activation_input_data, output_gradient):
        """
        param input_data: y
        param activation_input_data: w * y
        param output_gradient: gradient of mid output dL/dp 
        return: dL/dw, dL/dy
        dl/dw = dl/dp * yT
        dl/dy = wT * dl/dp
        """
        self.activation_gradient = output_gradient
        return np.matmul(input_data.T,
                         output_gradient), np.matmul(output_gradient,
                                                     self.host_weight)

    def update_w(self, g_w):
        if self.bias is not None:
            self.bias -= np.sum(self.activation_gradient, axis=0) * self.learning_rate 
        self.host_weight -= self.learning_rate * g_w.T

    def save_model(self, path):
        self.model[0].weight.data = torch.from_numpy(self.host_weight)
        self.model[0].bias.data = torch.from_numpy(self.bias)
        torch.save(self.model, path)

    def load_model(self, path, custom_objects=None):
        if not path:
            return None
        self.model = torch.load(path)
        self.host_weight, self.bias = self.__get_weight_from_model()
