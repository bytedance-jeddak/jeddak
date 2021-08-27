import torch


class NNModel(object):
    """
    basic nn model, wrapper of top or bottom model
    """

    def __init__(self,
                 loss_fn,
                 optimizer=None,
                 metrics=None,
                 batch_size=None,
                 epochs=1,
                 learning_rate=1e-3):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.learning_rate = float(learning_rate)

        self.device = None

    def load_data(self, data):
        """
        load model from file
        """
        pass

    def build_model_from_conf(self, model_conf):
        """
        build keras model from conf
        """
        pass

    def set_model(self, model):
        """
        build keras model from conf
        """
        self.model = model
        self.optimizer = self.optimizer(self.model.parameters(),
                                        lr=self.learning_rate)

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

    def get_input_shape(self):
        pass

    def get_output_shape(self):
        pass

    def set_layer_weight(self, weight, layer_idx):
        pass

    def get_layer_weight(self, layer_idx):
        pass

    def train(self, input_data, y_true):
        """
        param input_data: input tensor
        return: none
        """
        pass

    @staticmethod
    def data_adapter(data):
        data = torch.from_numpy(data)
        if data.dtype == torch.float64:
            data = data.float()
        return data

    def forward(self, input_data):
        """
        Forward propergation, predict the output tensor
        param input_data: input numpy array
        return: output np array
        """
        X = self.data_adapter(input_data)

        pred = self.model(X)
        return pred.detach().numpy()

    def backward(self, input_data, y_true):
        """
        Backward propergation, trainning and update layers
        param input_data: input tensor
        param y_true: labels 
        return: none
        """
        X = self.data_adapter(input_data)
        y = self.data_adapter(y_true)

        # X, y = X.to(self.device), y.to(self.device)
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        # print(f"loss: {loss.item():>7f}")
        self.optimizer.zero_grad()
        loss.backward(torch.ones_like(loss).float())
        self.optimizer.step()

    def get_input_data_gradient(self, input_data, y_true):
        """
        param input_data: input numpy array
        return: gradient np array of dL/dx, output to input
        """
        X = self.data_adapter(input_data)
        y = self.data_adapter(y_true)

        X.requires_grad = True
        pred = self.model(X)

        loss = self.loss_fn(pred, y)

        # loss.backward(torch.ones(X.size()).double())
        loss.backward()

        return X.grad.data.numpy()

    def predict(self, input_data):
        """
        param input_data: input tensor or numpy array
        return: output tensor
        """
        return self.forward(input_data)

    def evaluate(self, input_data, y_true):
        X = self.data_adapter(input_data)
        y = self.data_adapter(y_true)

        # X, y = X.to(self.device), y.to(self.device)
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        return loss.item()

    def save_model(self, path):
        """
        """
        torch.save(self.model, path)

    def load_model(self, path, custom_objects=None):
        """
        """
        self.model = torch.load(path)
