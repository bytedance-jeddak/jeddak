from common.util import constant
from fl.nn.backend.keras.keras_model_guest import NNModelKerasGuest
from fl.nn.backend.keras.keras_model_host import NNModelKerasHost
from fl.nn.backend.pytorch.torch_model_guest import VNNModelTorchGuest
from fl.nn.backend.pytorch.torch_model_host import VNNModelTorchHost


class NNInitializer(object):
    """
    initialize model and cryptographic interactive interface according to the parameters
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def create_keras_guest_from_conf(btm_conf, mid_conf, top_conf, optimizer,
                                     loss, learning_rate, batch_size, use_mid,
                                     mid_shape_in, mid_shape_out,
                                     mid_activation, train_mode):
        """
        param btm_conf: bottom model's conf json string
        param mid_conf: middle model's conf json string
        param top_conf: top model's conf json string
        return: keras guest model
        """
        guest_model = NNModelKerasGuest(optimizer=optimizer,
                                        loss_func=loss,
                                        train_mode=train_mode,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size,
                                        use_mid=use_mid,
                                        mid_shape_in=mid_shape_in,
                                        mid_shape_out=mid_shape_out,
                                        mid_activation=mid_activation)
        guest_model.build_model_from_conf(btm_conf, mid_conf, top_conf)

        return guest_model

    @staticmethod
    def create_keras_guest_from_file(btm, mid, top, optimizer, loss,
                                     learning_rate, batch_size, use_mid,
                                     mid_shape_in, mid_shape_out,
                                     mid_activation,
                                     train_mode
                                     ):
        """
        param btm_conf: bottom model's conf json string
        param mid_conf: middle model's conf json string
        param top_conf: top model's conf json string
        return: keras guest model
        """
        guest_model = NNModelKerasGuest(optimizer=optimizer,
                                        loss_func=loss,
                                        train_mode=train_mode,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size,
                                        use_mid=use_mid,
                                        mid_shape_in=mid_shape_in,
                                        mid_shape_out=mid_shape_out,
                                        mid_activation=mid_activation)
        guest_model.build_model_from_path(btm, mid, top)
        return guest_model

    @staticmethod
    def create_keras_host_from_conf(btm_conf, optimizer, loss, learning_rate,
                                    batch_size, train_mode):
        """
        param btm_conf: bottom model's conf json string
        return: keras host model
        """
        host_model = NNModelKerasHost(optimizer=optimizer,
                                      loss_func=loss,
                                      train_mode=train_mode,
                                      learning_rate=learning_rate,
                                      batch_size=batch_size)
        host_model.build_model_from_conf(btm_conf)

        return host_model

    @staticmethod
    def create_keras_host_from_file(btm,
                                    optimizer,
                                    loss,
                                    learning_rate,
                                    batch_size=None,
                                    train_mode=None):
        """
        param btm_conf: bottom model's conf json string
        return: keras host model
        """
        host_model = NNModelKerasHost(optimizer=optimizer,
                                      loss_func=loss,
                                      train_mode=train_mode,
                                      learning_rate=learning_rate,
                                      batch_size=batch_size)
        host_model.build_model_from_path(btm)
        return host_model

    @staticmethod
    def initialize_model(role,
                         backend,
                         format,
                         train_mode,
                         btm=None,
                         mid=None,
                         top=None,
                         optimizer=None,
                         loss=None,
                         learning_rate=None,
                         batch_size=None,
                         use_mid=True,
                         mid_shape_in=None,
                         mid_shape_out=None,
                         mid_activation=None):
        if role == constant.TaskRole.GUEST:
            if backend == constant.NeuralNetwork.KERAS:
                if format == constant.NeuralNetwork.CONF:
                    return NNInitializer.create_keras_guest_from_conf(
                        btm, mid, top, optimizer, loss, learning_rate,
                        batch_size, use_mid, mid_shape_in, mid_shape_out,
                        mid_activation, train_mode)
                elif format == constant.NeuralNetwork.FILE:
                    return NNInitializer.create_keras_guest_from_file(
                        btm, mid, top, optimizer, loss, learning_rate,
                        batch_size, use_mid, mid_shape_in, mid_shape_out,
                        mid_activation, train_mode)
            elif backend == constant.NeuralNetwork.PYTORCH:
                if format == constant.NeuralNetwork.FILE:
                    return NNInitializer.create_pytorch_guest_from_file(
                        btm, mid, top, optimizer, loss, learning_rate,
                        batch_size, use_mid, mid_shape_in, mid_shape_out,
                        mid_activation)
            else:
                raise Exception('unsupported guest nn backend')

        elif role == constant.TaskRole.HOST:
            if backend == constant.NeuralNetwork.KERAS:
                if format == constant.NeuralNetwork.CONF:
                    return NNInitializer.create_keras_host_from_conf(
                        btm,
                        optimizer,
                        loss,
                        learning_rate,
                        batch_size=batch_size, train_mode=train_mode)
                elif format == constant.NeuralNetwork.FILE:
                    return NNInitializer.create_keras_host_from_file(
                        btm,
                        optimizer,
                        loss,
                        learning_rate,
                        batch_size=batch_size,
                        train_mode=train_mode)
            elif backend == constant.NeuralNetwork.PYTORCH:
                if format == constant.NeuralNetwork.FILE:
                    return NNInitializer.create_pytorch_host_from_file(
                        btm,
                        optimizer,
                        loss,
                        learning_rate,
                        batch_size=batch_size)
            else:
                raise Exception('unsupported host nn backend')

    @staticmethod
    def create_pytorch_guest_from_file(btm, mid, top, optimizer, loss,
                                       learning_rate, batch_size, use_mid,
                                       mid_shape_in, mid_shape_out,
                                       mid_activation

                                       ):
        """
        param btm_conf: bottom model's conf json string
        param mid_conf: middle model's conf json string
        param top_conf: top model's conf json string
        return: keras guest model
        """
        guest_model = VNNModelTorchGuest(optimizer=optimizer,
                                         loss_func=loss,
                                         learning_rate=learning_rate,
                                         batch_size=batch_size,
                                         use_mid=use_mid,
                                         mid_shape_in=mid_shape_in,
                                         mid_shape_out=mid_shape_out,
                                         mid_activation=mid_activation)
        guest_model.build_model_from_path(btm, mid, top)
        return guest_model

    @staticmethod
    def create_pytorch_host_from_file(btm,
                                      optimizer,
                                      loss,
                                      learning_rate,
                                      batch_size=None):
        """
        param btm_conf: bottom model's conf json string
        return: keras host model
        """
        host_model = VNNModelTorchHost(optimizer=optimizer,
                                       learning_rate=learning_rate,
                                       batch_size=batch_size)
        host_model.build_model_from_path(btm)
        return host_model
