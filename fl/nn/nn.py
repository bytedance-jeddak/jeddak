import functools
import json
import os
from datetime import datetime

from tensorflow.keras.utils import to_categorical

from common.frame.data_frame.c_dataset import CDataset
from common.frame.data_frame.sample import Sample
from common.frame.data_frame.schema import Schema
from common.frame.message_frame.nn_message import NNMessage
from common.frame.model_frame.neural_network_model import NeuralNetworkModel
from common.frame.parameter_frame.nn_parameter import NNParameter
from common.util import constant
from coordinator.core.sqlite_manager import SqliteManager
from fl.algorithm import Algorithm
from fl.nn.backend.keras.util import auc
from fl.nn.dataset_loader import dataset_loader_initializer
from fl.nn.initializer import NNInitializer
from fl.nn.interactive.interactive_phe import NNInteractivePHE
from fl.operator.evaluator import Evaluator


class NeuralNetwork(Algorithm):
    """
    DNN
    """

    def __init__(self, parameter: NNParameter, message=NNMessage()):
        super(NeuralNetwork, self).__init__(parameter, message=message)

        # config
        self._parameter = parameter
        self._input_feature_dim = None

        # model
        self._model = None
        self._fed_average = None

        # communication
        self._interactor = None

        # predict
        self._predict_suffix = None
        self.train_test_split_ratio = 0.8

        self._evaluator = Evaluator(constant.EvaluationType.BINARY)

    def _build_models(self, input_feature_dim):
        self._logger.info("start build_models, train mode:{}".format(self._parameter.train_mode))
        role = self.get_this_party_role()
        if role == constant.TaskRole.GUEST or role == constant.TaskRole.HOST:
            self._interactor = NNInteractivePHE(
                self._messenger,
                self._message,
                self._parameter.learning_rate,
                guest_party=self.get_all_guest_names(),
                host_party=self.get_all_host_names(),
                role=role,
                privacy_mode=self._parameter.privacy_mode
            )

        self._logger.info(
            "finish init interactor, privacy_model:{}".format(
                self._parameter.privacy_mode))

        print("input_feature_dim:{}".format(input_feature_dim))
        self._input_feature_dim = input_feature_dim
        self._allocate_task(guest_task=self._build_role_model,
                            host_task=self._build_role_model, )

        self._logger.info("end build_models")

    def _build_role_model(self):
        """
        Build model for guest/host
        """
        self._model = NNInitializer.initialize_model(
            self.get_this_party_role(),
            backend=self._parameter.backend,
            format=self._parameter.format,
            train_mode=self._parameter.train_mode,
            btm=self._parameter.btm,
            mid=self._parameter.mid,
            top=self._parameter.top,
            optimizer=self._parameter.optimizer,
            loss=self._parameter.loss_fn,
            learning_rate=self._parameter.learning_rate,
            batch_size=self._parameter.batch_size,
            use_mid=self._parameter.use_mid,
            mid_shape_in=self._parameter.mid_shape_in,
            mid_shape_out=self._parameter.mid_shape_out,
            mid_activation=self._parameter.mid_activation)

        self._model.interactor = self._interactor

    def train(self, input_data=None, input_model=None):
        """
        :param input_data:
        :param input_model:
        :return: output_data, output_model
        """
        # train
        self._logger.info("start training")

        result = self._allocate_task(
            guest_task=functools.partial(self._train_guest,
                                         input_data=input_data,
                                         input_model=input_model),
            host_task=functools.partial(self._train_host,
                                        input_data=input_data,
                                        input_model=input_model), )

        self._logger.info("end training")

        return result

    def predict(self, input_data=None):
        """
        :param input_data: Dataset or None
        :return:
        """
        self._logger.info("start predicting")

        result = self._allocate_task(
            guest_task=functools.partial(self._predict_guest, input_data=input_data),
            host_task=functools.partial(self._predict_host, input_data=input_data),
        )

        self._logger.info("end predicting")

        return result

    def _train_guest(self, input_data, input_model):
        """
        Train guest model
        :param input_data: Dataset
        :param input_model:
        :return:
        """
        self._build_models(input_data.feature_dimension)

        dataset_loader = dataset_loader_initializer(input_data, self._parameter.train_mode, self._parameter.partitions)
        # train_test_split_ratio = self.train_test_split_ratio
        # dataset_loader.set_split_ratio(train_test_split_ratio)
        batches = dataset_loader.get_batches(self._parameter.batch_size)

        old_time4db = datetime.now()

        for e in range(self._parameter.epochs):
            # init & get train_auc and validate_auc
            train_auc = 0
            validate_auc = 0
            if self._parameter.train_validate_freq is not None and e % self._parameter.train_validate_freq == 0:
                train_auc = self.validate(input_data)[1][0]["auc"]
                self._logger.info("start validation, evaluate train data auc:{}".format(train_auc))
                validate_result = self.validate()[1]
                if validate_result:
                    validate_auc = validate_result[0]["auc"]
                    self._logger.info("start validation, evaluate validate data:{}".format(validate_auc))

            for idx in range(batches):
                self._logger.info("guest, epochs:{}, iteration:{}".format(
                    e, idx))

                self._interactor.epoch = e
                self._interactor.batch = idx
                x_batch, y_batch = dataset_loader.get_train_batch(
                    self._parameter.batch_size, idx)
                if self._parameter.train_mode == constant.NeuralNetwork.LOCAL:
                    if self._parameter.predict_model == "categorical":
                        y_batch = to_categorical(
                            y_batch, self._parameter.num_classes)
                    else:
                        y_batch = y_batch.reshape(-1, 1)

                self._model.train(x_batch, y_batch)

            # if not self._parameter.use_async:
            print("loss guest: " + str(self._model.current_loss))

            new_time4db = datetime.now()
            SqliteManager.task_progress_dbo.create(args=dict(
                task_id=self._task_chain_id,
                progress_type='loss',
                progress_value=json.dumps(dict(
                    loss=self._model.current_loss,
                    time=(new_time4db - old_time4db).seconds,
                    train_auc=train_auc,
                    validate_auc=validate_auc
                ))
            ))
            old_time4db = new_time4db
        print("loss guest: " + str(self._model.current_loss))
        return input_data, self.instance_to_model()

    def _train_host(self, input_data, input_model):
        """
        Train host model
        :param input_data: Dataset
        :param input_model:
        :return:
        """
        self._build_models(input_data.feature_dimension)

        dataset_loader = dataset_loader_initializer(input_data, self._parameter.train_mode, self._parameter.partitions)
        # train_test_split_ratio = self.train_test_split_ratio
        # dataset_loader.set_split_ratio(train_test_split_ratio)
        batches = dataset_loader.get_batches(self._parameter.batch_size)

        old_time4db = datetime.now()
        for e in range(self._parameter.epochs):
            if self._parameter.train_validate_freq is not None and e % self._parameter.train_validate_freq == 0:
                self.validate(input_data)
                self.validate()

            for idx in range(batches):
                self._interactor.epoch = e
                self._interactor.batch = idx
                self._logger.info("host, epochs:{}, iteration:{}".format(
                    e, idx))
                x_batch, _ = dataset_loader.get_train_batch(
                    self._parameter.batch_size, idx)
                self._model.train(x_batch)

            # save task progress to db
            new_time4db = datetime.now()
            SqliteManager.task_progress_dbo.create(args=dict(
                task_id=self._task_chain_id,
                progress_type=constant.BoardConstants.TASK_PROGRESS_LOSS,
                progress_value=json.dumps(dict(
                    loss=0,
                    time=(new_time4db - old_time4db).seconds
                ))
            ))
            old_time4db = new_time4db

        return input_data, self.instance_to_model()

    def _predict_guest(self, input_data):
        """
        :param input_data: Dataset
        :return: Dataset, predict results
        """
        dataset_loader = dataset_loader_initializer(input_data, self._parameter.train_mode)
        ids, sample_data, _ = dataset_loader.dataset_to_np_arrays_1(input_data)

        pred_res = self._get_predict_guest(sample_data)
        res = [[s, p] for s, p in zip(ids, pred_res)]

        return self._convert_to_c_dataset(res, schema=input_data.schema)

    def _predict_host(self, input_data):
        """
        :param input_data: Dataset
        :return: 
        """
        dataset_loader = dataset_loader_initializer(input_data, self._parameter.train_mode)
        ids, sample_data, _ = dataset_loader.dataset_to_np_arrays_1(input_data)

        self._get_predict_host(sample_data)

        return input_data

    @staticmethod
    def model_to_instance(model):
        algorithm_instance = NeuralNetwork(model.parameter)
        algorithm_instance.set_party_info(*model.party_info)
        algorithm_instance._build_models(model.input_feature_dim)
        print("model feature_input_shape:{}".format(
            model.input_feature_dim
        ))
        role = algorithm_instance.get_this_party_role()
        if role == constant.TaskRole.GUEST:
            algorithm_instance._model.load_model(model.btm_model_path,
                                                 model.mid_model_path,
                                                 model.top_model_path)

        elif role == constant.TaskRole.HOST:
            algorithm_instance._model.load_model(model.btm_model_path)
            if model.parameter.privacy_mode in constant.NeuralNetwork.SUPPORT_PRIVACY_MODE:
                algorithm_instance._interactor.set_noise_acc = model.noise_acc

        else:
            raise ValueError("Unsupported Task Role")

        return algorithm_instance

    def instance_to_model(self):
        if not self._save_model:
            return None
        root_dir = os.path.split(os.path.realpath(__file__))[0]
        model_path = os.path.join(root_dir, '', '..', '..', 'common', 'model',
                                  self._this_party, self._task_chain_id)

        this_party_role = self.get_this_party_role()
        if this_party_role == constant.TaskRole.GUEST:
            btm_model_path = os.path.join(model_path, "bottom_model")
            mid_model_path = os.path.join(model_path, "mid_model")
            top_model_path = os.path.join(model_path, "top_model")

            self._model.save_model(btm_model_path, mid_model_path,
                                   top_model_path)

            return NeuralNetworkModel(parameter=self._parameter,
                                      party_info=self.get_party_info(),
                                      btm_model_path=btm_model_path,
                                      mid_model_path=mid_model_path,
                                      top_model_path=top_model_path,
                                      input_feature_dim=self._input_feature_dim)

        elif this_party_role == constant.TaskRole.HOST:
            btm_model_path = os.path.join(model_path, "bottom_model")
            self._model.save_model(btm_model_path)

            noise_acc = self._interactor.get_noise_acc() if \
                self._parameter.privacy_mode in constant.NeuralNetwork.SUPPORT_PRIVACY_MODE else None

            return NeuralNetworkModel(parameter=self._parameter,
                                      party_info=self.get_party_info(),
                                      btm_model_path=btm_model_path,
                                      noise_acc=noise_acc,
                                      input_feature_dim=self._input_feature_dim)
        else:
            raise ValueError("Not support role {} in save model.".format(
                self._this_party))

    @staticmethod
    def _convert_to_c_dataset(data, schema=None):
        dataset = CDataset(data).mapValues(lambda val: Sample(label=val))
        dataset.schema = Schema(id_name=schema.id_name, label_name='y_pred')
        return dataset

    @staticmethod
    def _eval(y_true, y_pred):
        # temp for test
        return auc(y_true, y_pred)

    def _get_predict_guest(self, input_data):
        """
        :param input_data: numpy.ndarray
        :return:
        """
        if input_data is None:
            return

        if self._predict_suffix is None:
            self._predict_suffix = [-1, 0]
        self._interactor.epoch, self._interactor.batch = self._predict_suffix
        self._predict_suffix[1] += 1
        return self._model.predict(input_data)

    def _get_predict_host(self, input_data):
        if input_data is None:
            return

        if self._predict_suffix is None:
            self._predict_suffix = [-1, 0]
        self._interactor.epoch, self._interactor.batch = self._predict_suffix
        self._predict_suffix[1] += 1
        self._model.predict(input_data)
