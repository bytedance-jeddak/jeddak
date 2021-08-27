import copy
import os
import time

from common.factory.algorithm_factory import AlgorithmFactory
from common.factory.logger_factory import LoggerFactory
from common.factory.messenger_factory import MessengerFactory
from common.frame.parameter_frame.task_chain_parameter import TaskChainParameter
from common.util.random import Random
from common.util import constant
from coordinator.core.sqlite_manager import SqliteManager
from fl.algorithm import Algorithm
from fl.serving.model_loader import ModelLoader
from fl.serving.model_predict import ModelPredict
from fl.serving.online_predict import OnlinePredict
from fl.serving.model_inheritor import ModelInheritor


class TaskChain(object):
    """
    Task chain manager
    """

    def __init__(self, parameter_chain, task_chain_id=None, coordinator=None):
        """
        Generate a unique task chain id
        Initialize global components
        Initialize parameter chain
        :param parameter_chain: List[Parameter], which describes chain conf, task 0 conf, task 1 conf, ...
        :param task_chain_id: str
        """
        # check task chain validation
        self._check_parameter_chain_validation(parameter_chain)

        # generate task chain id if None
        if task_chain_id:
            self._task_chain_id = task_chain_id
        else:
            self._task_chain_id = TaskChain.generate_task_chain_id()

        # init global components
        self._task_chain_param = parameter_chain.pop(0)
        # list that contains of all parties' names in the task_chain including coordinator's
        self._party_names = self._task_chain_param.party_names
        # party name of the coordinator
        self._party_name = coordinator.party_name
        _other_party_names = copy.deepcopy(self._task_chain_param.party_names)
        _other_party_names.remove(self._party_name)
        # list that contains all other parties' names in the task_chain
        self._other_party_names = _other_party_names
        self._coordinator = coordinator
        self._save_model = self._task_chain_param.save_model
        self._global_init()

        # init param chain
        self._parameter_chain = parameter_chain

        # get logger
        self._byte_logger = LoggerFactory.get_global_instance()

    @property
    def parameter_chain(self):
        return [self._task_chain_param] + self._parameter_chain

    @property
    def save_model(self):
        return self._save_model

    def run(self):
        # cached data_io and model between every two consecutive tasks
        train_data_cache = None
        validate_data_cache = None
        model_cache = None
        model_idxes = []
        if self._save_model:
            task_chain_id = self._task_chain_id

            root_dir = os.path.split(os.path.realpath(__file__))[0]
            model_path = os.path.join(root_dir, '..', '..', 'common', 'model', self._party_name, task_chain_id)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

        is_stopped = False
        # run the tasks in a sequence
        for task_index, parameter in enumerate(self._parameter_chain):
            self._byte_logger.info("start task {}-{}: {}".format(self._task_chain_id,
                                                                 task_index,
                                                                 parameter.task_type))
            # check coordinator's action
            # update coordinator's status
            _task_id = self._get_task_id(task_index)

            algorithm = AlgorithmFactory.get_instance(
                parameter=parameter,
                all_parties=self._party_names,
                other_parties=self._other_party_names,
                this_party=self._party_name,
                task_id=_task_id,
                save=self._save_model
            )

            train_data_cache, validate_data_cache, model_cache = self._run_algorithm(algorithm, train_data_cache,
                                                                                     validate_data_cache, model_cache)

            # record the task progress
            SqliteManager.task_progress_dbo.create(dict(
                task_id=self._task_chain_id,
                progress_type=constant.BoardConstants.TASK_PROGRESS_MODULE,
                progress_value=parameter.task_type
            ))

            if self._save_model:
                self._byte_logger.info("task_chain saves model {}".format(_task_id))

                model_idxes.append(_task_id + ":" + parameter.task_type)
                algorithm.save_model(model_cache)
                self._byte_logger.info("task_chain saves model {} finish".format(_task_id))

            self._byte_logger.info(f"finish algorithm {self._task_chain_id}-{task_index}: {parameter.task_type}")
        self._byte_logger.info("finish task_chain {}".format(self._task_chain_id))
        if not is_stopped and self._save_model:
            model_file_name = os.path.join(model_path, task_chain_id + '.idx')
            with open(model_file_name, 'w') as f:
                for idx in model_idxes:
                    f.write(idx)
                    f.write("\n")

        MessengerFactory.get_global_instance().flush()
        return train_data_cache, validate_data_cache, model_cache

    def _run_algorithm(self, algorithm: Algorithm, input_data, validate_data, input_model=None):
        if type(algorithm) == ModelLoader or type(algorithm) == ModelInheritor:
            self._byte_logger.info('task_chain load model')
            output_data, output_model = algorithm.run_load_model(input_data, input_model)
            return output_data, None, output_model

        elif type(algorithm) == ModelPredict or type(algorithm) == OnlinePredict:
            self._byte_logger.info('task_chain predict')
            output_data, _ = algorithm.run_predict(input_data, input_model)
            return output_data, None, None

        if algorithm.task_type not in [constant.TaskType.LOGISTIC_REGRESSION, constant.TaskType.DPGBDT,
                                       constant.TaskType.NEURAL_NETWORK, constant.TaskType.POISSON_REGRESSION,
                                       constant.TaskType.LINEAR_REGRESSION]:
            need_train_and_validate = False
        else:
            need_train_and_validate = True
            algorithm.set_validate_data(validate_data)

        train_data, output_model = algorithm.train(input_data, input_model)
        if not need_train_and_validate:
            validate_data, _ = algorithm.validate(validate_data, evaluate=False)

        return train_data, validate_data, output_model

    @property
    def task_chain_id(self):
        return self._task_chain_id

    @staticmethod
    def generate_task_chain_id():
        return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + \
               Random.generate_random_digits()

    def _global_init(self):
        """
        Globally init
            byte logger factory,
            byte spark context factory,
            byte messenger factory
        """
        LoggerFactory.init(task_id=self._task_chain_id, party_name=self._party_name)

        MessengerFactory.init(self._task_chain_param.messenger_server,
                              self._task_chain_param.messenger_type,
                              self._party_name,
                              self._other_party_names,
                              task_chain_id=self._task_chain_id)

    def _check_parameter_chain_validation(self, parameter_chain):
        if type(parameter_chain) is not list or \
                type(parameter_chain[0]) != TaskChainParameter:
            raise ValueError("invalid parameter format, task_chain parameter should be at first")

    def _get_task_id(self, task_index):
        return str(self._task_chain_id) + '-' + str(task_index)
