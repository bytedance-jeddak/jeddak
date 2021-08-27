
from common.factory.factory import Factory
from common.util import constant
from fl.alignment.aligner import Aligner
from fl.data_io.data_loader import DataLoader
from fl.data_io.data_saver import DataSaver
from fl.gbdt.dpgbdt import DPGBDT
from fl.generalized_linear_model.linear_regression import LinearRegression
from fl.generalized_linear_model.logistic_regression import LogisticRegression
from fl.generalized_linear_model.poisson_regression import PoissonRegression
from fl.serving.model_loader import ModelLoader
from fl.serving.model_predict import ModelPredict
from fl.serving.online_predict import OnlinePredict
from fl.serving.model_inheritor import ModelInheritor
from fl.evaluate.evaluate import Evaluate
from fl.nn.nn import NeuralNetwork


class AlgorithmFactory(Factory):
    @staticmethod
    def get_instance(parameter, all_parties, other_parties, this_party, task_id=None, save=True):
        if parameter.task_type == constant.TaskType.TASK_CHAIN:
            raise ValueError("there is no specific task of task chain type")
        elif parameter.task_type == constant.TaskType.DATA_LOADER:
            algorithm = DataLoader(parameter)
        elif parameter.task_type == constant.TaskType.DATA_SAVER:
            algorithm = DataSaver(parameter)
        elif parameter.task_type == constant.TaskType.ALIGNER:
            algorithm = Aligner(parameter)
        elif parameter.task_type == constant.TaskType.LINEAR_REGRESSION:
            algorithm = LinearRegression(parameter)
        elif parameter.task_type == constant.TaskType.LOGISTIC_REGRESSION:
            algorithm = LogisticRegression(parameter)
        elif parameter.task_type == constant.TaskType.POISSON_REGRESSION:
            algorithm = PoissonRegression(parameter)
        elif parameter.task_type == constant.TaskType.DPGBDT:
            algorithm = DPGBDT(parameter)
        elif parameter.task_type == constant.TaskType.MODEL_LOADER:
            algorithm = ModelLoader(parameter)
        elif parameter.task_type == constant.TaskType.MODEL_PREDICT:
            algorithm = ModelPredict(parameter)
        elif parameter.task_type == constant.TaskType.EVALUATE:
            algorithm = Evaluate(parameter)
        elif parameter.task_type == constant.TaskType.ONLINE_PREDICT:
            algorithm = OnlinePredict(parameter)
        elif parameter.task_type == constant.TaskType.NEURAL_NETWORK:
            algorithm = NeuralNetwork(parameter)
        else:
            Factory._raise_value_error('Factory get_instance unknown task_type', parameter.task_type)

        algorithm.set_party_info(all_parties, other_parties, this_party)
        algorithm.set_task_id(task_id)
        algorithm.set_save_model(save)

        return algorithm
