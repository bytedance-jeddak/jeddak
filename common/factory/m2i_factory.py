from common.factory.factory import Factory
from common.util import constant
from fl.alignment.aligner import Aligner
from fl.data_io.data_loader import DataLoader
from fl.data_io.data_saver import DataSaver
from fl.gbdt.dpgbdt import DPGBDT
from fl.generalized_linear_model.linear_regression import LinearRegression
from fl.generalized_linear_model.logistic_regression import LogisticRegression
from fl.generalized_linear_model.poisson_regression import PoissonRegression
from fl.evaluate.evaluate import Evaluate
from fl.nn.nn import NeuralNetwork


class Model2InstanceFactory(Factory):
    @staticmethod
    def get_instance(model_type, model):
        if model_type == constant.TaskType.DATA_LOADER:
            algorithm = DataLoader.model_to_instance(model)
        elif model_type == constant.TaskType.DATA_SAVER:
            algorithm = DataSaver.model_to_instance(model)
        elif model_type == constant.TaskType.ALIGNER:
            algorithm = Aligner.model_to_instance(model)
        elif model_type == constant.TaskType.LINEAR_REGRESSION:
            algorithm = LinearRegression.model_to_instance(model)
        elif model_type == constant.TaskType.LOGISTIC_REGRESSION:
            algorithm = LogisticRegression.model_to_instance(model)
        elif model_type == constant.TaskType.POISSON_REGRESSION:
            algorithm = PoissonRegression.model_to_instance(model)
        elif model_type == constant.TaskType.DPGBDT:
            algorithm = DPGBDT.model_to_instance(model)
        elif model_type == constant.TaskType.EVALUATE:
            algorithm = Evaluate.model_to_instance(model)
        elif model_type == constant.TaskType.NEURAL_NETWORK:
            algorithm = NeuralNetwork.model_to_instance(model)

        else:
            raise Exception('Factory get_instance unkown task_type', model_type)
        return algorithm
