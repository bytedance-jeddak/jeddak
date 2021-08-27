from common.factory.factory import Factory
from common.frame.parameter_frame.model_predict_parameter import ModelPredictParameter
from common.frame.parameter_frame.online_predict_parameter import OnlinePredictParameter
from common.frame.parameter_frame.model_load_parameter import ModelLoadParameter
from common.frame.parameter_frame.aligner_parameter import AlignerParameter
from common.frame.parameter_frame.data_loader_parameter import DataLoaderParameter
from common.frame.parameter_frame.data_saver_parameter import DataSaverParameter
from common.frame.parameter_frame.dpgbdt_parameter import DPGBDTParameter
from common.frame.parameter_frame.linear_regression_parameter import LinearRegressionParameter
from common.frame.parameter_frame.logistic_regression_parameter import LogisticRegressionParameter
from common.frame.parameter_frame.poisson_regression_parameter import PoissonRegressionParameter
from common.frame.parameter_frame.task_chain_parameter import TaskChainParameter
from common.frame.parameter_frame.evaluate_parameter import EvaluateParameter
from common.frame.parameter_frame.nn_parameter import NNParameter
from common.util import constant


class ParameterFactory(Factory):
    Task_Parameter_Map = {
        constant.TaskType.TASK_CHAIN: TaskChainParameter,
        constant.TaskType.DATA_LOADER: DataLoaderParameter,
        constant.TaskType.DATA_SAVER: DataSaverParameter,
        constant.TaskType.ALIGNER: AlignerParameter,
        constant.TaskType.LINEAR_REGRESSION: LinearRegressionParameter,
        constant.TaskType.LOGISTIC_REGRESSION: LogisticRegressionParameter,
        constant.TaskType.POISSON_REGRESSION: PoissonRegressionParameter,
        constant.TaskType.DPGBDT: DPGBDTParameter,
        constant.TaskType.MODEL_LOADER: ModelLoadParameter,
        constant.TaskType.MODEL_PREDICT: ModelPredictParameter,
        constant.TaskType.EVALUATE: EvaluateParameter,
        constant.TaskType.ONLINE_PREDICT: OnlinePredictParameter,
        constant.TaskType.NEURAL_NETWORK: NNParameter,
    }

    @staticmethod
    def get_instance(parameter_dict):
        task_type = parameter_dict.pop('task_type')
        parameter_type = ParameterFactory.Task_Parameter_Map.get(task_type, None)
        if parameter_type:
            return parameter_type(**parameter_dict)
        else:
            Factory._raise_value_error('ParameterFactory task_type', task_type)
