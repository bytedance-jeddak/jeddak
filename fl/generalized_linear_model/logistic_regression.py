from common.frame.data_frame.sample import Sample
from common.frame.message_frame.logistic_regression_message import LogisticRegressionMessage
from common.frame.model_frame.logistic_regression_model import LogisticRegressionModel
from common.frame.parameter_frame.logistic_regression_parameter import LogisticRegressionParameter
from common.util import constant
from fl.generalized_linear_model.glm import GLM
from fl.operator.evaluator import Evaluator
from fl.operator.link_function import LinkFunction


class LogisticRegression(GLM):
    def __init__(self, parameter: LogisticRegressionParameter, message=LogisticRegressionMessage):
        super(LogisticRegression, self).__init__(parameter, message=message)

        self._link_function = LinkFunction(constant.LinkType.LOGISTIC)
        self._evaluator = Evaluator(constant.EvaluationType.BINARY)

    def _parse_label(self, input_data):
        def parse_label_for_each_row(sample: Sample):
            sample.label = int(sample.label)
            if sample.label == -1:
                sample.label = 0
            return sample

        parsed_data = input_data.mapValues(parse_label_for_each_row)
        return parsed_data

    def instance_to_model(self):
        return LogisticRegressionModel(self._parameter, self.get_party_info(),
                                       self._coef, self._parameter.intercept_scaling, self._intercept)

    @staticmethod
    def model_to_instance(model):
        algorithm_instance = LogisticRegression(model.parameter)

        algorithm_instance.set_party_info(*model.party_info)

        algorithm_instance.set_coef(model.coef)
        algorithm_instance.set_intercept_scaling(model.intercept_scaling)
        algorithm_instance.set_intercept(model.intercept)

        return algorithm_instance

