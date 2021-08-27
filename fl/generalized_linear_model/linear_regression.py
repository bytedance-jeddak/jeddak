from common.frame.message_frame.linear_regression_message import LinearRegressionMessage
from common.frame.model_frame.linear_regression_model import LinearRegressionModel
from common.frame.parameter_frame.linear_regression_parameter import LinearRegressionParameter
from common.util import constant
from fl.generalized_linear_model.glm import GLM
from fl.operator.evaluator import Evaluator
from fl.operator.link_function import LinkFunction


class LinearRegression(GLM):
    def __init__(self, parameter: LinearRegressionParameter, message=LinearRegressionMessage()):
        super(LinearRegression, self).__init__(parameter, message=message)

        self._link_function = LinkFunction(constant.LinkType.LINEAR)
        self._evaluator = Evaluator(constant.EvaluationType.REGRESSION)

    def _parse_label(self, input_data):
        return input_data

    def instance_to_model(self):
        return LinearRegressionModel(self._parameter, self.get_party_info(),
                                     self._coef, self._parameter.intercept_scaling, self._intercept)

    @staticmethod
    def model_to_instance(model):
        algorithm_instance = LinearRegression(model.parameter)

        algorithm_instance.set_party_info(*model.party_info)

        algorithm_instance.set_coef(model.coef)
        algorithm_instance.set_intercept_scaling(model.intercept_scaling)
        algorithm_instance.set_intercept(model.intercept)

        return algorithm_instance
