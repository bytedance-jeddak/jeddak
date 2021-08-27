from common.frame.data_frame.sample import Sample
from common.frame.message_frame.poisson_regression_message import PoissonRegressionMessage
from common.frame.model_frame.poisson_regression_model import PoissonRegressionModel
from common.frame.parameter_frame.poisson_regression_parameter import PoissonRegressionParameter
from common.util import constant
from fl.generalized_linear_model.glm import GLM
from fl.operator.evaluator import Evaluator
from fl.operator.link_function import LinkFunction


class PoissonRegression(GLM):
    def __init__(self, parameter: PoissonRegressionParameter, message=PoissonRegressionMessage()):
        super(PoissonRegression, self).__init__(parameter, message=message)

        self._link_function = LinkFunction(constant.LinkType.POISSON)
        self._evaluator = Evaluator(constant.EvaluationType.MULTICLASS)

    def _parse_label(self, input_data):
        def parse_label_for_each_row(sample: Sample):
            sample.label = int(sample.label)
            return sample

        parsed_data = input_data.mapValues(parse_label_for_each_row)
        return parsed_data

    def instance_to_model(self):
        return PoissonRegressionModel(self._parameter, self.get_party_info(),
                                      self._coef, self._parameter.intercept_scaling, self._intercept)

    @staticmethod
    def model_to_instance(model):
        algorithm_instance = PoissonRegression(model.parameter)

        algorithm_instance.set_party_info(*model.party_info)

        algorithm_instance.set_coef(model.coef)
        algorithm_instance.set_intercept_scaling(model.intercept_scaling)
        algorithm_instance.set_intercept(model.intercept)

        return algorithm_instance

