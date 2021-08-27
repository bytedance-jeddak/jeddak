from common.frame.model_frame.glm_model import GLMModel


class LinearRegressionModel(GLMModel):
    def __init__(self, parameter, party_info, coef, intercept_scaling, intercept=None):
        super(LinearRegressionModel, self).__init__(parameter, party_info, coef, intercept_scaling, intercept)
