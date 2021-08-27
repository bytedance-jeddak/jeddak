from common.frame.model_frame.glm_model import GLMModel


class LogisticRegressionModel(GLMModel):
    def __init__(self, parameter, party_info, coef, intercept_scaling, intercept=None):
        super(LogisticRegressionModel, self).__init__(parameter, party_info, coef, intercept_scaling, intercept)
