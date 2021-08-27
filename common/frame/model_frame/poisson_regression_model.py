from common.frame.model_frame.glm_model import GLMModel


class PoissonRegressionModel(GLMModel):
    def __init__(self, parameter, party_info, coef, intercept_scaling, intercept=None):
        super(PoissonRegressionModel, self).__init__(parameter, party_info, coef, intercept_scaling, intercept)
