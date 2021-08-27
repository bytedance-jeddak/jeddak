from common.frame.model_frame.model import Model


class GLMModel(Model):
    def __init__(self, parameter, party_info, coef, intercept_scaling, intercept=None):
        super(GLMModel, self).__init__(parameter, party_info)

        self._coef = coef
        self._intercept_scaling = intercept_scaling
        self._intercept = intercept

    @property
    def coef(self):
        return self._coef

    @property
    def intercept_scaling(self):
        return self._intercept_scaling

    @property
    def intercept(self):
        return self._intercept
