from common.frame.model_frame.model import Model


class NeuralNetworkModel(Model):
    def __init__(self, parameter, party_info, btm_model_path=None, mid_model_path=None, top_model_path=None,
                 nn_model_path=None, noise_acc=None, input_feature_dim=None, use_mid=True):
        super(NeuralNetworkModel, self).__init__(parameter, party_info)

        # nn model
        self._input_feature_dim = input_feature_dim

        # guest-host
        self._btm_model_path = btm_model_path
        self._mid_model_path = mid_model_path
        self._top_model_path = top_model_path
        self._noise_acc = noise_acc

        # server-client
        self._nn_model_path = nn_model_path
        
        self._use_mid = use_mid

    @property
    def btm_model_path(self):
        return self._btm_model_path

    @property
    def mid_model_path(self):
        return self._mid_model_path

    @property
    def top_model_path(self):
        return self._top_model_path

    @property
    def noise_acc(self):
        return self._noise_acc

    @property
    def nn_model_path(self):
        return self._nn_model_path

    @property
    def input_feature_dim(self):
        return self._input_feature_dim

    @property
    def use_mid(self):
        return self._use_mid
