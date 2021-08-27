from common.frame.parameter_frame.parameter import Parameter
from common.frame.parameter_frame.fed_ave_parameter import FedAveParameter
from common.util import constant


class GLMParameter(Parameter, FedAveParameter):
    def __init__(self,
                 task_type,
                 task_role,
                 penalty=constant.Regularizer.L2,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1.0,
                 solver=constant.Solver.GRADIENT_DESCENT,
                 max_iter=100,
                 learning_rate=0.15,
                 homomorphism=constant.Encryptor.CPAILLIER,
                 key_size=768,
                 gamma=0.9,
                 epsilon=1e-8,
                 batch_fraction=0.1,
                 batch_type="batch",
                 train_method="homo",
                 privacy_method="centralized",
                 client_frac=1.0,
                 local_epoch_num=1,
                 local_batch_size=1.0,
                 train_validate_freq=None):
        """

        :param task_role:
        :param penalty: str, constant.REGULARIZER.L1, constant.REGULARIZER.L2 or None
        :param tol: float, tolerance for stopping criteria
        :param C: float, inverse of regularization strength
        :param fit_intercept: bool, bias
        :param intercept_scaling: float, x becomes [x, self.intercept_scaling] if fit_intercept is true
        :param solver: str, constant.SOLVER.GRADIENT_DESCENT
        :param max_iter: int
        :param learning_rate: float
        :param homomorphism: str, constant.ENCRYPTOR.constant.ENCRYPTOR.PAILLIER
        :param key_size: int
        :param gamma: float, adjust the sum of past squared gradients, around 0.9
        :param epsilon: float, a smoothing term that avoids division by zero
        :param batch_fraction: float, the subset fraction of mini-batch training
        :param batch_type: string, batch or mini-batch
        :param train_method: string, homo or smm
        :param privacy_method: string, centralized or distributed
        :param client_frac: FedAve, the fraction of clients selected to update the global model
        :param local_epoch_num: FedAve, the number of local epochs
        :param local_batch_size: FedAve, the number of local batch size
        :param train_validate_freq validation frequency using validate data while train, type int, default None
        """
        super(GLMParameter, self).__init__(task_type, task_role)

        FedAveParameter.__init__(self, client_frac, local_epoch_num, local_batch_size)

        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.solver = solver
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.homomorphism = homomorphism
        self.key_size = key_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_fraction = batch_fraction
        self.batch_type = batch_type
        self.train_method = train_method
        self.privacy_method = privacy_method

        self.train_validate_freq = train_validate_freq
