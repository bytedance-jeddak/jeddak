from common.frame.parameter_frame.glm_parameter import GLMParameter
from common.util import constant


class LogisticRegressionParameter(GLMParameter):
    def __init__(self,
                 task_role=constant.TaskRole.GUEST,
                 penalty=constant.Regularizer.L2,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1.0,
                 solver=constant.Solver.GRADIENT_DESCENT,
                 max_iter=100,
                 learning_rate=0.15,
                 balanced_class_weight=True,
                 homomorphism=constant.Encryptor.CPAILLIER,
                 key_size=1024,
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

        :param balanced_class_weight: bool, automatically adjust weights inversely proportional to class frequencies in
                the input data as n_samples / (n_classes * np.bincount(y))
                auto-disabled for continuous labels
        """
        super(LogisticRegressionParameter, self).__init__(constant.TaskType.LOGISTIC_REGRESSION,
                                                          task_role,
                                                          penalty,
                                                          tol,
                                                          C,
                                                          fit_intercept,
                                                          intercept_scaling,
                                                          solver,
                                                          max_iter,
                                                          learning_rate,
                                                          homomorphism,
                                                          key_size,
                                                          gamma,
                                                          epsilon,
                                                          batch_fraction,
                                                          batch_type,
                                                          train_method,
                                                          privacy_method,
                                                          client_frac,
                                                          local_epoch_num,
                                                          local_batch_size,
                                                          train_validate_freq)

        self.balanced_class_weight = balanced_class_weight
