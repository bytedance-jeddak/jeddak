from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class DPGBDTParameter(Parameter):
    def __init__(self,
                 task_role,
                 objective=constant.Objective.BINARY_LOGISTIC,
                 base_score=0.5,
                 num_round=20,
                 eta=0.3,
                 gamma=0.0,
                 max_depth=6,
                 min_child_weight=1.0,
                 max_delta_step=0,
                 sub_sample=1.0,
                 lam=1.0,
                 sketch_eps=0.03,
                 homomorphism=constant.Encryptor.CPAILLIER,
                 key_size=768,
                 privacy_budget=10.0,
                 dp_mech=constant.DifferentialPrivacyMechanism.LAPLACE,
                 privacy_mode=constant.PrivacyMode.ADA,
                 importance_type='weight',
                 first_order_approx=False,
                 train_validate_freq=None):
        """

        :param task_role:
        :param objective: learning objective
        :param base_score: initial prediction score
        :param num_round: the number of boosting rounds
        :param eta: learning rate
        :param gamma: minimum loss reduction required to make a further partition on a leaf node of the tree
        :param max_depth: maximum depth of a tree
        :param min_child_weight: minimum sum of instance weight (hessian) needed in a child.
                If the tree partition step results in a leaf node with the sum of instance weight
                less than min_child_weight, then the building process will give up further partitioning.
        :param max_delta_step: maximum delta step we allow each leaf output to be.
                If the value is set to 0, it means there is no constraint. In effect with default 0.7 for Poisson
        :param sub_sample: subsample ratio of the training instances at each boosting iteration
        :param lam: L2 regularization strength
        :param sketch_eps: convert every column into 1 / sketch_eps number of bins at most
        :param homomorphism: homomorphic encryptor type
        :param key_size: homomorphic key length
        :param privacy_budget: differential privacy budget
        :param dp_mech: differential privacy mechanism
        :param privacy_mode: privacy-preserving technique
        :param importance_type: feature importance type, e.g.,
                "weight", "gain", "cover", "total_gain", "total_cover", "all"
        :param first_order_approx: True if first-order-approximating gain calculation method is adopted, otherwise
                                    second-order
        :param train_validate_freq validation frequency using validate data while train, type int, default None
        """
        super(DPGBDTParameter, self).__init__(constant.TaskType.DPGBDT, task_role)

        self.objective = objective
        self.base_score = base_score
        self.num_round = num_round
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.sub_sample = sub_sample
        self.lam = lam
        self.sketch_eps = sketch_eps
        self.homomorphism = homomorphism
        self.key_size = key_size
        self.privacy_budget = privacy_budget
        self.dp_mech = dp_mech
        self.privacy_mode = privacy_mode
        self.importance_type = importance_type
        self.first_order_approx = first_order_approx

        self.train_validate_freq = train_validate_freq
