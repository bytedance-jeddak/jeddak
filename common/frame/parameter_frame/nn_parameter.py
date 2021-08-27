from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class NNParameter(Parameter):
    def __init__(self,
                 task_role,
                 backend="keras",
                 format="conf",
                 btm=None,
                 mid=None,
                 top=None,
                 epochs=1,
                 batch_size=32,
                 optimizer="SGD",
                 learning_rate=0.001,
                 loss_fn="categorical_crossentropy",
                 metrics=None,
                 use_mid=True,
                 mid_shape_in=64,
                 mid_shape_out=32,
                 mid_activation="linear",
                 use_async=False,
                 privacy_mode="paillier",
                 privacy_budget=2,
                 predict_model="categorical",
                 num_classes=None,
                 client_frac=1.0,
                 model_conf=None,
                 train_mode="local",
                 partitions=None,
                 train_validate_freq=None):
        super(NNParameter, self).__init__(constant.TaskType.NEURAL_NETWORK,
                                          task_role)

        # nn parameter
        self.backend = backend
        self.format = format
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.use_mid = use_mid
        self.mid_shape_in = mid_shape_in
        self.mid_shape_out = mid_shape_out
        self.mid_activation = mid_activation
        self.predict_model = predict_model
        self.num_classes = num_classes
        self.train_validate_freq = train_validate_freq

        # distribution
        self.train_mode = train_mode
        self.partitions = partitions

        # guest-host parameter
        self.btm = btm
        self.mid = mid
        self.top = top
        self.privacy_mode = privacy_mode
        self.use_async = use_async
        self.privacy_budget = privacy_budget

        # server-client parameter
        self.client_frac = client_frac
        self.model_conf = model_conf