class KeyWord:
    TASK_TYPE = 'task_type'
    TASK_ROLE = 'task_role'
    PARTY_NAMES = 'party_names'
    SAVE_MODEL = 'save_model'


class MessageThreshold:
    MAX_MESSAGE_SIZE = 512000


class MessengerType:
    KAFKA = 'kafka'
    LIGHT_KAFKA = 'light_kafka'


class TaskType:
    TASK_CHAIN = 'task_chain'
    DATA_LOADER = 'data_loader'
    DATA_SAVER = 'data_saver'
    ALIGNER = 'aligner'
    LINEAR_REGRESSION = 'linear_regression'
    LOGISTIC_REGRESSION = 'logistic_regression'
    POISSON_REGRESSION = 'poisson_regression'
    DPGBDT = 'dpgbdt'
    NEURAL_NETWORK = 'neural_network'
    MODEL_LOADER = 'model_loader'
    MODEL_PREDICT = 'predict_offline'
    ONLINE_PREDICT = 'predict_online'
    EVALUATE = 'evaluate'


class DataSource:
    CSV = 'csv'
    RAW = 'raw'


class DataCarrier:
    NdArray = 'NdArray'


class TaskRole:
    # GUEST and HOST appear in a task
    GUEST = 'guest'             # parties holding labels, sometimes also features
    HOST = 'host'               # parties holding only features
    # SERVER and CLIENT appear in a task
    SERVER = 'server'           # parties aggregating distributed variables
    CLIENT = 'client'           # parties performing local computation
    # SOLE and SLACK appear in a task
    SOLE = 'sole'               # parties that independently execute tasks
    SLACK = 'slack'             # parties that are slack and away from task execution


class TaskChainID:
    DEFAULT_CHAIN_ID = 'default_chain_id'


class Encryptor:
    PLAIN = 'plain'
    DIFFIE_HELLMAN = 'diffie_hellman'
    CM20 = 'cm20'
    CM20_Parallelized = 'cm20_parallelized'
    CPAILLIER = 'cpaillier'


class Regularizer:
    DISABLE = 'disable'
    L1 = 'l1'
    L2 = 'l2'


class Solver:
    GRADIENT_DESCENT = 'gradient_descent'
    ADA_GRAM = 'AdaGram'
    ADA_DELTA = 'AdaDelta'
    RMS_PROP = 'RMSprop'


class LinkType:
    LINEAR = 'linear'
    LOGISTIC = 'logistic'
    POISSON = 'poisson'


class EvaluationType:
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'
    CLUSTER = 'cluster'


class Metric:
    ACCURACY = 'accuracy'
    AUC = 'auc'
    AUPR = 'aupr'
    TPR = 'tpr'
    FPR = 'fpr'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'f1_score'
    MAE = 'mae'
    MSE = 'mse'
    RMSE = 'rmse'
    DBI = 'dbi'


class Objective:
    REG_SQUAREDERROR = 'reg_squarederror'
    BINARY_LOGISTIC = 'binary_logistic'
    COUNT_POISSON = 'count_poisson'


class BucketizerType:
    QUANTILE = 'quantile'
    BUCKET = 'bucket'


class DifferentialPrivacyMechanism:
    LAPLACE = 'laplace'
    PIECEWISE = 'piecewise'
    DUCHI = 'duchi'
    HYBRID = 'hybrid'


class LossType:
    GBDT = 'gbdt'


class EarlyStopType:
    SMALL_LOSS_REDUCTION = 'small_loss_reduction'
    SMALL_CHILD_WEIGHT = 'small_child_weight'
    MIN_CHILD_WEIGHT = 'min_child_weight'


class LogLevel:
    INFO = 'info'
    DEBUG = 'debug'
    ERROR = 'error'


class DataFrameType:
    GH_PAIR = 'gh_pair'


class TaskStatus:
    READY = 'ready'
    RUN = 'run'
    FINISH = 'finish'
    ERROR = 'error'
    ABORT = 'abort'


class TaskAction:
    WAIT = 'wait'
    RUN = 'run'
    STOP = 'stop'


class PrivacyMode:
    HOMO = 'homo'               # Homomorphism
    DP = 'dp'
    LDP = 'ldp'
    ADA = 'ada'


class DistributionType:
    GAUSSIAN = 'gaussian'
    LAPLACE = 'laplace'
    POISSON = 'poisson'
    UNIFORM = 'uniform'


class StatisticalDistanceType:
    KL_DIVERGENCE = 'kl_divergence'
    TOTAL_VARIATION_DISTANCE = 'total_variation_distance'
    RENYI_DIVERGENCE = 'renyi_divergence'


class ModelLoaderAction:
    LOAD_MODEL = "load"
    UNLOAD_MODEL = "unload"


class SecurityConfig:
    SASL_PLAINTEXT = 'SASL_PLAINTEXT'
    SASL_SSL = 'SASL_SSL'
    SASL_MECHANISMS = 'PLAIN'
    SASL_USERNAME = 'admin'


class PrivacyRole:
    LEADER = 'leader'
    FOLLOWER = 'follower'


class PrivacyMethod:
    DISTRIBUTED = 'distributed'
    CENTRALIZED = 'centralized'


class NeuralNetwork:
    BACKEND = "backend"
    KERAS = "keras"
    FORMAT = "format"
    CONF = "conf"
    FILE = "file"
    BTM = "btm"
    TOP = "top"
    MID = "mid"
    INTERACTIVE = "interactive"
    # PHE = "phe"
    SUPPORT_PRIVACY_MODE = ["paillier", "shac", "real_vomo", "plain"]
    PLAIN = "plain"
    LDP = "ldp"
    OPTIMIZER = "optimizer",
    LOSS_FN = "loss_fn",
    LEARNING_RATE = "learning_rate"
    PYTORCH = "pytorch"
    LOCAL = "local"
    DISTRIBUTION = "distribution"
    PREDICT = "predict"
    TRAIN = "train"


class BoardConstants:
    DEFAULT_USER = 'admin'
    DEFAULT_PASS = 'admin'
    TASK_PROGRESS_MODULE = 'module'
    TASK_PROGRESS_LOSS = 'loss'
    TASK_PROGRESS_METRICS = 'metrics'
    TASK_PROGRESS_ERROR_MSG = 'error'


class TEEConstants:
    PARTY_NAME_TEE = 'party_name_tee'
    TOPIC_TEE = 'topic_tee'
    BUCKET_NUM = 10
    BUCKET_SIZE = 20000


class ParameterValue:
    ALL_COLUMNS = "all_columns"


class Math:
    U32_MAX = 0xFFFFFFFF
    U64_MAX = 0xFFFFFFFFFFFFFFFF
