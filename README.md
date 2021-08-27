# Competition-Oriented Jeddak Platform

# Overview
Jeddak provides a both academia- and industry-oriented platform for privacy computing and federated learning.

This is a competition-oriented lite version of Jeddak. Three guides for deploy, develop and use, respectively are provided below.


# Deploy Guide
Jeddak provides two deployment modes: standalone and cluster,
where standalone mode is for fast experimental verifications
of new algorithms over a single host, and cluster mode supports production in
real multi-host applications. Note that the competition is conducted over the cluster mode.

Refer to `doc/guide/quickstart.md` for deployment guide.


# Develop Guide
Jeddak provides standardized interfaces for developing your own
federated learning and privacy-preserving algorithms.

Refer to `doc/guide/develop_guide.md` for more details.


# Use Guide

## Algorithm List
Jeddak provides a series of developed privacy-preserving algorithms
as described in the following table. For this lite version, a limited number of such algorithms are mainly for the purpose of demonstration. Their configurations can be found
at `example/conf/`.

|  Algorithm Name   | Classification | Description  |
|  ----  | ----  | ---- |
| data_loader  | Preprocessing |  Read data from various data sources |
| data_saver  | Postprocessing | Save data to disk in various data structures |
| aligner | Preprocessing | Seek the intersection of the private sets held by multiple parties in a privacy-preserving fashion |
| glm | Federated Learning | A set of generalized linear models, including linear regression, logistic regression and poisson regression |
| dpgbdt | Federated Learning | Differentially Private Gradient Boosting Decision Tree |
| neural_network | Federated Learning | Deep Neural Network |
| evaluate | Postprocessing | Evaluate a federated learning model |
| model_loader| Postprocessing | Load model from local file / unload model from memory |
| predict_offline | Postprocessing | Offline prediction through specified model |

## Parameter List


### data_loader parameters

|  Parameter   | Type | Range  | Default | Description |
|  ----  | ----  | ---- | ---- | ---- |
| task_type  | str | "data_loader" | "data_loader" | task type |
| task_role  | str | {"guest", "host", "sole", "slack"} | "guest" | task role. "guest/host" means party's role in a task. "sole" means only this party carries out the task. "slack" means the party does nothing in the task |
| input_data_source | str | {"csv", "hdfs"} | "csv" | type of input data source. "csv" means local files and "hdfs" means a file path of Hadoop HDFS. |
| input_data_path | str | any strings | N/A | file path of input data which is valid and readable |
| train_data_path | str | any strings | N/A | file path of train data which is valid and readable, if not, will get from input_data_path. |
| validate_data_path | str | any strings | N/A | file path of validate data which is valid and readable |
| convert_sparse_to_index | bool | {true, false} | true | convert sparse features to natural numbers if true |

### data_saver parameters
|  Parameter   | Type | Range  | Default | Description |
|  ----  | ----  | ---- | ---- | ---- |
| task_type  | str | "data_saver" | "data_saver" | task type |
| task_role  | str | {"guest", "host", "sole", "slack"} | "guest" | task role |
| output_data_source | str | {"csv"} | "csv" | type of output data source |


### aligner parameters
|  Parameter   | Type | Range  | Default | Description |
|  ----  | ----  | ---- | ---- | ---- |
| task_type  | str | "aligner" | "aligner" | task type |
| task_role  | str | {"guest", "host"} | "guest" | task role |
| align_mode | str | {"diffie_hellman", "cm20", "dh_PSI", "tee"} | "cm20" | psi type |
| output_id_only | bool | {true, false} | true | output only id of each element in the intersection set |
| sync_intersection | bool | {true, false} | true | synchronizing the intersection set among all parties |
| key_size | int | {1024, 2048, 3072, 4096} | 1024 | cryptographic key length (in bits) |
| batch_num | int | {"auto"}, [1, inf) | "auto" | batch number for PSI in "cm20" mode, integer will be rounded up to power of 2 |


### glm parameters
|  Parameter   | Type | Range  | Default | Description |
|  ----  | ----  | ---- | ---- | ---- |
| task_type  | str | {"linear_regression", "logistic_regression", "poisson_regression"} | N/A | task type |
| task_role  | str | {"guest", "host", "server", "client"} | "guest" | task role |
| penalty | str | {"l1", "l2", null} | "l2" | penalty term |
| tol | float | [0, inf) | 1e-4 | tolerance for stopping criteria |
| C | float | (0, inf) | 1.0 | inverse of regularization strength |
| fit_intercept | bool | {true, false} | true | bias |
| intercept_scaling | float | [0, inf) | 1.0 | x becomes [x, self.intercept_scaling] if fit_intercept is true |
| solver | str | {"gradient_descent", "AdaGram", "AdaDelta", "RMSprop"} | "gradient_descent" | optimization method |
| max_iter | int | [1, inf) | 100 | maximum iteration rounds |
| learning_rate | float | (0, inf) | 0.15 | learning step size |
| homomorphism | str | {"cpaillier"} | "cpaillier" | homomorphic encryption method |
| key_size | int | [1, inf) | 1024 | homomorphic encryption key size |
| gamma | float | (0, inf) | 0.9 | adjust the sum of past squared gradients |
| epsilon | float | (0, inf) | 1e-8|  smooth gradient and avoid division by zero |
| batch_fraction | float | (0, 1] | 0.1 | the subset fraction of mini-batch training |
| batch_type | str | {"batch", "mini-batch"} | "batch" | batch method |
| balanced_class_weight | bool | {true, false} | true | automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)), auto-disabled for continuous labels |
| train_validate_freq | int | [1, inf) | None | validation using validate data each train_validate_freq epoch if train_validate_freq is not None


### dpgbdt parameters
|  Parameter   | Type | Range  | Default | Description |
|  ----  | ----  | ---- | ---- | ---- |
| task_type  | str | "dpgbdt" | "dpgbdt" | task type |
| task_role  | str | {"guest", "host"} | "guest" | task role |
| objective  | str | {"reg_squarederror", "binary_logistic", "count_poisson"} | "binary_logistic" | learing objective |
| num_round  | int | [1, inf) | 20 | the number of boosting rounds |
| eta  | float | (0, inf) | 0.3 | learning rate |
| gamma  | float | [0, inf) | 0.0 | minimum loss reduction required to make a further partition on a leaf node of the tree |
| max_depth  | int | [1, inf) | 3 | maximum depth of a tree |
| min_child_weight  | float | [0, inf) | 1.0 | minimum sum of instance weight (hessian) needed in a child |
| max_delta_step  | float | [0, inf) | 0.0 | maximum delta step we allow each leaf output to be |
| sub_sample  | float | (0, 1] | 1.0 | subsample ratio of the training instances at each boosting iteration |
| lam  | float | [0, inf) | 1.0 | L2 regularization strength |
| sketch_eps  | float | (0, 1) | 0.03 | convert every column into 1 / sketch_eps number of bins at most |
| homomorphism | str | {"cpaillier"} | "cpaillier" | homomorphic encryption method |
| key_size | int | [1, inf) | 1024 | homomorphic encryption key size |
| importance_type  | str | {"weight", "gain", "cover", "total_gain", "total_cover", "all"} | "weight" | feature importance type |
| train_validate_freq | int | [1, inf) | None | validation using validate data each train_validate_freq tree if train_validate_freq is not None


### neural_network parameters
|  Parameter   | Type | Range  | Default | Description |
|  ----  | ----  | ---- | ---- | ---- |
| task_type  | str | "neural_network" | "neural_network" | task type |
| task_role  | str | {"guest", "host"} | "guest" | task role |
| backend | str | {"keras", "pytorch"} | None | backend framework of deep learning |
| format | str | {"file", "conf"} | None | input format of top/mid/bottom model to be loaded |
| btm | str | Any | None | keras model config json string or model file path |
| mid | str| Any | None | keras model config json string or model file path |
| top | str| Any | None | keras model config json string or model file path |
| epochs  | int | [1, inf) | 1 | epochs of training |
| batch_size  | int | [1, inf) | 1 | batch size of training |
| loss_fn  | str | {"CrossEntropyLoss", "MSELoss", ...} | None | loss function of top model |
| learning_rate | float | [0, inf) | 0.001 | learning rate of training |
| optimizer | str | {"SGD", "Adam", ...} | None | optimizer of top/bottom model |
| use_mid | bool | {true, false} | true | use mid model for vertical-nn or not (only top/bottom models) |
| mid_shape_in | int | [1, inf) | 1 | input shape of mid model, the same as output shape of host bottom model |
| mid_shape_out | int | [1, inf) | 1 | output shape of mid model, equals to input shape of guest top model minus output shape of guest btm model |
| mid_activation | str | {"linear", "Relu", ...} | "linear" | activation function of mid model |
| privacy_mode | str | {"plain"} | "plain" | encryption mechanism of interaction between multiple parties |
| metrics | str | {"accuracy", "", ...} | None | output metrics for model evaluation |
| predict_model | str | {"categorical", "", ...} | None | set to "categorical" only if top model is a classification model and the prediction value should be transformed to categorical vector |
| num_classes | int | [1, inf) | None | number of categories, only needed in the case of top model is a classification model and the option "predict_model" is "categorical" |
| client_frac | (0.0, 1.0] | float | 1.0 | (cluster-server mode) the fraction of clients selected to update the global model |
| model_conf | str | Any | None | keras model config json string or model file path |
| train_validate_freq | int | [1, inf) | None | validation using validate data each train_validate_freq epoch if train_validate_freq is not None


### evaluate parameters
|  Parameter   | Type | Range  | Default | Description |
|  ----  | ----  | ---- | ---- | ---- |
| task_type  | str | "evaluate" | "evaluate" | task type |
| task_role  | str | {"guest", "host"} | "guest" | task role |

### model_loader parameters
|  Parameter   | Type | Range  | Default | Description |
|  ----  | ----  | ---- | ---- | ---- |
| task_type  | str | "model_loader" | "model_loader" | task type |
| task_role  | str | {"guest", "host"} | "guest" | task role |
| model_id  | str | {model_id} | None | id of model to be loaded/unloaded |
| action  | str | {"load", "unload"} | None | load/unload model |


### predict_offline parameters
|  Parameter   | Type | Range  | Default | Description |
|  ----  | ----  | ---- | ---- | ---- |
| task_type  | str | "predict_offline" | "predict_offline" | task type |
| task_role  | str | {"guest", "host"} | "guest" | task role |
| model_id  | str | {model_id} | None | id of model used for prediction |
| input_data_path  | str | {file_path} | None | input file's path and filename |
