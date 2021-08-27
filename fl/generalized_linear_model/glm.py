from datetime import datetime
from typing import Iterable
import functools
import json

import numpy as np

from common.factory.encryptor_factory import EncryptorFactory
from common.frame.data_frame.sample import Sample
from common.frame.data_frame.schema import Schema
from common.frame.message_frame.glm_message import GLMMessage
from common.frame.parameter_frame.glm_parameter import GLMParameter
from fl.algorithm import Algorithm
from fl.operator.regularizer import Regularizer
from privacy.math.linear_algebra import LinearAlgebra
from common.util import constant
from coordinator.core.sqlite_manager import SqliteManager

import hashlib


class GLM(Algorithm):
    def __init__(self, parameter: GLMParameter, message=GLMMessage()):
        super(GLM, self).__init__(parameter, message=message)

        self._coef = None
        self._intercept = None

        self._intercept_scaling = self._parameter.intercept_scaling

        # init regularizer
        self._regularizer = None
        if self._parameter.penalty is not None:
            self._regularizer = Regularizer(penalty=self._parameter.penalty,
                                            strength=1 / self._parameter.C)

        # init placeholders
        self._link_function = None

    def train(self, input_data=None, input_model=None):
        """
        :param input_data:
        :param input_model:
        :return: output_data, output_model
        """
        # train
        self._logger.info("start training")
        result = self._allocate_task(
            guest_task=functools.partial(self._train_guest, input_data=input_data, input_model=input_model),
            host_task=functools.partial(self._train_host, input_data=input_data, input_model=input_model),
        )
        self._logger.info("end training")

        return result

    def _save_task_loss(self, loss, last_update_time):
        current = datetime.now()
        SqliteManager.task_progress_dbo.create(args=dict(
            task_id=self._task_chain_id,
            progress_type=constant.BoardConstants.TASK_PROGRESS_LOSS,
            progress_value=json.dumps(dict(
                loss=loss,
                time=(current - last_update_time).seconds
            ))
        ))
        return current

    def predict(self, input_data=None):
        """
        :param input_data: features
        :return:
        """
        self._logger.info("start predicting")

        result = self._allocate_task(guest_task=functools.partial(self._predict_guest, input_data=input_data),
                                     host_task=functools.partial(self._predict_host, input_data=input_data),)
        self._logger.info("end predicting")

        return result

    def _initialize_model(self, input_data):
        # init coefficient and intercept
        feature_dimension = input_data.feature_dimension
        self._logger.info("guest has {} features".format(feature_dimension))

        self._coef = np.zeros(feature_dimension)  # np.array([]) if no features
        self._logger.info("coef initialized")

        if self._parameter.fit_intercept:
            self._intercept = 0.0
            self._logger.info("intercept initialized")
        return feature_dimension

    def _train_guest(self, input_data, input_model):
        # init a homomorphic cryptosystem
        homomorphic_encryptor = EncryptorFactory.get_instance(task_type=self._parameter.task_type,
                                                              encrypter=self._parameter.homomorphism,
                                                              key_size=self._parameter.key_size)

        feature_dimension = self._initialize_model(input_data)

        # data preprocessing
        sample_data = self._parse_label(input_data)

        # init class weight
        class_weight = None
        if getattr(self._parameter, 'balanced_class_weight', False):
            label_vector = sample_data.map(lambda row: row[1].label)
            class_weight = label_vector.countByValue()  # {0: 50, 1: 100, 2: 150, ...}
            sample_num = sample_data.count()
            for label in class_weight.keys():
                class_weight[label] = sample_num / (len(class_weight) * class_weight[label])

        # start iteration
        batch_fraction = self._parameter.batch_fraction if self._parameter.batch_type == "mini-batch" else 1.0
        label_size = int(1 / batch_fraction)
        epsilon = self._parameter.epsilon
        gamma = self._parameter.gamma
        grad_square_sum = np.zeros(feature_dimension + 1, dtype=np.float)
        para_update_square_sum = np.zeros(feature_dimension + 1, dtype=np.float)
        sample_data_with_index = sample_data.sortByKey(keyfunc=lambda x: hashlib.md5(str(x).encode()).hexdigest()). \
            zipWithIndex().persist()

        old_time4db = datetime.now()
        for iter_idx in range(self._parameter.max_iter):
            # init & get train_auc and validate_auc
            train_auc = 0
            validate_auc = 0
            if self._parameter.train_validate_freq is not None and iter_idx % self._parameter.train_validate_freq == 0:
                train_auc = self.validate(input_data)[1][0]["auc"]
                self._logger.info("start validation, evaluate train data auc:{}".format(train_auc))
                validate_result = self.validate()[1]
                if validate_result:
                    validate_auc = validate_result[0]["auc"]
                    self._logger.info("start validation, evaluate validate data:{}".format(validate_auc))

            loss = 0.0
            for batch_idx in range(label_size):
                self._logger.info("batch_iter: {}".format(batch_idx))
                sample_data = sample_data_with_index.filter(lambda x: x[1] % label_size == batch_idx). \
                    map(lambda x: x[0])
                self._logger.info("batch size: {}".format(sample_data.count()))
                # compute linear predictor
                guest_linear_predictor = self._get_linear_predictor_guest(sample_data)
                self._logger.info("got guest linear predictor")

                # receive host linear predictor
                host_linear_predictors = self._messenger.receive(
                    tag=self._message.LINEAR_PREDICTOR, suffix=[iter_idx], parties=self.get_all_host_names())
                self._logger.info("received host linear predictors: {}".format(
                    len(host_linear_predictors)))

                # aggregate linear predictor
                if not isinstance(guest_linear_predictor.take(1)[0][1], float):
                    linear_predictor = None
                    for host_linear_predictor in host_linear_predictors:
                        if linear_predictor == None:
                            linear_predictor = host_linear_predictor
                        else:
                            linear_predictor = linear_predictor.join_mapValues(host_linear_predictor,
                                                                               lambda val: val[0] + val[1])
                else:
                    linear_predictor = guest_linear_predictor
                    for host_linear_predictor in host_linear_predictors:
                        linear_predictor = guest_linear_predictor.join_mapValues(host_linear_predictor,
                                                                                 lambda val: val[0] + val[1])

                # get and receive regularization likelihood
                if self._regularizer is not None:
                    guest_reg = self._regularizer.get_likelihood(self._coef, self._intercept)
                    host_regs = self._messenger.receive(tag=self._message.REG_LIKELIHOOD,
                                                        suffix=[iter_idx],
                                                        parties=self.get_all_host_names())
                    self._logger.info("received host regs: {}".format(host_regs))
                    reg = guest_reg + sum(host_regs)
                else:
                    reg = 0.0

                # get residue = y - y_hat
                residue = self._get_residue_guest(sample_data, linear_predictor, class_weight)

                # encrypt residue and send
                enc_residue = residue.mapValues(lambda val: homomorphic_encryptor.encrypt(val))
                self._messenger.send(enc_residue,
                                     tag=self._message.ENC_RESIDUE,
                                     suffix=[iter_idx],
                                     parties=self.get_all_host_names())
                self._logger.info("sent encrypted residue: {}".format(enc_residue.count()))

                # get loss
                loss = self._get_loss_guest(sample_data, linear_predictor) - reg
                self._logger.info("loss = {}".format(loss))

                # get guest gradient and apply
                gradient_guest = self._get_loss_gradient_guest(sample_data, residue)

                if self._parameter.solver == constant.Solver.ADA_GRAM:
                    grad_square_sum += gradient_guest ** 2
                    self._apply_gradient_ada_gram_guest(gradient_guest, grad_square_sum, epsilon)
                elif self._parameter.solver == constant.Solver.ADA_DELTA:
                    grad_square_sum = grad_square_sum * gamma + (1 - gamma) * (gradient_guest ** 2)
                    cur_update_val = self._apply_gradient_ada_delta_guest(
                        gradient_guest, grad_square_sum, para_update_square_sum, epsilon
                    )
                    para_update_square_sum = para_update_square_sum * gamma + (1 - gamma) * (cur_update_val ** 2)
                elif self._parameter.solver == constant.Solver.GRADIENT_DESCENT:
                    self._apply_gradient_guest(gradient_guest)
                elif self._parameter.solver == constant.Solver.RMS_PROP:
                    grad_square_sum = grad_square_sum * gamma + (1 - gamma) * (gradient_guest ** 2)
                    self._apply_gradient_rmsprop_guest(gradient_guest, grad_square_sum, epsilon)

                self._logger.info("model updated")

                # receive host gradient
                masked_enc_gradient_hosts = self._messenger.receive(
                    tag=self._message.MASKED_ENC_GRADIENT_HOST,
                    suffix=[iter_idx],
                    parties=self.get_all_host_names())
                self._logger.info("received masked host gradient ciphertexts: {}".format(
                    len(masked_enc_gradient_hosts)))

                # decrypt and send
                for masked_enc_gradient_host, other_party in zip(masked_enc_gradient_hosts,
                                                                 self.get_other_host_names()):
                    masked_gradient_host = [0 for _ in range(len(masked_enc_gradient_host))]
                    for idx, enc_component in enumerate(masked_enc_gradient_host):
                        masked_gradient_host[idx] = homomorphic_encryptor.decrypt(enc_component)
                    self._messenger.send(np.asarray(masked_gradient_host),
                                         tag=self._message.MASKED_GRADIENT_HOST,
                                         suffix=[iter_idx],
                                         parties=[other_party])
                    self._logger.info("sent masked host gradient back: {} to {}".format(
                        len(masked_gradient_host), other_party))

                # check early stop
                local_early_stop = self._early_stop_guest()
                if self._check_early_stop(iter_idx, local_early_stop):
                    self._logger.info("early stopped")
                    break

            # save task progress to db
            new_time4db = datetime.now()
            SqliteManager.task_progress_dbo.create(args=dict(
                task_id=self._task_chain_id,
                progress_type=constant.BoardConstants.TASK_PROGRESS_LOSS,
                progress_value=json.dumps(dict(
                    loss=loss,
                    time=(new_time4db - old_time4db).seconds,
                    train_auc=train_auc,
                    validate_auc=validate_auc
                ))
            ))
            old_time4db = new_time4db

        sample_data_with_index.unpersist()
        # construct output model
        output_model = self.instance_to_model()

        return input_data, output_model

    def _train_host(self, input_data, input_model):
        # init coefficient
        feature_dimension = self._initialize_model(input_data)

        # start iteration
        batch_fraction = self._parameter.batch_fraction if self._parameter.batch_type == "mini-batch" else 1.0
        label_size = int(1 / batch_fraction)
        epsilon = self._parameter.epsilon
        gamma = self._parameter.gamma
        grad_square_sum = np.zeros(feature_dimension, dtype=np.float)
        para_update_square_sum = np.zeros(feature_dimension, dtype=np.float)
        input_data_with_index = input_data.sortByKey(keyfunc=lambda x: hashlib.md5(str(x).encode()).hexdigest()). \
            zipWithIndex().persist()

        old_time4db = datetime.now()
        for iter_idx in range(self._parameter.max_iter):
            self._logger.info("iter: {}".format(iter_idx))
            if self._parameter.train_validate_freq is not None and iter_idx % self._parameter.train_validate_freq == 0:
                self.validate(input_data)
                self.validate()

            for batch_idx in range(label_size):
                self._logger.info("batch_iter: {}".format(batch_idx))
                sample_data = input_data_with_index.filter(lambda x: x[1] % label_size == batch_idx). \
                    map(lambda x: x[0])
                self._logger.info("batch size: {}".format(sample_data.count()))
                # compute linear predictor
                host_linear_predictor = self._get_linear_predictor_host(sample_data)
                self._logger.info("got host linear predictor")

                # send host linear predictor
                self._messenger.send(host_linear_predictor,
                                     tag=self._message.LINEAR_PREDICTOR,
                                     suffix=[iter_idx],
                                     parties=self.get_all_guest_names())
                self._logger.info("sent host linear predictor: {}".format(host_linear_predictor.count()))

                # get and send regularization likelihood
                if self._regularizer is not None:
                    host_reg = self._regularizer.get_likelihood(self._coef)
                    self._messenger.send(host_reg,
                                         tag=self._message.REG_LIKELIHOOD,
                                         suffix=[iter_idx],
                                         parties=self.get_all_guest_names())
                    self._logger.info("sent host reg: {}".format(host_reg))

                enc_residue = self._messenger.receive(tag=self._message.ENC_RESIDUE,
                                                      suffix=[iter_idx],
                                                      parties=self.get_all_guest_names())[0]
                self._logger.info("received encrypted residue")

                # get and send gradient in cipher space
                enc_gradient_host = self._get_loss_gradient_host(sample_data, enc_residue)
                masked_enc_gradient_host, mask_rule = self._mask_gradient_host(enc_gradient_host)
                self._messenger.send(masked_enc_gradient_host,
                                     tag=self._message.MASKED_ENC_GRADIENT_HOST,
                                     suffix=[iter_idx],
                                     parties=self.get_all_guest_names())
                self._logger.info("sent masked host gradient ciphertexts: {}".format(
                    len(masked_enc_gradient_host)))

                # receive decrypted host gradient and unmask
                masked_gradient_host = self._messenger.receive(tag=self._message.MASKED_GRADIENT_HOST,
                                                               suffix=[iter_idx],
                                                               parties=self.get_all_guest_names())[0]
                self._logger.info("received masked host gradient: {}".format(len(masked_gradient_host)))
                gradient_host = self._unmask_gradient_host(masked_gradient_host, mask_rule)

                # apply regularization term if applicable
                if self._regularizer is not None:
                    gradient_host += self._regularizer.get_loss_gradient(self._coef)

                # apply gradient
                if self._parameter.solver == constant.Solver.ADA_GRAM:
                    grad_square_sum += gradient_host ** 2
                    self._apply_gradient_ada_gram_guest(gradient_host, grad_square_sum, epsilon)
                elif self._parameter.solver == constant.Solver.ADA_DELTA:
                    grad_square_sum = grad_square_sum * gamma + (1 - gamma) * (gradient_host ** 2)
                    cur_update_val = self._apply_gradient_ada_delta_guest(
                        gradient_host, grad_square_sum, para_update_square_sum, epsilon
                    )
                    para_update_square_sum = para_update_square_sum * gamma + (1 - gamma) * (cur_update_val ** 2)
                elif self._parameter.solver == constant.Solver.GRADIENT_DESCENT:
                    self._apply_gradient_host(gradient_host)
                elif self._parameter.solver == constant.Solver.RMS_PROP:
                    grad_square_sum = grad_square_sum * gamma + (1 - gamma) * (gradient_host ** 2)
                    self._apply_gradient_rmsprop_guest(gradient_host, grad_square_sum, epsilon)

                self._logger.info("model updated")

                # check early stop
                local_early_stop = self._early_stop_host()
                if self._check_early_stop(iter_idx, local_early_stop):
                    self._logger.info("early stopped")
                    break

            # save task progress to db
            new_time4db = datetime.now()
            SqliteManager.task_progress_dbo.create(args=dict(
                task_id=self._task_chain_id,
                progress_type=constant.BoardConstants.TASK_PROGRESS_LOSS,
                progress_value=json.dumps(dict(
                    loss=0,
                    time=(new_time4db - old_time4db).seconds
                ))
            ))
            old_time4db = new_time4db

        input_data_with_index.unpersist()
        # construct output model
        output_model = self.instance_to_model()

        return input_data, output_model

    def _predict_guest(self, input_data):
        linear_predictor_guest = self._get_linear_predictor_guest(input_data)
        linear_predictor_hosts = self._messenger.receive(tag=self._message.PRED_HOST,
                                                         parties=self.get_all_host_names())
        self._logger.info("received linear predictors in prediction: {}".format(
            len(linear_predictor_hosts)))

        if not isinstance(linear_predictor_guest.take(1)[0][1], float):
            linear_predictor = None
            for host_linear_predictor in linear_predictor_hosts:
                if linear_predictor:
                    linear_predictor = linear_predictor.join_mapValues(host_linear_predictor,
                                                                       lambda val: val[0] + val[1])
                else:
                    linear_predictor = host_linear_predictor
        else:
            linear_predictor = linear_predictor_guest
            for host_linear_predictor in linear_predictor_hosts:
                linear_predictor = linear_predictor_guest.join_mapValues(host_linear_predictor,
                                                                         lambda val: val[0] + val[1])

        y_hat = linear_predictor.mapValues(functools.partial(self._link_function.invert))

        # construct DDataset as output
        y_hat = y_hat.mapValues(lambda val: Sample(label=val))
        y_hat.schema = Schema(id_name=input_data.schema.id_name, label_name='y_pred')

        return y_hat

    def _predict_host(self, input_data):
        linear_predictor_host = self._get_linear_predictor_host(input_data)

        self._messenger.send(linear_predictor_host,
                             tag=self._message.PRED_HOST,
                             parties=self.get_all_guest_names())
        self._logger.info("sent linear predictor in prediction: {}".format(linear_predictor_host.count()))

        return input_data

    @staticmethod
    def _parse_label(input_data):
        """
        Parse label to binary or multi-class
        Must be explicitly inherited
        :param input_data:
        :return:
        """
        return input_data

    def _get_linear_predictor_guest(self, input_data):
        def compute_linear_predictor_for_each_row(sample, intercept_scaling, coef, intercept):
            features = sample.features
            linear_prediction = LinearAlgebra.pdot(features, coef)
            if intercept is not None:
                linear_prediction += intercept_scaling * intercept
            return linear_prediction

        f = functools.partial(compute_linear_predictor_for_each_row,
                              intercept_scaling=self._intercept_scaling,
                              coef=self._coef,
                              intercept=self._intercept)

        linear_predictor = input_data.mapValues(f)

        return linear_predictor

    def _get_linear_predictor_host(self, input_data):
        def compute_linear_predictor_for_each_row(sample, coef):
            features = sample.features
            linear_prediction = LinearAlgebra.pdot(features, coef)
            return linear_prediction

        f = functools.partial(compute_linear_predictor_for_each_row, coef=self._coef)

        linear_predictor = input_data.mapValues(f)

        return linear_predictor

    def _get_residue_guest(self, input_data, linear_predictor, class_weight):
        """
        residue = y - y^hat
        :param input_data
        :param linear_predictor
        :param class_weight: dict or None
        :return:
        """

        def get_residue_for_each_row(val, link_function, class_weight):
            """
            :param val: (y, lp)
            :param link_function:
            :param class_weight
            :return: residue
            """
            y, lp = val
            y_hat = link_function.invert(lp)
            res = y - y_hat
            # add class weight
            if class_weight is not None:
                res *= class_weight[y]
            return res

        y_lp = input_data.join_mapValues(linear_predictor, lambda val: (val[0].label, val[1]))  # (id, (y, lp))
        residue = y_lp.mapValues(functools.partial(get_residue_for_each_row,
                                                   link_function=self._link_function,
                                                   class_weight=class_weight))
        return residue

    def _get_likelihood_guest(self, input_data, linear_predictor):
        def get_likelihood_for_each_partition(partition, link_function):
            """
            :param partition: (y, lp)
            :param link_function:
            :return:
            """
            likelihood = 0.0
            for _, val in partition:
                y, lp = val
                local_likelihood = y * lp - link_function.int_invert(lp)
                likelihood += local_likelihood
            yield likelihood

        y_lp = input_data.join_mapValues(linear_predictor, lambda val: (val[0].label, val[1]))  # (id, (y, lp))

        likelihood = y_lp.mapPartitions(
            functools.partial(get_likelihood_for_each_partition, link_function=self._link_function)
        ).reduce(lambda v, u: v + u) / input_data.count()
        return likelihood

    def _get_loss_guest(self, input_data, linear_predictor):
        return -self._get_likelihood_guest(input_data, linear_predictor)

    def _get_likelihood_gradient_guest(self, input_data, residue):
        def get_gradient_for_each_partition(partition):
            """

            :param partition: (feature, residue)
            :return:
            """
            grad = 0.0
            for _, val in partition:
                features, res = val
                local_gradient = res * features
                grad += local_gradient
            yield grad

        feature_residue = input_data.join_mapValues(
            residue, lambda val: (val[0].features, val[1]))  # (id, (feat, residue))

        gradient = feature_residue.mapPartitions(
            get_gradient_for_each_partition).reduce(lambda v, u: v + u) / input_data.count()

        # compute intercept gradient
        if self._parameter.fit_intercept:
            intercept_gradient = residue.sum_values() / residue.count() * self._intercept_scaling
            # construct gradient
            gradient = np.append(gradient, intercept_gradient)

        # apply regularization term if applicable
        if self._regularizer is not None:
            gradient += self._regularizer.get_likelihood_gradient(self._coef, self._intercept)

        return gradient

    @staticmethod
    def _get_likelihood_gradient_host(input_data, residue):
        def get_gradient_for_each_partition(partition):
            """
            :param partition: (feature, residue)
            :return:
            """
            grad = 0.0
            for _, val in partition:
                features, res = val
                local_gradient = LinearAlgebra.cmult(res, features)
                if grad == 0.0:
                    grad = local_gradient
                else:
                    grad = LinearAlgebra.cadd(grad, local_gradient)
            yield grad

        sample_num = input_data.count()
        feature_residue = input_data.join_mapValues(
            residue, lambda val: (val[0].features / sample_num, val[1]))  # (id, (feature / n, residue))

        def collect_gradient(v, u):
            if v and u:
                return LinearAlgebra.cadd(v, u)
            elif v:
                return v
            else:
                return u
        gradient = feature_residue.mapPartitions(
            get_gradient_for_each_partition).reduce(collect_gradient)
        return gradient

    def _get_loss_gradient_guest(self, input_data, residue):
        return -self._get_likelihood_gradient_guest(input_data, residue)

    def _get_loss_gradient_host(self, input_data, residue):
        return LinearAlgebra.cneg(self._get_likelihood_gradient_host(input_data, residue))

    def _apply_gradient_guest(self, gradient):
        if self._parameter.fit_intercept:
            self._coef -= self._parameter.learning_rate * gradient[:-1]
            self._intercept -= self._parameter.learning_rate * gradient[-1]
        else:
            self._coef -= self._parameter.learning_rate * gradient

    def _apply_gradient_ada_gram_guest(self, gradient, grad_square_sum, epsilon):
        if self._parameter.fit_intercept:
            self._coef -= self._parameter.learning_rate * gradient[:-1] / (grad_square_sum[:-1] + epsilon) ** (1 / 2)
            self._intercept -= self._parameter.learning_rate * gradient[-1] / (grad_square_sum[-1] + epsilon) ** (1 / 2)
        else:
            self._coef -= self._parameter.learning_rate * gradient / (grad_square_sum + epsilon) ** (1 / 2)

    def _apply_gradient_ada_delta_guest(self, gradient, grad_square_sum, val_square_sum, epsilon):
        val = (val_square_sum + epsilon) ** (1 / 2) / ((grad_square_sum + epsilon) ** (1 / 2)) * gradient
        if self._parameter.fit_intercept:
            self._coef -= val[:-1]
            self._intercept -= val[-1]
        else:
            self._coef -= val
        return val

    def _apply_gradient_rmsprop_guest(self, gradient, grad_square_sum, epsilon):
        val = self._parameter.learning_rate / ((grad_square_sum + epsilon) ** (1 / 2)) * gradient
        if self._parameter.fit_intercept:
            self._coef -= val[:-1]
            self._intercept -= val[-1]
        else:
            self._coef -= val

    def _apply_gradient_host(self, gradient):
        self._coef -= self._parameter.learning_rate * gradient

    def _apply_gradient_ada_gram_host(self, gradient, grad_square_sum, epsilon):
        self._coef -= self._parameter.learning_rate * gradient / (grad_square_sum + epsilon) ** (1 / 2)

    def _apply_gradient_ada_delta_host(self, gradient, grad_square_sum, val_square_sum, epsilon):
        val = (val_square_sum + epsilon) ** (1 / 2) / ((grad_square_sum + epsilon) ** (1 / 2)) * gradient
        self._coef -= val
        return -val

    def _apply_gradient_rmsprop_host(self, gradient, grad_square_sum, epsilon):
        val = self._parameter.learning_rate / ((grad_square_sum + epsilon) ** (1 / 2)) * gradient
        self._coef -= val

    @staticmethod
    def _mask_gradient_host(gradient):
        """
        Use random permutation to mask gradient
        :param gradient:
        :return:
        """
        # generate permutation rule
        mask_rule = np.random.permutation(len(gradient))
        # construct masked gradient
        masked_gradient = [0 for _ in range(len(gradient))]
        for i in range(len(mask_rule)):
            masked_gradient[i] = gradient[mask_rule[i]]
        return masked_gradient, mask_rule

    @staticmethod
    def _unmask_gradient_host(gradient, mask_rule):
        real_gradient = np.zeros_like(gradient)
        # reconstruct the unmasked gradient
        for i in range(len(mask_rule)):
            real_gradient[mask_rule[i]] = gradient[i]
        return real_gradient

    def _early_stop_guest(self):
        if np.max(np.abs(np.append(self._coef, self._intercept))) < self._parameter.tol:
            return True
        else:
            return False

    def _early_stop_host(self):
        if np.max(np.abs(self._coef)) < self._parameter.tol:
            return True
        else:
            return False

    def _check_early_stop(self, iter_idx, local_early_stop):
        self._messenger.send(local_early_stop, tag=self._message.EARLY_STOP, suffix=[iter_idx])
        self._logger.info("sent local early stop: {}".format(local_early_stop))
        other_early_stop = self._messenger.receive(tag=self._message.EARLY_STOP, suffix=[iter_idx])
        self._logger.info("received other early stop: {}".format(other_early_stop))
        return local_early_stop and any(other_early_stop)

    def set_coef(self, coef):
        self._coef = coef

    def set_intercept_scaling(self, intercept_scaling):
        self._intercept_scaling = intercept_scaling

    def set_intercept(self, intercept):
        self._intercept = intercept

    def _check_feature_dimensions(self, input_data=None):
        """
        Check clients' feature dimensions
        :param input_data:
        :return: int
        """
        if self.get_this_party_role() == constant.TaskRole.SERVER:
            cli_feat_dims = self._messenger.receive(
                tag=self._message.CLIENT_FEATURE_DIMENSION,
                parties=self.get_all_client_names()
            )

            min_cli_feat_dim = min(cli_feat_dims)

            if min_cli_feat_dim == 0 or min_cli_feat_dim != max(cli_feat_dims):
                raise RuntimeError("clients have invalid feature dimension inputs: {}".format(
                    list(zip(self.get_all_client_names(), cli_feat_dims))))

            return min_cli_feat_dim

        elif self.get_this_party_role() == constant.TaskRole.CLIENT:
            assert input_data is not None

            cli_feat_dim = input_data.feature_dimension
            self._messenger.send(cli_feat_dim,
                                 tag=self._message.CLIENT_FEATURE_DIMENSION,
                                 parties=self.get_all_server_names())

            return cli_feat_dim

        else:
            raise ValueError("invalid task role: {}".format(self._parameter.task_role))

    def _sync_local_sample_num(self, input_data=None):
        """
        Clients report to the server the number of local clients
        :param input_data:
        :return:
        """
        if self.get_this_party_role() == constant.TaskRole.SERVER:
            client_num_local_samples = self._messenger.receive(
                tag=self._message.CLIENT_NUM_LOCAL_SAMPLE,
                parties=self.get_all_client_names()
            )
            self._logger.info("received client_num_local_samples: {}".format(client_num_local_samples))

            return client_num_local_samples

        elif self.get_this_party_role() == constant.TaskRole.CLIENT:
            client_num_local_sample = input_data.count()
            self._messenger.send(client_num_local_sample,
                                 tag=self._message.CLIENT_NUM_LOCAL_SAMPLE,
                                 parties=self.get_all_server_names())
            self._logger.info("sent client_num_local_sample: {}".format(client_num_local_sample))

        else:
            raise ValueError("invalid task role: {}".format(self._parameter.task_role))

    def _parse_feature(self, input_data):
        """
        Append a column of constant features for activating bias
        :return:
        """

        def append_column_for_each_value(sample):
            """

            :param sample:
            :return:
            """
            sample.features = np.append(sample.features, 1.0)
            return sample

        if self._parameter.fit_intercept:
            input_data = input_data.mapValues(append_column_for_each_value, preserve_schema=True)
            input_data.feature_dimension += 1
            input_data.schema.feature_names.append('bias')

        return input_data

    def _get_trainer(self, weight):
        """
        Get local trainer in FedAve.
        Must be explicitly inherited.
        :return: func
        """
        pass

    @staticmethod
    def _aggregate_weights(nxt_weights, random_client_weights):
        """
        :param nxt_weights: List[numpy.ndarray]
        :param random_client_weights: List[float]
        :return: numpy.ndarray
        """
        return sum(x*y for x, y in zip(nxt_weights, random_client_weights))

    @staticmethod
    def _aggregate_loss(loss: Iterable, random_client_weights: Iterable[float]) -> float:
        return sum(x*y for x, y in zip(loss, random_client_weights))
