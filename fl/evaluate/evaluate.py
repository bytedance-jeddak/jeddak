import json

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

from common.frame.message_frame.evaluate_message import EvaluateMessage
from common.frame.model_frame.evaluate_model import EvaluateModel
from common.frame.parameter_frame.evaluate_parameter import EvaluateParameter
from common.util import constant
from coordinator.core.sqlite_manager import SqliteManager
from fl.algorithm import Algorithm
from fl.gbdt.dpgbdt import DPGBDT
from fl.generalized_linear_model.linear_regression import LinearRegression
from fl.generalized_linear_model.logistic_regression import LogisticRegression
from fl.generalized_linear_model.poisson_regression import PoissonRegression


class Evaluate(Algorithm):
    """
    Evaluate
    """

    def __init__(self, parameter: EvaluateParameter, message=EvaluateMessage()):
        super(Evaluate, self).__init__(parameter, message=message)

    def train(self, input_data=None, input_model=None):
        self._logger.info("start new evaluating")
        model_type = input_model.parameter.task_type
        if model_type == constant.TaskType.LINEAR_REGRESSION:
            algorithm_instance = LinearRegression.model_to_instance(input_model)
        elif model_type == constant.TaskType.LOGISTIC_REGRESSION:
            algorithm_instance = LogisticRegression.model_to_instance(input_model)
        elif model_type == constant.TaskType.POISSON_REGRESSION:
            algorithm_instance = PoissonRegression.model_to_instance(input_model)
        elif model_type == constant.TaskType.DPGBDT:
            algorithm_instance = DPGBDT.model_to_instance(input_model)
        else:
            raise Exception('Evaluate get_instance unkown task_type', model_type)

        pred_label = algorithm_instance.predict(input_data)
        pred_label = pred_label.mapValues(lambda val: val.label)
        true_label = input_data.mapValues(lambda val: val.label)
        my_party_role = self.get_this_party_role()
        if my_party_role == constant.TaskRole.GUEST:
            metrics = self._eval(pred_label, true_label, algorithm_instance.get_evaluate_type())
            output_model = self.instance_to_model()
            self._logger.info("metrics: {}".format(metrics))
            self.save_metrics(metrics)
            return input_data, output_model
        elif my_party_role == constant.TaskRole.HOST:
            output_model = self.instance_to_model()
            return input_data, output_model
        else:
            raise ValueError("invalid task role: {}".format(self.get_this_party_role()))

    def _eval(self, pred_label=None, true_label=None, evaluation_type=constant.EvaluationType.BINARY):
        """
        :param pred_label: DDataset ((id, pred_score))
        :param true_label: DDataset ((id, true_label))
        :return:
        """
        y_agg = []
        true_label_list = []
        pred_label_list = []
        sample = pred_label.first()
        if isinstance(sample[1], list):
            for i in range(len(sample[1])):
                true_label_i = true_label.mapValues(lambda row: row[i])
                pred_label_i = pred_label.mapValues(lambda row: row[i])
                true_label_list.append(true_label_i)
                pred_label_list.append(pred_label_i)
                y_agg_i = self._preprocess_label(pred_label_i, true_label_i, evaluation_type)
                y_agg.append(y_agg_i)
        else:
            true_label_list.append(true_label)
            pred_label_list.append(pred_label)
            y_agg_i = self._preprocess_label(pred_label, true_label, evaluation_type)
            y_agg.append(y_agg_i)
        metrics_dict_list = []

        if evaluation_type == constant.EvaluationType.BINARY:
            for i in range(len(y_agg)):
                metrics_dict = {}
                pred_true_agg = pred_label_list[i].join(true_label_list[i]).eliminate_key()
                pred_label = pred_true_agg.map(lambda row: row[0])
                true_label = pred_true_agg.map(lambda row: row[1])
                precision, recall, f1 = self._binary_precision_recall(true_label.collect(), pred_label.collect())
                metrics_dict[constant.Metric.PRECISION] = [('threshlod', 'precision')] + precision
                metrics_dict[constant.Metric.RECALL] = [('threshlod', 'recall')] + recall
                metrics_dict[constant.Metric.F1] = [('threshlod', 'f1', 'precision', 'recall')] + f1
                prediction_array, label_array = np.array(pred_label.rows), np.array(true_label.rows)
                metrics_dict[constant.Metric.AUC] = roc_auc_score(label_array, prediction_array)
                p, r, _ = precision_recall_curve(label_array, prediction_array)
                metrics_dict[constant.Metric.AUPR] = auc(r, p)
                metrics_dict_list.append(metrics_dict)
        else:
            raise ValueError("invalid evaluation type: {}".format(evaluation_type))

        return metrics_dict_list

    @staticmethod
    def _preprocess_label(pred_score, true_label, evaluation_type):
        y_agg = pred_score.join(true_label).eliminate_key()
        if evaluation_type in [constant.EvaluationType.BINARY, constant.EvaluationType.REGRESSION]:
            return y_agg.map_for_list(lambda row: (float(row[0]), float(row[1]))).rows
        elif evaluation_type == constant.EvaluationType.MULTICLASS:
            return y_agg.map_for_list(lambda row: (float(np.round(row[0])), float(np.round(row[1])))).rows
        else:
            raise ValueError("invalid evaluation type: {}".format(evaluation_type))

    @staticmethod
    def _binary_clf_curve(y_true, y_score):
        y_true = np.ravel(np.asarray(y_true))
        y_score = np.ravel(np.asarray(y_score))
        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        # accumulate the true positives with decreasing threshold
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        return fps, tps, y_score[threshold_idxs]

    def _binary_precision_recall(self, y_true, probas_pred):
        fps, tps, thresholds = self._binary_clf_curve(y_true, probas_pred)
        precision = tps / (tps + fps)
        precision[np.isnan(precision)] = 0
        recall = tps / tps[-1]
        last_ind = tps.searchsorted(tps[-1])
        f1 = 2 * precision * recall / (precision + recall)
        sl = slice(last_ind, None, -1)
        thresholds = np.r_[thresholds[sl], 1]

        precision = np.r_[precision[sl], 1]
        desc_score_pre = np.argsort(precision, kind="mergesort")[::-1]
        precision_curve = list(zip(thresholds[desc_score_pre], precision[desc_score_pre]))[:10]

        recall = np.r_[recall[sl], 0]
        desc_score_recall = np.argsort(recall, kind="mergesort")[::-1]
        recall_curve = list(zip(thresholds[desc_score_recall], recall[desc_score_recall]))[:10]

        f1 = np.r_[f1[sl], 0]
        desc_score_f1 = np.argsort(f1, kind="mergesort")[::-1]
        f1_curve = list(
            zip(thresholds[desc_score_f1], f1[desc_score_f1], precision[desc_score_f1], recall[desc_score_f1]))[:10]
        return precision_curve, recall_curve, f1_curve

    def predict(self, input_data=None):
        return input_data

    def instance_to_model(self):
        return EvaluateModel(self._parameter, self.get_party_info())

    @staticmethod
    def model_to_instance(model):
        algorithm_instance = Evaluate(model.parameter)
        algorithm_instance.set_party_info(*model.party_info)
        return algorithm_instance

    def save_metrics(self, metrics_dict):
        progress_value = list()
        for metric in metrics_dict:
            pv = dict()
            for k in metric:
                if isinstance(metric[k], float):
                    pv[k] = metric[k]
            progress_value.append(pv)

        SqliteManager.task_progress_dbo.create(dict(
            task_id=self._task_chain_id,
            progress_type=constant.BoardConstants.TASK_PROGRESS_METRICS,
            progress_value=json.dumps(progress_value)
        ))
