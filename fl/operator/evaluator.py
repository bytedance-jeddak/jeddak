
from common.util import constant

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

from fl.operator.operator import Operator


class Evaluator(Operator):
    def __init__(self, evaluation_type=constant.EvaluationType.BINARY, need_log=False):
        super(Evaluator, self).__init__(need_log)

        self._evaluation_type = evaluation_type

    def eval(self, pred_score, true_label):
        """
        :param pred_score: DDataset ((id, pred_score))
        :param true_label: DDataset ((id, true_label))
        :return:
        """
        y_agg = []
        true_label_list = []
        one_data = pred_score.first()

        if isinstance(one_data[1], list):
            for i in range(len(one_data[1])):
                true_label_i = true_label.mapValues(lambda row: row[i])
                pred_score_i = pred_score.mapValues(lambda row: row[i])
                true_label_list.append(true_label_i)
                # pred_score_list.append(pred_score_i)
                y_agg_i = self._preprocess_label(pred_score_i, true_label_i)
                y_agg.append(y_agg_i)
        else:
            y_agg_i = self._preprocess_label(pred_score, true_label)
            true_label_list.append(true_label)
            y_agg.append(y_agg_i)

        metrics_dict = {}
        metrics_dict_list = []

        if self._evaluation_type == constant.EvaluationType.BINARY:
            for i in range(len(y_agg)):
                pred_label = y_agg[i].map(lambda row: row[0])
                true_label = y_agg[i].map(lambda row: row[1])
                precision, recall, f1 = self._binary_precision_recall(true_label.collect(), pred_label.collect())
                metrics_dict[constant.Metric.PRECISION] = [('threshlod', 'precision')] + precision
                metrics_dict[constant.Metric.RECALL] = [('threshlod', 'recall')] + recall
                metrics_dict[constant.Metric.F1] = [('threshlod', 'f1', 'precision', 'recall')] + f1
                prediction_array, label_array = np.array(pred_label.rows), np.array(true_label.rows)
                try:
                    metrics_dict[constant.Metric.AUC] = roc_auc_score(label_array, prediction_array)
                    p, r, _ = precision_recall_curve(label_array, prediction_array)
                    metrics_dict[constant.Metric.AUPR] = auc(r, p)
                except ValueError:
                    metrics_dict[constant.Metric.AUC] = 0.5
                    metrics_dict[constant.Metric.AUPR] = 0.5
                metrics_dict_list.append(metrics_dict)

                metrics_dict_list.append(metrics_dict)
        else:
            raise ValueError("invalid evaluation type: {}".format(self._evaluation_type))

        return metrics_dict_list

    def _preprocess_label(self, pred_score, true_label):
        y_agg = pred_score.join(true_label).eliminate_key()

        if self._evaluation_type == constant.EvaluationType.BINARY:
            return y_agg.map_for_list(lambda row: (float(row[0]), float(row[1])))

        elif self._evaluation_type == constant.EvaluationType.MULTICLASS:
            return y_agg.map_for_list(lambda row: (float(np.round(row[0])), float(np.round(row[1]))))

        elif self._evaluation_type == constant.EvaluationType.REGRESSION:
            return y_agg.map_for_list(lambda row: (float(row[0]), float(row[1])))

        else:
            raise ValueError("invalid evaluation type: {}".format(self._evaluation_type))

    def _binary_clf_curve(self, y_true, y_score):
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
