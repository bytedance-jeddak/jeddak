from common.frame.data_frame.bucket_summary import BucketSummary
from common.util import constant
from collections import OrderedDict

from fl.operator.operator import Operator
from fl.operator.stats_maker import StatsMaker

import numpy as np
import pandas as pd


class Bucketizer(Operator):
    # if empty feature suffix is provided, use the following variable for temporary bucketization
    FEATURE_SUFFIX_TEMP = '_feature_suffix_temp'

    def __init__(self, bucketizer_type=constant.BucketizerType.QUANTILE,
                 num_bucket=30,
                 remove_original_columns=True,
                 output_sampleset=True,
                 feature_suffix='',
                 need_log=False):
        """
        :param bucketizer_type:
        :param num_bucket:
        :param remove_original_columns: whether to remove original columns
        :param output_sampleset: if True, then fit() outputs (id, Sample) and (id, f0, f1, ...) otherwise
        :param feature_suffix: new_feature_name = feature_name + feature_suffix, only in effect if output bytedataset
        :param need_log: bool
        """
        if not remove_original_columns and feature_suffix == '':
            raise ValueError("if original columns not removed, derived features must have different names")

        super(Bucketizer, self).__init__(need_log)

        self._bucketizer_type = bucketizer_type
        self._num_bucket = num_bucket
        self._remove_original_columns = remove_original_columns
        self._output_sampleset = output_sampleset
        self._feature_suffix = feature_suffix

    def fit(self, input_data, all_none_features_index=None, target_columns=None):
        return self._local_fit(input_data, all_none_features_index, target_columns)

    def _local_fit(self, input_data, all_none_features_index=None, target_columns=None):
        self._log("start local fitting")
        if len(input_data.schema.feature_names) == 0:
            return None, None
        data_frame = input_data.to_data_frame()
        if all_none_features_index:
            if not target_columns:
                target_columns = list(set(range(len(input_data.schema.feature_names))) - set(all_none_features_index))
            else:
                target_columns = list(set(target_columns) - set(all_none_features_index))
        target_feature_names = []
        if not target_columns:
            for i in range(len(input_data.schema.feature_names)):
                target_feature_names.append(input_data.schema.feature_names[i])
        else:
            for t_column in target_columns:
                target_feature_names.append(input_data.schema.feature_names[t_column])
            self._log("created target feature names: {}".format(target_feature_names))
        # create output_cols
        output_columns = []
        for f_name in target_feature_names:
            if self._feature_suffix == '':
                output_column = f_name + Bucketizer.FEATURE_SUFFIX_TEMP
            else:
                output_column = f_name + self._feature_suffix

            output_columns.append(output_column)
        if self._bucketizer_type == constant.BucketizerType.QUANTILE:
            self._log("perform quantile discretization")
            b_data, b_summary = self._qcut(data_frame=data_frame,
                                           input_columns=target_feature_names,
                                           output_columns=output_columns)
        elif self._bucketizer_type == constant.BucketizerType.BUCKET:
            self._log("perform bucket discretization")
            b_data, b_summary = self._cut(data_frame=data_frame,
                                          input_data=input_data,
                                          target_columns=target_columns,
                                          input_columns=target_feature_names,
                                          output_columns=output_columns)
        else:
            raise ValueError("invalid bucketizer_type: {}".format(self._bucketizer_type))
        if self._remove_original_columns:
            b_data.drop(target_feature_names, axis=1, inplace=True)
        input_data.from_data_frame(b_data)
        if not self._output_sampleset:
            input_data.map(lambda x: (x[0], x[1].features))
        return input_data, b_summary

    def _qcut(self, data_frame, input_columns, output_columns):
        summary = OrderedDict()
        for original_column, new_column in zip(input_columns, output_columns):
            data = pd.qcut(data_frame[original_column], self._num_bucket, duplicates='drop')
            intervals = data.dtype.categories
            if len(intervals):
                split_points = [intervals[0].left]
                for interval in intervals:
                    split_points.append(interval.right)
                summary[original_column] = BucketSummary(split_points)
                category_index_map = {c: i for i, c in enumerate(intervals)}
                data_frame[new_column] = data.map(category_index_map, na_action='ignore').astype(np.float)
            else:
                # all data are the same
                sample_value = data[0]
                data_frame[new_column] = data
                summary[original_column] = BucketSummary([sample_value])
        return data_frame, summary

    def _cut(self, data_frame, input_data, target_columns, input_columns, output_columns):
        split_points = StatsMaker().get_bucket_split_points(dataset=input_data,
                                                            num_buckets=self._num_bucket,
                                                            target_column_indices=target_columns)
        summary = {}
        for i in range(len(input_columns)):
            input_column, output_column, split = input_columns[i], output_columns[i], split_points[i]
            summary[input_column] = split
            data = pd.cut(data_frame[input_column], bins=split)
            category_index_map = {c: i for i, c in enumerate(data.dtype.categories)}
            data_frame[output_column] = data.map(category_index_map, na_action='ignore').astype(np.float)
        return data_frame, summary

