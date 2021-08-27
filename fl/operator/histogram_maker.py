import functools
from collections import OrderedDict

from common.frame.data_frame.histogram import Histogram
from common.util import constant
from fl.operator.operator import Operator


class HistogramMaker(Operator):
    def __init__(self, to_cumulative=True, need_log=False):
        """

        :param to_cumulative: fit to get cumulative histogram if true, and general histogram otherwise
        :param need_log: bool
        """
        super(HistogramMaker, self).__init__(need_log)

        self._to_cumulative = to_cumulative

    def fit(self, b_data, b_summary, targets):
        """
        Fit from bucketized DDataset ('u0', Sample) and DDataset ('u0', target0)
        :param b_data: DDataset ('u0', Sample)
        :param b_summary: OrderedDict {'x0': BucketSummary}
        :param targets: DDataset ('u0', target0)
        :return: OrderedDict {'x0': Histogram} or OrderedDict {'x0': CumulativeHistogram}
        """
        def get_feature_hist_for_each_partition(partition_data, num_bucket):
            """

            :param partition_data: ('u0', (Sample, target0))
            :param num_bucket: the number of buckets, e.g., OrderedDict {'x0': 33}
            :return: OrderedDict {'x0': Histogram}
            """
            # init feature_hist
            feature_hist = OrderedDict()
            for f_name, num in num_bucket.items():
                feature_hist[f_name] = Histogram.generate(
                    num, element_type=constant.DataFrameType.GH_PAIR)

            # parse partition data to obtain feature_hist
            for _, value in partition_data:
                cur_features = value[0].features
                cur_target = value[1]

                for f_idx, f_name in enumerate(num_bucket.keys()):
                    feature = cur_features[f_idx]
                    # check for float('NaN')
                    if feature and not (feature != feature):
                        bucket_idx = int(cur_features[f_idx])       # figure out the bucket index under the f_idx-th feature
                        feature_hist[f_name].grow(value=cur_target, index=bucket_idx)

            yield feature_hist

        def get_feature_hist_reduce(feature_hist_a, feature_hist_b):
            """

            :param feature_hist_a: OrderedDict {'x0': Histogram}
            :param feature_hist_b: OrderedDict {'x0': Histogram}
            :return:
            """
            feature_hist_agg = OrderedDict()

            for f_name in feature_hist_a.keys():
                hist_a = feature_hist_a[f_name]
                hist_b = feature_hist_b[f_name]
                feature_hist_agg[f_name] = hist_a + hist_b

            return feature_hist_agg

        # construct the number of buckets
        num_bucket = OrderedDict()
        for f_name, bs in b_summary.items():
            num_bucket[f_name] = bs.get_num_bucket()
        self._log("constructed the number of buckets")

        # append targets to the tail of b_data
        data_targets = b_data.join(targets)
        self._log("appended targets to the tail of b_data")

        # get histogram
        feature_hist = data_targets.mapPartitions(
            functools.partial(get_feature_hist_for_each_partition,
                              num_bucket=num_bucket)
        ).reduce(get_feature_hist_reduce)
        self._log("got histogram")

        # convert to cumulative histogram if applicable
        if self._to_cumulative:
            for f_name in feature_hist.keys():
                feature_hist[f_name] = feature_hist[f_name].to_cumulative_histogram()
            self._log("converted to cumulative histogram")

        return feature_hist


