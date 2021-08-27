from fl.operator.operator import Operator


class StatsMaker(Operator):
    def __init__(self, need_log=False):
        super(StatsMaker, self).__init__(need_log)

    def get_max(self, dataset, target_column_indices=None):
        """

        :param dataset: must be DDataset ('u0', Sample) with complete schema and feature_dimension
        :param target_column_indices: List[int]
        :return: [max_f0, max_f1, ...]
        """
        max_list = []

        if target_column_indices is None:
            target_column_indices = range(dataset.feature_dimension)

        for f_idx in target_column_indices:
            one_column_data = dataset.map(lambda row: row[1].features[f_idx]).filter_missing_value()
            max_val = one_column_data.max()
            max_list.append(max_val)

        return max_list

    def get_min(self, dataset, target_column_indices=None):
        """

        :param dataset: must be DDataset ('u0', Sample) with complete schema and feature_dimension
        :param target_column_indices: List[int]
        :return: [min_f0, min_f1, ...]
        """
        min_list = []

        if target_column_indices is None:
            target_column_indices = range(dataset.feature_dimension)

        for f_idx in target_column_indices:
            one_column_data = dataset.map(lambda row: row[1].features[f_idx]).filter_missing_value()
            min_val = one_column_data.min()
            min_list.append(min_val)

        return min_list

    def get_min_max(self, dataset, target_column_indices=None):
        """

        :param dataset:
        :param target_column_indices:
        :return: [(min_f0, max_f0), (min_f1, max_f1), ...]
        """
        min_max_list = []

        if target_column_indices is None:
            target_column_indices = range(dataset.feature_dimension)

        for f_idx in target_column_indices:
            one_column_data = dataset.map(lambda row: row[1].features[f_idx]).filter_missing_value()
            max_val = one_column_data.max()
            min_val = one_column_data.min()
            min_max_list.append((min_val, max_val))

        return min_max_list

    def get_average(self, dataset, target_column_indices=None):
        """

        :param dataset:
        :param target_column_indices:
        :return: [ave_f0, ave_f1, ...]
        """
        ave_list = []
        num_row = dataset.count()

        if target_column_indices is None:
            target_column_indices = range(dataset.feature_dimension)

        for f_idx in target_column_indices:
            one_column_data = dataset.map(lambda row: row[1].features[f_idx]).filter_missing_value()
            ave_list.append(one_column_data.sum() / num_row)

        return ave_list

    def get_bucket_split_points(self, dataset, num_buckets, target_column_indices=None):
        """

        :param dataset:
        :param num_buckets:
        :param target_column_indices:
        :return: [[min_f0, 1.5, 3.1, max_f0], ...]
        """
        min_max_list = self.get_min_max(dataset, target_column_indices)

        split_points_list = []
        for min_max_val in min_max_list:
            min_val, max_val = min_max_val
            total_length = max_val - min_val
            interval_length = total_length / num_buckets

            split_points = [min_val]
            for b_idx in range(num_buckets - 1):
                cur_val = split_points[b_idx]
                nxt_val = cur_val + interval_length
                split_points.append(nxt_val)
            split_points.append(max_val)

            split_points_list.append(split_points)

        return split_points_list
