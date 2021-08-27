import math
from hashlib import md5

import numpy as np

from common.frame.data_frame.dataset import Dataset
from common.util import constant


def dataset_loader_initializer(input_data, mode, partitions=None):
    if mode == constant.NeuralNetwork.LOCAL:
        return LocalDatasetLoader(input_data)
    else:
        raise ValueError("Unknown mode:{}".format(mode))


class DatasetLoader(object):
    def __init__(self, data_set=None):
        # self.data_set = data_set

        self.data_set = data_set.sortByKey(
            keyfunc=lambda x: md5(str(x).encode('utf-8')).hexdigest())

    def release(self):
        pass


class LocalDatasetLoader(DatasetLoader):
    """
    provide input data for nn model
    """

    def __init__(self, data_set: Dataset = None) -> None:
        super().__init__(data_set)
        self.sample_size = data_set.count()

        self.input_with_index = self.data_set.zipWithIndex()

        # option 1: rdd persist
        self.input_with_index.persist()
        self.data_batches = []

        # option 2: the whole np array
        # self.data_np = self.dataset_to_np_arrays(self.input_with_index.map(lambda x: x[0]))

        self.ids = None

    def get_data_ids(self):
        if self.ids is None:
            ids = self.data_set.mapValues(lambda v: None).collect()
            self.ids = [[_id[0]] for _id in ids]

        return self.ids

    def get_sample_size(self):
        return self.sample_size

    def get_batches(self, batch_size):
        batches = math.ceil(self.sample_size / batch_size)
        return batches

    def get_train_batch(self, batch_size, idx, ratio=1):
        """
        param idx: idx of the batch
        batch_size:
        return: x, y
        """
        lower = batch_size * idx
        upper = lower + batch_size
        train_set = self.filter_from_range(self.input_with_index, lower,
                                           upper).map(lambda x: x[0])

        return self.dataset_to_np_arrays(train_set)


    @staticmethod
    def filter_from_range(input_data: Dataset, lower, upper):
        return input_data.filter(lambda x: lower <= x[1] < upper)

    def dataset_to_np_arrays(self, input_data):
        """
        param input_data: Dataset
        return: np_array x, y
        """
        train_x = input_data.map(lambda x: x[1].features)
        train_y = None

        if input_data.first()[1].label is not None:
            train_y = input_data.map(lambda x: x[1].label)
        if train_y:
            return self.get_nparray(train_x), self.get_nparray(train_y)
        else:
            return self.get_nparray(train_x), None

    @staticmethod
    def dataset_to_np_arrays_1(input_data):
        """
        param input_data: Dataset
        return: np_array id, x, y
        """

        if input_data.first()[1].label is not None:
            id_x_y = input_data.map(lambda x: (x[0], (x[1].features.tolist()), x[1].label)).collect()
            ids = [x[0] for x in id_x_y]
            x = [x[1] for x in id_x_y]
            y = [x[2] for x in id_x_y]
            return np.array(ids), np.array(x), np.array(y)

        else:
            id_x_y = input_data.map(lambda x: (x[0], (x[1].features.tolist()))).collect()
            ids = [x[0] for x in id_x_y]
            x = [x[1] for x in id_x_y]
            return np.array(ids), np.array(x), None

    @staticmethod
    def get_nparray(data):
        """
        sample to np array
        param rdd_data: input data, each row's value should be a np_array
        return: transformed np_array
        """
        return np.array(data.collect())

