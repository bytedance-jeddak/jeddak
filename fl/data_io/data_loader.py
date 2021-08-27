import ast
import functools

from common.frame.data_frame.c_dataset import CDataset
from common.frame.data_frame.sample import Sample
from common.frame.data_frame.schema import Schema
from common.frame.message_frame.data_loader_message import DataLoaderMessage
from common.frame.model_frame.data_loader_model import DataLoaderModel
from common.frame.parameter_frame.data_loader_parameter import DataLoaderParameter
from common.util import constant
from fl.algorithm import Algorithm
import numpy as np


class DataLoader(Algorithm):
    """
    Data loader
    The first column is used as id_name.
    For the data with labels, the second column named 'y' is the label value.
    The rest columns are features.
    Data loader always outputs (id: str, Sample) despite that there might be no features.
    """

    def __init__(self, parameter: DataLoaderParameter, message=DataLoaderMessage()):
        super(DataLoader, self).__init__(parameter, message=message)
        self._sparse_index2dict = None

    def train(self, input_data=None, input_model=None, is_predicting=False):
        if self._parameter.train_data_path is not None and not is_predicting:
            data_path = self._parameter.train_data_path
        elif self._parameter.input_data_path is not None:
            data_path = self._parameter.input_data_path
        else:
            raise ValueError("'train_data_path' or 'input_data_path' should not be None")

        return self._process(data_path, input_data, input_model, is_predicting)

    def validate(self, input_data=None, evaluate=True):
        if self._parameter.validate_data_path is not None:
            return self._process(self._parameter.validate_data_path)[0], None
        return None, None

    def _process(self, data_path, input_data=None, _input_model=None, is_predicting=False):
        my_party_role = self.get_this_party_role()

        data_carrier = self._parameter.data_carrier
        if data_carrier != constant.DataCarrier.NdArray:
            raise ValueError("Unsupported Data Carrier: {}".format(data_carrier))
        output_data, total_header = self._load_to_ndarray(self._parameter.input_data_source, data_path)

        if my_party_role == constant.TaskRole.HOST or is_predicting:
            if len(total_header) > 1 and isinstance(total_header[1], str) and total_header[1].lower() == 'y':
                raise ValueError('predicting or host\'s dataset should not contain labels')
            id_name, label_name = total_header[0], None
            feature_names = [] if len(total_header) == 1 else total_header[1:]
        elif my_party_role == constant.TaskRole.GUEST and not is_predicting:
            if not (len(total_header) > 1 and isinstance(total_header[1], str) and total_header[1].lower() == 'y'):
                raise ValueError('guest\'s dataset must contain labels')
            id_name, label_name = total_header[0], total_header[1]
            feature_names = total_header[2:]
        else:
            raise ValueError('unknown party role: {}'.format(my_party_role))

        output_data.schema = Schema(id_name, feature_names, label_name)
        self._logger.info(output_data.schema)

        # parse each row to (id, sample)
        has_label = (label_name is not None)

        # for a training procedure
        if not is_predicting:
            # which columns are sparse-features?
            sparse_feature_index_list = []
            start = 2 if has_label else 1

            def is_number(s):
                if s is None:
                    return 0
                _, number = DataLoader.is_number(s)
                return 0 if number else 1

            # use rdd.fold to get aggregate information
            is_num_fold = output_data.map(
                lambda x: [is_number(xx) for xx in x]
            ).fold(
                [0 for _ in range(len(feature_names) + start)],
                lambda x, y: [x[i] + y[i] for i in range(len(x))]
            )
            for (idx, v) in enumerate(is_num_fold):
                if idx >= start and v > 0:
                    self._logger.info('column {} has {} sparse values'.format(idx, v))
                    sparse_feature_index_list.append(idx)

            self._logger.info("sparse_feature_index: {}".format(sparse_feature_index_list))
            # collect sparse features
            self._sparse_index2dict = dict()

            for sparse_feature_index in sparse_feature_index_list:
                sparse_values = output_data.map(lambda x: x[sparse_feature_index]).distinct().collect()
                self._logger.info("sparse_values of index {}: {}".format(sparse_feature_index, sparse_values))
                self._sparse_index2dict[sparse_feature_index] = {k: float(v) for v, k in enumerate(sparse_values)}

        # for a predicting procedure
        else:
            self._logger.info("sparse_index2dict exists: {}".format(self._sparse_index2dict))

        output_data = output_data.map(functools.partial(
            DataLoader.foreach_row_parse_partition,
            has_label=has_label,
            sparse_index2dict=self._sparse_index2dict,
            my_party_role=my_party_role,
            convert_sparse_to_index=self._parameter.convert_sparse_to_index),
            preserve_schema=True)

        # eliminate duplicate ids
        output_data = output_data.make_key_distinct()

        # update feature dimension
        output_data.update_feature_dimension()

        # construct output model
        output_model = self.instance_to_model(sparse_index2dict=self._sparse_index2dict)
        return output_data, output_model

    @staticmethod
    def _load_to_ndarray(data_source, data_path):
        if data_source == constant.DataSource.CSV:
            with open(data_path, 'r') as f:
                total_header = f.readline().strip().split(',')
                output_data = CDataset(np.loadtxt(f, str, delimiter=','))
                output_data = output_data.map(lambda row: list(row))
        else:
            raise ValueError("invalid data_io source: {}".format(data_source))
        return output_data, total_header

    @staticmethod
    def foreach_row_parse_partition(row, has_label, sparse_index2dict, my_party_role, convert_sparse_to_index):
        # parse id
        identifier = str(row.pop(0))

        # parse label
        if has_label:
            if isinstance(row[0], str):
                label = ast.literal_eval(row.pop(0).replace(' ', ','))
            else:
                label = np.float(row.pop(0))
            sparse_offset = 2
        else:
            label = None
            # guest lacks y column during predicting procedure
            sparse_offset = 2 if my_party_role == constant.TaskRole.GUEST else 1

        # parse features
        for i in range(len(row)):
            # Case.1 for None, just keep it
            if row[i] is None:
                continue
            elif isinstance(row[i], str):
                # Case.2 for sparse-feature: use 0,1,... to replace its value
                if convert_sparse_to_index and sparse_offset + i in sparse_index2dict:
                    row[i] = sparse_index2dict[sparse_offset + i][row[i]]
                else:
                    # Case.3 for numeric string, transform to float
                    float_number, is_num = DataLoader.is_number(row[i])
                    if is_num:
                        row[i] = float_number
                    else:
                        raise ValueError("Invalid sparse_index2dict key: %s, value: %s,\n sparse_index2dict: %s" %
                                         (str(sparse_offset + i), str(row[i]), str(sparse_index2dict)))
            # By default, transform to float
            else:
                row[i] = np.float(row[i])
        features = np.asarray(row)
        return identifier, Sample(features, label)

    def set_sparse_index2dict(self, sparse_index2dict):
        self._sparse_index2dict = sparse_index2dict

    @staticmethod
    def is_number(s):
        try:
            float_number = np.float(s)
            return float_number, True
        except ValueError:
            return None, False

    def set_input_data_source(self, input_data_source):
        self._parameter.input_data_source = input_data_source

    def set_input_data_path(self, input_data_path):
        self._parameter.input_data_path = input_data_path

    def predict(self, input_data=None):
        output_data, _ = self.train(input_data=input_data, is_predicting=True)
        return output_data

    def instance_to_model(self, sparse_index2dict=None):
        return DataLoaderModel(self._parameter, party_info=self.get_party_info(), sparse_index2dict=sparse_index2dict)

    @staticmethod
    def model_to_instance(model):
        algorithm_instance = DataLoader(model.parameter)
        algorithm_instance.set_party_info(*model.party_info)
        algorithm_instance.set_sparse_index2dict(model.sparse_index2dict)
        return algorithm_instance
