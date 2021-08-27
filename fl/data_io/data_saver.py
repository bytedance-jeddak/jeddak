import os
import csv
import functools
from common.frame.message_frame.data_saver_message import DataSaverMessage
from common.frame.model_frame.data_saver_model import DataSaverModel
from common.frame.parameter_frame.data_saver_parameter import DataSaverParameter
from common.util import constant
from fl.algorithm import Algorithm


class DataSaver(Algorithm):
    """
    Data saver
    DataSaver saves DDataset object to CSV file
    """

    def __init__(self, parameter: DataSaverParameter, message=DataSaverMessage()):
        super(DataSaver, self).__init__(parameter, message=message)

    def train(self, input_data=None, input_model=None):
        # save data to csv file
        if self._parameter.output_data_source == constant.DataSource.CSV:
            # fill the output_data_path
            root_dir = os.path.split(os.path.realpath(__file__))[0]
            data_saver_dir = os.path.join(root_dir, '..', '..', 'common', 'data_saver', self._this_party)
            if not os.path.exists(data_saver_dir):
                os.makedirs(data_saver_dir)
            output_data_path = os.path.join(data_saver_dir, '{}.csv'.format(self._task_chain_id))
            self._logger.info('output_data_path: {}'.format(output_data_path))
            # construct the total header
            id_name = input_data.schema.id_name
            feature_names = input_data.schema.feature_names
            label_name = input_data.schema.label_name
            total_header = [id_name]
            if label_name is not None:
                total_header.append(label_name)
            if feature_names is not None:
                total_header += list(feature_names)
            # change from Sample to list
            has_label = (input_data.schema.label_name is not None)
            output_data = input_data.map(
                functools.partial(DataSaver.foreach_row_parse_partition, has_label=has_label), preserve_schema=True)
            with open(output_data_path, 'w', encoding='utf-8-sig') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(total_header)
                data_local_iterator = output_data.toLocalIterator()
                while True:
                    next_element = next(data_local_iterator, None)
                    if next_element is None:
                        break
                    else:
                        csv_write.writerow(next_element)
            return input_data, self.instance_to_model()
        else:
            raise ValueError("invalid data_io source: {}".format(self._parameter.output_data_source))

    @staticmethod
    def foreach_row_parse_partition(row, has_label):
        identifier, sample = row
        total_data = [identifier]
        if has_label:
            total_data.append(float(sample.label) if sample.label is not None else None)
        total_data += sample.features.tolist()
        return total_data

    def predict(self, input_data=None):
        output_data, _ = self.train(input_data=input_data)
        return output_data

    def instance_to_model(self):
        return DataSaverModel(self._parameter, self.get_party_info())

    @staticmethod
    def model_to_instance(model):
        algorithm_instance = DataSaver(model.parameter)
        algorithm_instance.set_party_info(*model.party_info)
        return algorithm_instance
