import os

from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class DataLoaderParameter(Parameter):
    def __init__(self,
                 task_role=constant.TaskRole.GUEST,
                 input_data_source=constant.DataSource.CSV,
                 input_data_path=None,
                 train_data_path=None,
                 validate_data_path=None,
                 input_model_path=None,
                 convert_sparse_to_index=True,
                 data_carrier=constant.DataCarrier.NdArray):
        """
        :param task_role: str
        :param input_data_source: str
        :param input_data_path: str
        :param input_model_path: str
        :param train_data_path: str, default None, train_data for training, if None, will get data from input_data_path.
        :param validate_data_path: str, default None, you can set data for validation
        :param convert_sparse_to_index: bool
        :param data_carrier: str
        """
        super(DataLoaderParameter, self).__init__(constant.TaskType.DATA_LOADER, task_role)

        self.input_data_source = input_data_source
        self.input_data_path = input_data_path
        self.train_data_path = train_data_path
        self.validate_data_path = validate_data_path
        self.input_model_path = input_model_path
        self.convert_sparse_to_index = convert_sparse_to_index
        self.data_carrier = data_carrier
