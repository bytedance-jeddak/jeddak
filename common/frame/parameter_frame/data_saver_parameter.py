import os

from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class DataSaverParameter(Parameter):
    def __init__(self,
                 task_role=constant.TaskRole.GUEST,
                 output_data_source=constant.DataSource.CSV):
        """

        :param task_role: str
        :param output_data_source: str
        """
        super(DataSaverParameter, self).__init__(constant.TaskType.DATA_SAVER, task_role)

        self.output_data_source = output_data_source
