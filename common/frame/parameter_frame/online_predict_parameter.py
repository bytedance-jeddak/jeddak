from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class OnlinePredictParameter(Parameter):
    def __init__(self,
                 task_role=constant.TaskRole.GUEST,
                 model_id=None,
                 input_data=None):
        """

        :param task_role: str
        :param model_id: str
        :param input_data_path: str
        :param output_data_path: str
        """
        super(OnlinePredictParameter, self).__init__(constant.TaskType.ONLINE_PREDICT, task_role)

        self.model_id = model_id
        self.input_data = input_data
