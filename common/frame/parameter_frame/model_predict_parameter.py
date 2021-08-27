from common.frame.parameter_frame.parameter import Parameter
from common.util import constant


class ModelPredictParameter(Parameter):
    def __init__(self,
                 task_role=constant.TaskRole.GUEST,
                 model_id=None,
                 input_data_path=None):
        """

        :param task_role: str
        :param model_id: str
        :param input_data_path: str
        """
        super(ModelPredictParameter, self).__init__(constant.TaskType.MODEL_PREDICT, task_role)

        self.model_id = model_id
        self.input_data_path = input_data_path
